package com.github.mv

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.plans.logical._

/**
 * Matches aggregate queries against materialized views and rewrites them
 * to read from the MV's backing table.  Inspired by StarRocks' MV rewrite
 * capabilities, this supports:
 *
 *  1. '''Aggregate rewrite''' – a query whose GROUP BY columns and aggregate
 *     functions are an exact match of the MV can be redirected to the backing
 *     table.
 *
 *  2. '''Roll-up (上卷) rewrite''' – a query that groups by a ''subset'' of
 *     the MV's GROUP BY columns can be served by re-aggregating the MV data.
 *     SUM and COUNT are re-summed, MIN/MAX stay as-is, and AVG is recomputed
 *     from SUM/COUNT columns stored in the MV.
 *
 *  3. '''Predicate compensation''' – if the query's WHERE condition is a
 *     conjunction (AND) of predicates that is a ''superset'' of the MV's
 *     predicates, the extra predicates are applied on top of the MV scan.
 *
 *  4. '''Column subset selection''' – the query may select fewer columns
 *     than the MV stores; the rule emits a Project to prune them.
 */
object AggregateRewriter {

  /**
   * Attempt to rewrite `queryPlan` using one of the registered MVs.
   * Returns `Some(rewrittenPlan)` on success, `None` if no MV applies.
   */
  def tryRewrite(
      session: SparkSession,
      queryPlan: LogicalPlan): Option[LogicalPlan] = {

    val mvs = MaterializedViewCatalog.listAll()
    if (mvs.isEmpty) return None

    mvs.foreach { meta =>
      try {
        val mvParsed = session.sessionState.sqlParser.parsePlan(meta.query)
        val mvAnalyzed = session.sessionState.analyzer.execute(mvParsed)

        val result = tryRewriteWithMV(session, queryPlan, mvAnalyzed, meta)
        if (result.isDefined) return result
      } catch {
        case _: Exception => // skip
      }
    }
    None
  }

  // ─────────────────────────────────────────────────────────────────
  //  Core matching
  // ─────────────────────────────────────────────────────────────────

  private def tryRewriteWithMV(
      session: SparkSession,
      queryPlan: LogicalPlan,
      mvPlan: LogicalPlan,
      meta: MaterializedViewMeta): Option[LogicalPlan] = {

    // Decompose both plans
    val queryInfo = decompose(queryPlan)
    val mvInfo    = decompose(mvPlan)
    if (queryInfo.isEmpty || mvInfo.isEmpty) return None

    val qi = queryInfo.get
    val mi = mvInfo.get

    // The base tables must be the same (compare canonicalized child)
    if (!sameBaseTable(qi.baseTable, mi.baseTable)) return None

    // ── Predicate compensation ─────────────────────────────────────
    val extraPreds = predicateCompensation(qi.predicates, mi.predicates)
    if (extraPreds.isEmpty) return None

    // ── GROUP BY match / roll-up ───────────────────────────────────
    val queryGroupNames = qi.groupByNames
    val mvGroupNames    = mi.groupByNames

    val isExactGroupBy = queryGroupNames == mvGroupNames
    val isRollUp       = queryGroupNames.nonEmpty &&
                         queryGroupNames.subsetOf(mvGroupNames)

    if (!isExactGroupBy && !isRollUp) return None

    // ── Aggregate function compatibility ───────────────────────────
    // Map MV output column name -> (aggFuncName, source column name)
    val mvAggMap = buildAggMap(mi)
    // For roll-up we also need a map of MV group columns
    val mvGroupColNames = mi.groupByNames

    // Build the backing-table scan
    val backingScan = session.sessionState.sqlParser
      .parsePlan(s"SELECT * FROM ${meta.backingTable}")
    val resolvedBacking = session.sessionState.analyzer.execute(backingScan)

    val backingOutput = resolvedBacking.output
    // Map MV output name -> backing table attribute
    val backingAttrMap: Map[String, Attribute] =
      backingOutput.map(a => a.name.toLowerCase -> a).toMap

    if (isExactGroupBy) {
      // ── Exact aggregate match ──────────────────────────────────
      tryExactAggRewrite(session, qi, mi, resolvedBacking, backingAttrMap, extraPreds.get)
    } else {
      // ── Roll-up rewrite ────────────────────────────────────────
      tryRollUpRewrite(session, qi, mi, mvAggMap, resolvedBacking, backingAttrMap, extraPreds.get)
    }
  }

  // ─────────────────────────────────────────────────────────────────
  //  Exact aggregate rewrite
  // ─────────────────────────────────────────────────────────────────

  private def tryExactAggRewrite(
      session: SparkSession,
      qi: PlanInfo,
      mi: PlanInfo,
      backingPlan: LogicalPlan,
      backingAttrMap: Map[String, Attribute],
      extraPreds: Seq[Expression]): Option[LogicalPlan] = {

    // Every query output must be present in the MV output
    val queryOutNames = qi.outputNames
    val mvOutNames    = mi.outputNames

    if (!queryOutNames.subsetOf(mvOutNames)) return None

    // Build projection list from backing table
    val projectExprs = qi.outputNames.toSeq.flatMap { name =>
      backingAttrMap.get(name).map(a => Alias(a, name)(exprId = NamedExpression.newExprId))
    }
    if (projectExprs.size != queryOutNames.size) return None

    var result: LogicalPlan = backingPlan

    // Apply extra predicates (predicate compensation)
    if (extraPreds.nonEmpty) {
      val compensated = rewritePredicateRefs(extraPreds, backingAttrMap)
      if (compensated.isEmpty) return None
      result = Filter(compensated.get.reduceLeft(And), result)
    }

    // Project to match query output
    Some(Project(projectExprs, result))
  }

  // ─────────────────────────────────────────────────────────────────
  //  Roll-up rewrite
  // ─────────────────────────────────────────────────────────────────

  private def tryRollUpRewrite(
      session: SparkSession,
      qi: PlanInfo,
      mi: PlanInfo,
      mvAggMap: Map[String, (String, String)],
      backingPlan: LogicalPlan,
      backingAttrMap: Map[String, Attribute],
      extraPreds: Seq[Expression]): Option[LogicalPlan] = {

    var result: LogicalPlan = backingPlan

    // Apply extra predicates before re-aggregating
    if (extraPreds.nonEmpty) {
      val compensated = rewritePredicateRefs(extraPreds, backingAttrMap)
      if (compensated.isEmpty) return None
      result = Filter(compensated.get.reduceLeft(And), result)
    }

    // Build new GROUP BY referencing backing-table columns
    val newGroupExprs: Seq[Expression] = qi.groupByNames.toSeq.flatMap { name =>
      backingAttrMap.get(name)
    }
    if (newGroupExprs.size != qi.groupByNames.size) return None

    // Build new aggregate expressions
    val newAggExprs = scala.collection.mutable.ArrayBuffer[NamedExpression]()

    // First add group-by columns
    qi.groupByNames.toSeq.foreach { name =>
      backingAttrMap.get(name).foreach { attr =>
        newAggExprs += Alias(attr, name)(exprId = NamedExpression.newExprId)
      }
    }

    // Then map each query aggregate to a roll-up over the MV column
    for ((queryAggName, queryAggFunc, querySourceCol) <- qi.aggregateDetails) {
      // Find matching MV column
      val mvColOpt = mvAggMap.collectFirst {
        case (mvColName, (mvFunc, mvSourceCol))
          if mvSourceCol == querySourceCol && canRollUp(mvFunc, queryAggFunc) =>
          mvColName
      }

      mvColOpt match {
        case Some(mvColName) =>
          val backingAttr = backingAttrMap.get(mvColName)
          if (backingAttr.isEmpty) return None

          val rollUpExpr = buildRollUpExpression(queryAggFunc, backingAttr.get, mvAggMap, backingAttrMap)
          if (rollUpExpr.isEmpty) return None

          newAggExprs += Alias(rollUpExpr.get, queryAggName)(exprId = NamedExpression.newExprId)

        case None =>
          // Check if query asks for AVG and MV has both SUM and COUNT
          if (queryAggFunc.equalsIgnoreCase("avg")) {
            val sumOpt = mvAggMap.collectFirst {
              case (mvColName, ("sum", sc)) if sc == querySourceCol => mvColName
            }
            val countOpt = mvAggMap.collectFirst {
              case (mvColName, ("count", sc)) if sc == querySourceCol => mvColName
            }
            if (sumOpt.isDefined && countOpt.isDefined) {
              val sumAttr = backingAttrMap.get(sumOpt.get)
              val cntAttr = backingAttrMap.get(countOpt.get)
              if (sumAttr.isEmpty || cntAttr.isEmpty) return None
              // AVG = SUM(sum_col) / SUM(count_col)
              val avgExpr = Divide(
                Sum(sumAttr.get).toAggregateExpression(),
                Sum(cntAttr.get).toAggregateExpression())
              newAggExprs += Alias(avgExpr, queryAggName)(exprId = NamedExpression.newExprId)
            } else {
              return None
            }
          } else {
            return None
          }
      }
    }

    Some(Aggregate(newGroupExprs, newAggExprs.toSeq, result))
  }

  // ─────────────────────────────────────────────────────────────────
  //  Helper: can a function be rolled up?
  // ─────────────────────────────────────────────────────────────────

  private def canRollUp(mvFunc: String, queryFunc: String): Boolean = {
    (mvFunc.toLowerCase, queryFunc.toLowerCase) match {
      case ("sum", "sum")     => true
      case ("count", "count") => true  // COUNT rolls up via SUM
      case ("min", "min")     => true
      case ("max", "max")     => true
      case _                  => false
    }
  }

  private def buildRollUpExpression(
      queryFunc: String,
      mvColAttr: Attribute,
      mvAggMap: Map[String, (String, String)],
      backingAttrMap: Map[String, Attribute]): Option[Expression] = {
    queryFunc.toLowerCase match {
      case "sum"   => Some(Sum(mvColAttr).toAggregateExpression())
      case "count" => Some(Sum(mvColAttr).toAggregateExpression())  // roll up count by summing
      case "min"   => Some(Min(mvColAttr).toAggregateExpression())
      case "max"   => Some(Max(mvColAttr).toAggregateExpression())
      case _       => None
    }
  }

  // ─────────────────────────────────────────────────────────────────
  //  Plan decomposition
  // ─────────────────────────────────────────────────────────────────

  /**
   * Decompose a logical plan into its structural components:
   * base table, predicates, group-by columns, aggregate functions, output columns.
   */
  private def decompose(plan: LogicalPlan): Option[PlanInfo] = {
    plan match {
      // Aggregate over optional Filter over base relation
      case agg @ Aggregate(groupExprs, aggExprs, child) =>
        val (preds, baseTable) = extractFilterAndBase(child)
        val groupByNames = groupExprs.flatMap(exprColumnNames).toSet
        val aggregates = extractAggregates(aggExprs)
        val aggDetails = extractAggregateDetails(aggExprs)
        val outputNames = aggExprs.flatMap {
          case a: Alias => Some(a.name.toLowerCase)
          case a: Attribute => Some(a.name.toLowerCase)
          case _ => None
        }.toSet

        Some(PlanInfo(baseTable, preds, groupByNames, aggregates, aggDetails, outputNames))

      // Project over Aggregate
      case Project(projectList, agg @ Aggregate(_, _, _)) =>
        decompose(agg).map { info =>
          val projNames = projectList.flatMap {
            case a: Alias => Some(a.name.toLowerCase)
            case a: Attribute => Some(a.name.toLowerCase)
            case _ => None
          }.toSet
          info.copy(outputNames = projNames)
        }

      // Non-aggregate: simple projection with optional filter
      case Project(projectList, child) =>
        val (preds, baseTable) = extractFilterAndBase(child)
        val outputNames = projectList.flatMap {
          case a: Alias => Some(a.name.toLowerCase)
          case a: Attribute => Some(a.name.toLowerCase)
          case _ => None
        }.toSet
        Some(PlanInfo(baseTable, preds, Set.empty, Seq.empty, Seq.empty, outputNames))

      case _ => None
    }
  }

  /**
   * Walk through Filter nodes to the base relation, collecting predicates.
   */
  private def extractFilterAndBase(plan: LogicalPlan): (Seq[Expression], LogicalPlan) = {
    plan match {
      case Filter(cond, child) =>
        val (childPreds, base) = extractFilterAndBase(child)
        (splitConjunction(cond) ++ childPreds, base)
      case Project(_, child) =>
        extractFilterAndBase(child)
      case other =>
        (Seq.empty, other)
    }
  }

  /**
   * Split an AND expression into its conjuncts.
   */
  private def splitConjunction(expr: Expression): Seq[Expression] = {
    expr match {
      case And(left, right) => splitConjunction(left) ++ splitConjunction(right)
      case other            => Seq(other)
    }
  }

  /**
   * Compare two base tables by their canonicalized form.
   */
  private def sameBaseTable(a: LogicalPlan, b: LogicalPlan): Boolean = {
    try {
      a.canonicalized == b.canonicalized
    } catch {
      case _: Exception => false
    }
  }

  /**
   * Check predicate compensation.
   *
   * Returns `Some(extraPredicates)` where `extraPredicates` are predicates in
   * the query that are NOT in the MV (may be empty if an exact match).
   * Returns `None` if the MV has predicates the query doesn't, meaning the
   * MV data is more restrictive than what the query needs.
   */
  private def predicateCompensation(
      queryPreds: Seq[Expression],
      mvPreds: Seq[Expression]): Option[Seq[Expression]] = {
    // All MV predicates must be present in the query predicates
    val qCanon = queryPreds.map(_.canonicalized).toSet
    val mCanon = mvPreds.map(_.canonicalized).toSet

    if (!mCanon.subsetOf(qCanon)) return None

    // Extra predicates = query preds that are NOT in MV
    val extra = queryPreds.filter(p => !mCanon.contains(p.canonicalized))
    Some(extra)
  }

  /**
   * Rewrite predicate attribute references to point to backing-table attributes.
   */
  private def rewritePredicateRefs(
      preds: Seq[Expression],
      backingAttrMap: Map[String, Attribute]): Option[Seq[Expression]] = {
    val rewritten = preds.map { pred =>
      pred.transform {
        case a: AttributeReference =>
          backingAttrMap.getOrElse(a.name.toLowerCase, a)
      }
    }
    Some(rewritten)
  }

  /**
   * Extract column names from an expression.
   */
  private def exprColumnNames(expr: Expression): Seq[String] = {
    expr match {
      case a: AttributeReference => Seq(a.name.toLowerCase)
      case _                     => expr.children.flatMap(exprColumnNames)
    }
  }

  /**
   * Extract aggregate function info from aggregate expressions.
   * Returns a list of (aggregateFunctionName, sourceColumnName).
   */
  private def extractAggregates(
      exprs: Seq[NamedExpression]): Seq[(String, String)] = {
    exprs.flatMap {
      case Alias(AggregateExpression(fn, _, _, _, _), _) =>
        fn match {
          case Sum(child, _)     => exprColumnNames(child).headOption.map(c => ("sum", c))
          case Count(children)   => children.flatMap(exprColumnNames).headOption.map(c => ("count", c))
          case Min(child)        => exprColumnNames(child).headOption.map(c => ("min", c))
          case Max(child)        => exprColumnNames(child).headOption.map(c => ("max", c))
          case Average(child, _) => exprColumnNames(child).headOption.map(c => ("avg", c))
          case _                 => None
        }
      case _ => None
    }
  }

  /**
   * Extract aggregate details including output name.
   * Returns a list of (outputAlias, aggregateFunctionName, sourceColumnName).
   */
  private def extractAggregateDetails(
      exprs: Seq[NamedExpression]): Seq[(String, String, String)] = {
    exprs.flatMap {
      case Alias(AggregateExpression(fn, _, _, _, _), aliasName) =>
        fn match {
          case Sum(child, _)     => exprColumnNames(child).headOption.map(c => (aliasName.toLowerCase, "sum", c))
          case Count(children)   => children.flatMap(exprColumnNames).headOption.map(c => (aliasName.toLowerCase, "count", c))
          case Min(child)        => exprColumnNames(child).headOption.map(c => (aliasName.toLowerCase, "min", c))
          case Max(child)        => exprColumnNames(child).headOption.map(c => (aliasName.toLowerCase, "max", c))
          case Average(child, _) => exprColumnNames(child).headOption.map(c => (aliasName.toLowerCase, "avg", c))
          case _                 => None
        }
      case _ => None
    }
  }

  /**
   * Build a map of MV output column name -> (function name, source column name).
   */
  private def buildAggMap(mi: PlanInfo): Map[String, (String, String)] = {
    mi.aggregateDetails.map { case (alias, func, srcCol) =>
      alias -> (func, srcCol)
    }.toMap
  }

  /**
   * Structural info extracted from a logical plan.
   */
  case class PlanInfo(
      baseTable: LogicalPlan,
      predicates: Seq[Expression],
      groupByNames: Set[String],
      aggregates: Seq[(String, String)],           // (funcName, sourceCol)
      aggregateDetails: Seq[(String, String, String)], // (alias, funcName, sourceCol)
      outputNames: Set[String])
}
