package com.github.mv

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.command.LeafRunnableCommand

/**
 * A post-hoc resolution rule that rewrites SELECT queries to read from a
 * materialized view's backing table.
 *
 * Supports three levels of rewriting (inspired by StarRocks MV capabilities):
 *
 *  1. '''Exact match''' – the query's canonical plan is identical to the
 *     MV's defining query, so the entire plan is replaced by a scan of the
 *     backing table.
 *
 *  2. '''Aggregate rewrite''' – the query contains aggregations whose GROUP
 *     BY columns and functions can be served directly by the MV.
 *
 *  3. '''Roll-up (上卷) rewrite''' – the query groups by a subset of the
 *     MV's GROUP BY columns; the rule emits a re-aggregation on top of
 *     the backing table (SUM→SUM, COUNT→SUM, MIN→MIN, MAX→MAX,
 *     AVG→SUM(sum)/SUM(count)).
 *
 * Additionally supports:
 *  - '''Predicate compensation''' – extra WHERE conditions in the query are
 *    applied on top of the MV scan.
 *  - '''Column subset''' – the query may select fewer columns than the MV.
 *
 * Uses a thread-local guard to prevent:
 *  - infinite recursion when calling the analyzer from within this rule
 *  - rewriting during MV refresh operations
 */
class MaterializedViewOptimizationRule(session: SparkSession)
    extends Rule[LogicalPlan] {

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (MaterializedViewOptimizationRule.isDisabled.get()) return plan
    if (!plan.resolved) return plan
    plan match {
      case _: LeafRunnableCommand => return plan
      case _ =>
    }

    val mvs = MaterializedViewCatalog.listAll()
    if (mvs.isEmpty) return plan

    MaterializedViewOptimizationRule.isDisabled.set(true)
    try {
      // ── 1. Exact canonical match ────────────────────────────────
      mvs.foreach { meta =>
        try {
          val mvPlan = session.sessionState.sqlParser.parsePlan(meta.query)
          val analyzedMvPlan = session.sessionState.analyzer.execute(mvPlan)
          if (plan.canonicalized == analyzedMvPlan.canonicalized) {
            val backingPlan = session.sessionState.sqlParser
              .parsePlan(s"SELECT * FROM ${meta.backingTable}")
            val resolvedBacking = session.sessionState.analyzer.execute(backingPlan)
            return resolvedBacking
          }
        } catch {
          case _: Exception =>
        }
      }

      // ── 2. Aggregate / roll-up rewrite ─────────────────────────
      AggregateRewriter.tryRewrite(session, plan).getOrElse(plan)

    } finally {
      MaterializedViewOptimizationRule.isDisabled.set(false)
    }
  }
}

object MaterializedViewOptimizationRule {
  /** Thread-local flag to disable rewriting (prevents recursion and refresh loops). */
  val isDisabled: ThreadLocal[Boolean] =
    ThreadLocal.withInitial(() => false)
}
