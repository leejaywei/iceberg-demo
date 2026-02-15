package com.github.mv

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.command.LeafRunnableCommand

/**
 * A post-hoc resolution rule that rewrites SELECT queries to read from a
 * materialized view's backing table when the query matches the MV's
 * defining query exactly.
 *
 * Uses a thread-local guard to prevent:
 *  - infinite recursion when calling the analyzer from within this rule
 *  - rewriting during MV refresh operations (which re-execute the
 *    defining query and need to read from the original source tables)
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
          case _: Exception => // skip MVs whose query can't be parsed/analyzed
        }
      }
      plan
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
