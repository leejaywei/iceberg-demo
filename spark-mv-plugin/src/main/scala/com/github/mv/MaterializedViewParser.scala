package com.github.mv

import org.apache.spark.sql.catalyst.parser.ParserInterface
import org.apache.spark.sql.catalyst.{FunctionIdentifier, TableIdentifier}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.types.{DataType, StructType}

/**
 * A delegating parser that intercepts materialized-view SQL commands
 * and forwards everything else to the underlying Spark SQL parser.
 *
 * Recognized commands:
 * {{{
 *   CREATE MATERIALIZED VIEW <name> AS <query>
 *   DROP MATERIALIZED VIEW <name>
 *   REFRESH MATERIALIZED VIEW <name>
 *   REFRESH MATERIALIZED VIEW <name> INCREMENTAL
 *   SHOW MATERIALIZED VIEWS
 * }}}
 */
class MaterializedViewParser(delegate: ParserInterface) extends ParserInterface {

  override def parsePlan(sqlText: String): LogicalPlan = {
    val trimmed = sqlText.trim
    val upper = trimmed.toUpperCase

    if (upper.startsWith("CREATE MATERIALIZED VIEW ")) {
      parseCreateMV(trimmed)
    } else if (upper.startsWith("DROP MATERIALIZED VIEW ")) {
      parseDropMV(trimmed)
    } else if (upper.startsWith("REFRESH MATERIALIZED VIEW ")) {
      parseRefreshMV(trimmed)
    } else if (upper == "SHOW MATERIALIZED VIEWS") {
      ShowMaterializedViewsCommand()
    } else {
      delegate.parsePlan(sqlText)
    }
  }

  private def parseCreateMV(sql: String): CreateMaterializedViewCommand = {
    // CREATE MATERIALIZED VIEW <name> AS <query>
    val asIndex = sql.toUpperCase.indexOf(" AS ")
    if (asIndex < 0) {
      throw new org.apache.spark.sql.AnalysisException(
        errorClass = "_LEGACY_ERROR_TEMP_0035",
        messageParameters = Map("message" ->
          "Expected 'AS' in CREATE MATERIALIZED VIEW statement"))
    }
    val nameAndPrefix = sql.substring("CREATE MATERIALIZED VIEW ".length, asIndex).trim
    val query = sql.substring(asIndex + 4).trim
    CreateMaterializedViewCommand(nameAndPrefix, query)
  }

  private def parseDropMV(sql: String): DropMaterializedViewCommand = {
    val name = sql.substring("DROP MATERIALIZED VIEW ".length).trim
    DropMaterializedViewCommand(name)
  }

  private def parseRefreshMV(sql: String): LogicalPlan = {
    val rest = sql.substring("REFRESH MATERIALIZED VIEW ".length).trim
    if (rest.toUpperCase.endsWith(" INCREMENTAL")) {
      val name = rest.substring(0, rest.length - " INCREMENTAL".length).trim
      RefreshMaterializedViewIncrementalCommand(name)
    } else {
      RefreshMaterializedViewCommand(rest)
    }
  }

  // ── delegate all other ParserInterface methods ──────────────────

  override def parseExpression(sqlText: String): Expression =
    delegate.parseExpression(sqlText)

  override def parseTableIdentifier(sqlText: String): TableIdentifier =
    delegate.parseTableIdentifier(sqlText)

  override def parseFunctionIdentifier(sqlText: String): FunctionIdentifier =
    delegate.parseFunctionIdentifier(sqlText)

  override def parseMultipartIdentifier(sqlText: String): Seq[String] =
    delegate.parseMultipartIdentifier(sqlText)

  override def parseTableSchema(sqlText: String): StructType =
    delegate.parseTableSchema(sqlText)

  override def parseDataType(sqlText: String): DataType =
    delegate.parseDataType(sqlText)

  override def parseQuery(sqlText: String): LogicalPlan =
    delegate.parseQuery(sqlText)
}
