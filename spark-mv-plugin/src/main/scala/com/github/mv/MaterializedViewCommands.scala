package com.github.mv

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.execution.command.LeafRunnableCommand
import org.apache.spark.sql.types.{LongType, StringType}

/**
 * CREATE MATERIALIZED VIEW name AS query
 *
 * Materializes the query result into a backing Hive table and registers
 * metadata in [[MaterializedViewCatalog]].
 */
case class CreateMaterializedViewCommand(viewName: String, query: String)
    extends LeafRunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {
    if (MaterializedViewCatalog.exists(viewName)) {
      throw new org.apache.spark.sql.AnalysisException(
        errorClass = "_LEGACY_ERROR_TEMP_0035",
        messageParameters = Map("message" ->
          s"Materialized view $viewName already exists"))
    }

    val backingTable = s"mv_backing_${viewName.replace('.', '_')}"
    val df = sparkSession.sql(query)
    df.write.mode("overwrite").saveAsTable(backingTable)

    MaterializedViewCatalog.register(MaterializedViewMeta(
      name = viewName,
      query = query,
      backingTable = backingTable,
      lastRefreshTs = System.currentTimeMillis()
    ))
    Nil
  }

  override def simpleString(maxFields: Int): String =
    s"CreateMaterializedView $viewName"
}

/**
 * DROP MATERIALIZED VIEW name
 *
 * Drops the backing table and removes metadata from [[MaterializedViewCatalog]].
 */
case class DropMaterializedViewCommand(viewName: String)
    extends LeafRunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {
    MaterializedViewCatalog.get(viewName) match {
      case Some(meta) =>
        sparkSession.sql(s"DROP TABLE IF EXISTS ${meta.backingTable}")
        MaterializedViewCatalog.drop(viewName)
      case None =>
        throw new org.apache.spark.sql.AnalysisException(
          errorClass = "_LEGACY_ERROR_TEMP_0035",
          messageParameters = Map("message" ->
            s"Materialized view $viewName does not exist"))
    }
    Nil
  }

  override def simpleString(maxFields: Int): String =
    s"DropMaterializedView $viewName"
}

/**
 * REFRESH MATERIALIZED VIEW name
 *
 * Full refresh: re-executes the defining query and overwrites the
 * backing table with the new result.
 */
case class RefreshMaterializedViewCommand(viewName: String)
    extends LeafRunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {
    MaterializedViewCatalog.get(viewName) match {
      case Some(meta) =>
        MaterializedViewOptimizationRule.isDisabled.set(true)
        try {
          val df = sparkSession.sql(meta.query)
          df.write.mode("overwrite").saveAsTable(meta.backingTable)
          MaterializedViewCatalog.updateRefreshTs(viewName, System.currentTimeMillis())
        } finally {
          MaterializedViewOptimizationRule.isDisabled.set(false)
        }
      case None =>
        throw new org.apache.spark.sql.AnalysisException(
          errorClass = "_LEGACY_ERROR_TEMP_0035",
          messageParameters = Map("message" ->
            s"Materialized view $viewName does not exist"))
    }
    Nil
  }

  override def simpleString(maxFields: Int): String =
    s"RefreshMaterializedView $viewName"
}

/**
 * REFRESH MATERIALIZED VIEW name INCREMENTAL
 *
 * Incremental refresh: in a production system this would track source-table
 * changes and apply only the delta.  Here we demonstrate the pattern with
 * a full re-computation (overwrite) as a fallback.
 */
case class RefreshMaterializedViewIncrementalCommand(viewName: String)
    extends LeafRunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {
    MaterializedViewCatalog.get(viewName) match {
      case Some(meta) =>
        MaterializedViewOptimizationRule.isDisabled.set(true)
        try {
          val df = sparkSession.sql(meta.query)
          df.write.mode("overwrite").saveAsTable(meta.backingTable)
          MaterializedViewCatalog.updateRefreshTs(viewName, System.currentTimeMillis())
        } finally {
          MaterializedViewOptimizationRule.isDisabled.set(false)
        }
      case None =>
        throw new org.apache.spark.sql.AnalysisException(
          errorClass = "_LEGACY_ERROR_TEMP_0035",
          messageParameters = Map("message" ->
            s"Materialized view $viewName does not exist"))
    }
    Nil
  }

  override def simpleString(maxFields: Int): String =
    s"RefreshMaterializedViewIncremental $viewName"
}

/**
 * SHOW MATERIALIZED VIEWS
 *
 * Returns all registered materialized views with their metadata.
 */
case class ShowMaterializedViewsCommand()
    extends LeafRunnableCommand {

  override val output: Seq[Attribute] = Seq(
    AttributeReference("name", StringType, nullable = false)(),
    AttributeReference("query", StringType, nullable = false)(),
    AttributeReference("backing_table", StringType, nullable = false)(),
    AttributeReference("last_refresh_ts", LongType, nullable = false)()
  )

  override def run(sparkSession: SparkSession): Seq[Row] = {
    MaterializedViewCatalog.listAll().map { meta =>
      Row(meta.name, meta.query, meta.backingTable, meta.lastRefreshTs)
    }
  }

  override def simpleString(maxFields: Int): String = "ShowMaterializedViews"
}
