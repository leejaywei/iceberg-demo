package com.github.readonly

import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.analysis.{ResolvedIdentifier, ResolvedTable}
import org.apache.spark.sql.catalyst.catalog.{CatalogTable, HiveTableRelation}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.connector.catalog.{Table, TableCatalog}
import org.apache.spark.sql.execution.command.{
  AlterTableAddColumnsCommand,
  AlterTableChangeColumnCommand,
  AlterTableRenameCommand,
  AlterTableSerDePropertiesCommand,
  AlterTableSetLocationCommand,
  AlterTableSetPropertiesCommand,
  AlterTableUnsetPropertiesCommand,
  DropTableCommand,
  TruncateTableCommand
}
import org.apache.spark.sql.execution.datasources.{
  InsertIntoDataSourceCommand,
  InsertIntoHadoopFsRelationCommand,
  LogicalRelation
}
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2Relation

/**
 * A check rule that prevents write and DDL operations on tables whose
 * properties contain `hive-ext.readOnly=true`.
 *
 * Covers:
 *  - V2 data writes: INSERT, INSERT OVERWRITE, DELETE, UPDATE, MERGE
 *  - V2 DDL: ALTER TABLE (add/drop/rename/alter columns, set/unset properties),
 *            DROP TABLE, RENAME TABLE, COMMENT ON TABLE
 *  - V1 data writes: INSERT via Hive / Hadoop FS / DataSource
 *  - V1 DDL: ALTER TABLE variants, DROP TABLE, TRUNCATE TABLE
 */
class ReadOnlyTableCheckRule(session: SparkSession) extends (LogicalPlan => Unit) {

  val READONLY_KEY = "hive-ext.readOnly"

  override def apply(plan: LogicalPlan): Unit = plan.foreach(checkNode)

  private def checkNode(node: LogicalPlan): Unit = node match {

    // ── V2 data write operations ──────────────────────────────────
    case AppendData(table, _, _, _, _, _) =>
      checkV2Plan(table, "INSERT")
    case OverwriteByExpression(table, _, _, _, _, _, _) =>
      checkV2Plan(table, "INSERT OVERWRITE")
    case OverwritePartitionsDynamic(table, _, _, _, _) =>
      checkV2Plan(table, "INSERT OVERWRITE DYNAMIC")
    case d: DeleteFromTable =>
      checkV2Plan(d.table, "DELETE")
    case u: UpdateTable =>
      checkV2Plan(u.table, "UPDATE")
    case m: MergeIntoTable =>
      checkV2Plan(m.targetTable, "MERGE")

    // ── V2 DDL operations ─────────────────────────────────────────
    case cmd: AlterTableCommand =>
      checkV2Plan(cmd.table, "ALTER TABLE")
    case d: DropTable =>
      checkV2Plan(d.child, "DROP TABLE")
    case r: RenameTable if !r.isView =>
      checkV2Plan(r.child, "RENAME TABLE")
    case c: CommentOnTable =>
      checkV2Plan(c.child, "COMMENT ON TABLE")

    // ── V1 data write operations ──────────────────────────────────
    case cmd: InsertIntoHadoopFsRelationCommand =>
      cmd.catalogTable.foreach(ct => checkV1CatalogTable(ct, "INSERT"))
    case cmd: InsertIntoDataSourceCommand =>
      cmd.logicalRelation.catalogTable.foreach(ct =>
        checkV1CatalogTable(ct, "INSERT"))

    // InsertIntoHiveTable (spark-hive): matched via HiveTableRelation child
    case node if isInsertIntoHiveTable(node) =>
      extractHiveTableRelation(node).foreach { htr =>
        checkV1CatalogTable(htr.tableMeta, "INSERT")
      }

    // ── V1 DDL operations ─────────────────────────────────────────
    case cmd: AlterTableAddColumnsCommand =>
      checkV1ByIdentifier(cmd.table, "ALTER TABLE ADD COLUMNS")
    case cmd: AlterTableChangeColumnCommand =>
      checkV1ByIdentifier(cmd.tableName, "ALTER TABLE CHANGE COLUMN")
    case cmd: AlterTableRenameCommand =>
      checkV1ByIdentifier(cmd.oldName, "ALTER TABLE RENAME")
    case cmd: AlterTableSetPropertiesCommand =>
      checkV1ByIdentifier(cmd.tableName, "ALTER TABLE SET PROPERTIES")
    case cmd: AlterTableUnsetPropertiesCommand =>
      checkV1ByIdentifier(cmd.tableName, "ALTER TABLE UNSET PROPERTIES")
    case cmd: AlterTableSetLocationCommand =>
      checkV1ByIdentifier(cmd.tableName, "ALTER TABLE SET LOCATION")
    case cmd: AlterTableSerDePropertiesCommand =>
      checkV1ByIdentifier(cmd.tableName, "ALTER TABLE SET SERDE")
    case cmd: TruncateTableCommand =>
      checkV1ByIdentifier(cmd.tableName, "TRUNCATE TABLE")
    case cmd: DropTableCommand =>
      checkV1ByIdentifier(cmd.tableName, "DROP TABLE")

    case _ => // no check needed
  }

  // ─── V2 helpers ─────────────────────────────────────────────────

  private def checkV2Plan(plan: LogicalPlan, operation: String): Unit = {
    plan match {
      case rel: DataSourceV2Relation =>
        checkV2Table(rel.table, rel.name, operation)
      case rt: ResolvedTable =>
        checkV2Table(rt.table, rt.name, operation)
      case ri: ResolvedIdentifier =>
        checkResolvedIdentifier(ri, operation)
      case _ =>
        plan.children.foreach(checkV2Plan(_, operation))
    }
  }

  private def checkV2Table(table: Table, name: String, operation: String): Unit = {
    val props = table.properties()
    if (props != null && "true".equalsIgnoreCase(props.get(READONLY_KEY))) {
      throwReadOnly(name, operation)
    }
  }

  private def checkResolvedIdentifier(ri: ResolvedIdentifier, operation: String): Unit = {
    ri.catalog match {
      case tc: TableCatalog =>
        val table = try {
          tc.loadTable(ri.identifier)
        } catch {
          case _: Exception => null // table may not exist yet, skip
        }
        if (table != null) {
          checkV2Table(table, ri.identifier.toString, operation)
        }
      case _ => // not a table catalog
    }
  }

  // ─── V1 helpers ─────────────────────────────────────────────────

  private def checkV1CatalogTable(ct: CatalogTable, operation: String): Unit = {
    if (ct.properties.get(READONLY_KEY).exists(_.equalsIgnoreCase("true"))) {
      throwReadOnly(ct.identifier.unquotedString, operation)
    }
  }

  private def checkV1ByIdentifier(
      name: TableIdentifier,
      operation: String): Unit = {
    val catalog = session.sessionState.catalog
    if (catalog.tableExists(name)) {
      checkV1CatalogTable(catalog.getTableMetadata(name), operation)
    }
  }

  // ─── Hive helpers (avoid hard compile-time dep on spark-hive) ──

  private def isInsertIntoHiveTable(node: LogicalPlan): Boolean =
    node.getClass.getName == "org.apache.spark.sql.hive.execution.InsertIntoHiveTable"

  private def extractHiveTableRelation(node: LogicalPlan): Option[HiveTableRelation] =
    node.children.collectFirst { case htr: HiveTableRelation => htr }

  // ─── Error ──────────────────────────────────────────────────────

  private def throwReadOnly(tableName: String, operation: String): Nothing = {
    val msg = s"Table $tableName is read-only (hive-ext.readOnly=true), $operation is not allowed"
    throw new AnalysisException(
      errorClass = "_LEGACY_ERROR_TEMP_0035",
      messageParameters = Map("message" -> msg))
  }
}
