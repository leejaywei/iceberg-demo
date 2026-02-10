package com.github.readonly

import java.io.File
import java.nio.file.Files

import org.apache.spark.sql.{AnalysisException, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/**
 * Comprehensive unit tests for [[ReadOnlyTableCheckRule]].
 *
 * Test matrix:
 *   V1 Hive tables  – INSERT, INSERT OVERWRITE, ALTER (columns/properties/location/rename),
 *                      DROP, TRUNCATE, SELECT (allowed), non-read-only (allowed)
 *   V2 tables       – INSERT, DELETE, UPDATE, MERGE, ALTER (columns/properties),
 *                      DROP, SELECT (allowed), non-read-only (allowed)
 */
class ReadOnlyTableSuite extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _
  private var warehouseDir: File = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    warehouseDir = Files.createTempDirectory("readonly-test-warehouse").toFile
    spark = SparkSession.builder()
      .master("local[2]")
      .config("spark.sql.extensions", classOf[ReadOnlyTableExtensions].getName)
      .config("spark.sql.catalog.testcat",
        "org.apache.spark.sql.connector.catalog.InMemoryTableCatalog")
      .config("spark.sql.warehouse.dir", warehouseDir.getAbsolutePath)
      .config("spark.ui.enabled", "false")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .enableHiveSupport()
      .getOrCreate()

    // ── V1 (Hive) tables ────────────────────────────────────────
    spark.sql("CREATE DATABASE IF NOT EXISTS ro_test")
    spark.sql(
      """CREATE TABLE ro_test.readonly_v1 (id INT, name STRING)
        |STORED AS PARQUET
        |TBLPROPERTIES ('hive-ext.readOnly' = 'true')""".stripMargin)
    spark.sql(
      """CREATE TABLE ro_test.normal_v1 (id INT, name STRING)
        |STORED AS PARQUET""".stripMargin)

    // ── V2 tables (testcat) ─────────────────────────────────────
    spark.sql(
      """CREATE TABLE testcat.ns.readonly_v2 (id INT, name STRING)
        |TBLPROPERTIES ('hive-ext.readOnly' = 'true')""".stripMargin)
    spark.sql(
      """CREATE TABLE testcat.ns.normal_v2 (id INT, name STRING)""".stripMargin)
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.sql("DROP TABLE IF EXISTS ro_test.readonly_v1")
        spark.sql("DROP TABLE IF EXISTS ro_test.normal_v1")
        spark.sql("DROP DATABASE IF EXISTS ro_test CASCADE")
        spark.stop()
      }
    } finally {
      deleteRecursively(warehouseDir)
      super.afterAll()
    }
  }

  private def deleteRecursively(file: File): Unit = {
    if (file != null && file.exists()) {
      if (file.isDirectory) file.listFiles().foreach(deleteRecursively)
      file.delete()
    }
  }

  private def assertReadOnlyBlocked(sql: String): Unit = {
    val ex = intercept[AnalysisException](spark.sql(sql))
    assert(ex.getMessage.contains("read-only"),
      s"Expected read-only error but got: ${ex.getMessage}")
  }

  // ══════════════════════════════════════════════════════════════
  //  V1 HIVE TABLE – WRITE OPERATIONS (should be blocked)
  // ══════════════════════════════════════════════════════════════

  test("V1: INSERT INTO read-only table is blocked") {
    assertReadOnlyBlocked("INSERT INTO ro_test.readonly_v1 VALUES (1, 'a')")
  }

  test("V1: INSERT OVERWRITE read-only table is blocked") {
    assertReadOnlyBlocked("INSERT OVERWRITE TABLE ro_test.readonly_v1 VALUES (1, 'a')")
  }

  // ══════════════════════════════════════════════════════════════
  //  V1 HIVE TABLE – DDL OPERATIONS (should be blocked)
  // ══════════════════════════════════════════════════════════════

  test("V1: ALTER TABLE ADD COLUMNS on read-only table is blocked") {
    assertReadOnlyBlocked("ALTER TABLE ro_test.readonly_v1 ADD COLUMNS (age INT)")
  }

  test("V1: ALTER TABLE SET TBLPROPERTIES on read-only table is blocked") {
    assertReadOnlyBlocked(
      "ALTER TABLE ro_test.readonly_v1 SET TBLPROPERTIES ('key' = 'value')")
  }

  test("V1: ALTER TABLE UNSET TBLPROPERTIES on read-only table is blocked") {
    assertReadOnlyBlocked(
      "ALTER TABLE ro_test.readonly_v1 UNSET TBLPROPERTIES ('hive-ext.readOnly')")
  }

  test("V1: ALTER TABLE SET LOCATION on read-only table is blocked") {
    assertReadOnlyBlocked(
      s"ALTER TABLE ro_test.readonly_v1 SET LOCATION '${warehouseDir.getAbsolutePath}/tmp'")
  }

  test("V1: ALTER TABLE RENAME read-only table is blocked") {
    assertReadOnlyBlocked("ALTER TABLE ro_test.readonly_v1 RENAME TO ro_test.renamed_v1")
  }

  test("V1: DROP TABLE read-only table is blocked") {
    assertReadOnlyBlocked("DROP TABLE ro_test.readonly_v1")
  }

  test("V1: TRUNCATE TABLE read-only table is blocked") {
    assertReadOnlyBlocked("TRUNCATE TABLE ro_test.readonly_v1")
  }

  // ══════════════════════════════════════════════════════════════
  //  V1 HIVE TABLE – READ / NON-READONLY (should be allowed)
  // ══════════════════════════════════════════════════════════════

  test("V1: SELECT from read-only table is allowed") {
    spark.sql("SELECT * FROM ro_test.readonly_v1").collect()
  }

  test("V1: INSERT INTO non-read-only table is allowed") {
    spark.sql("INSERT INTO ro_test.normal_v1 VALUES (1, 'a')")
  }

  test("V1: ALTER TABLE on non-read-only table is allowed") {
    spark.sql("ALTER TABLE ro_test.normal_v1 SET TBLPROPERTIES ('key' = 'value')")
  }

  test("V1: DROP TABLE non-read-only table is allowed") {
    spark.sql(
      """CREATE TABLE ro_test.temp_v1 (id INT)
        |STORED AS PARQUET""".stripMargin)
    spark.sql("DROP TABLE ro_test.temp_v1")
  }

  // ══════════════════════════════════════════════════════════════
  //  V2 TABLE – WRITE OPERATIONS (should be blocked)
  // ══════════════════════════════════════════════════════════════

  test("V2: INSERT INTO read-only table is blocked") {
    assertReadOnlyBlocked("INSERT INTO testcat.ns.readonly_v2 VALUES (1, 'a')")
  }

  test("V2: DELETE FROM read-only table is blocked") {
    assertReadOnlyBlocked("DELETE FROM testcat.ns.readonly_v2 WHERE id = 1")
  }

  test("V2: UPDATE read-only table is blocked") {
    assertReadOnlyBlocked("UPDATE testcat.ns.readonly_v2 SET name = 'b' WHERE id = 1")
  }

  test("V2: MERGE INTO read-only table is blocked") {
    // Insert seed data into normal_v2 first
    spark.sql("INSERT INTO testcat.ns.normal_v2 VALUES (1, 'src')")
    assertReadOnlyBlocked(
      """MERGE INTO testcat.ns.readonly_v2 t
        |USING testcat.ns.normal_v2 s ON t.id = s.id
        |WHEN MATCHED THEN UPDATE SET t.name = s.name
        |WHEN NOT MATCHED THEN INSERT *""".stripMargin)
  }

  // ══════════════════════════════════════════════════════════════
  //  V2 TABLE – DDL OPERATIONS (should be blocked)
  // ══════════════════════════════════════════════════════════════

  test("V2: ALTER TABLE ADD COLUMNS on read-only table is blocked") {
    assertReadOnlyBlocked("ALTER TABLE testcat.ns.readonly_v2 ADD COLUMNS (age INT)")
  }

  test("V2: ALTER TABLE SET TBLPROPERTIES on read-only table is blocked") {
    assertReadOnlyBlocked(
      "ALTER TABLE testcat.ns.readonly_v2 SET TBLPROPERTIES ('key' = 'value')")
  }

  test("V2: ALTER TABLE DROP COLUMNS on read-only table is blocked") {
    assertReadOnlyBlocked("ALTER TABLE testcat.ns.readonly_v2 DROP COLUMNS name")
  }

  test("V2: DROP TABLE read-only V2 table is blocked") {
    assertReadOnlyBlocked("DROP TABLE testcat.ns.readonly_v2")
  }

  // ══════════════════════════════════════════════════════════════
  //  V2 TABLE – READ / NON-READONLY (should be allowed)
  // ══════════════════════════════════════════════════════════════

  test("V2: SELECT from read-only table is allowed") {
    spark.sql("SELECT * FROM testcat.ns.readonly_v2").collect()
  }

  test("V2: INSERT INTO non-read-only V2 table is allowed") {
    spark.sql("INSERT INTO testcat.ns.normal_v2 VALUES (2, 'b')")
  }

  test("V2: ALTER TABLE on non-read-only V2 table is allowed") {
    spark.sql("ALTER TABLE testcat.ns.normal_v2 SET TBLPROPERTIES ('key' = 'value')")
  }

  test("V2: DROP TABLE non-read-only V2 table is allowed") {
    spark.sql("CREATE TABLE testcat.ns.temp_v2 (id INT)")
    spark.sql("DROP TABLE testcat.ns.temp_v2")
  }

  // ══════════════════════════════════════════════════════════════
  //  EDGE CASES
  // ══════════════════════════════════════════════════════════════

  test("V2: property value is case-insensitive (TRUE)") {
    spark.sql(
      """CREATE TABLE testcat.ns.upper_ro (id INT)
        |TBLPROPERTIES ('hive-ext.readOnly' = 'TRUE')""".stripMargin)
    assertReadOnlyBlocked("INSERT INTO testcat.ns.upper_ro VALUES (1)")
    // Cleanup via direct catalog since DROP is blocked
  }

  test("V2: property value 'false' does not block writes") {
    spark.sql(
      """CREATE TABLE testcat.ns.false_ro (id INT)
        |TBLPROPERTIES ('hive-ext.readOnly' = 'false')""".stripMargin)
    spark.sql("INSERT INTO testcat.ns.false_ro VALUES (1)")
    spark.sql("DROP TABLE testcat.ns.false_ro")
  }

  test("V2: table without property allows all operations") {
    spark.sql("CREATE TABLE testcat.ns.no_prop (id INT)")
    spark.sql("INSERT INTO testcat.ns.no_prop VALUES (1)")
    spark.sql("ALTER TABLE testcat.ns.no_prop ADD COLUMNS (v STRING)")
    spark.sql("DROP TABLE testcat.ns.no_prop")
  }

  test("V1: hive-ext.readOnly=false does not block writes") {
    spark.sql(
      """CREATE TABLE ro_test.false_v1 (id INT)
        |STORED AS PARQUET
        |TBLPROPERTIES ('hive-ext.readOnly' = 'false')""".stripMargin)
    spark.sql("INSERT INTO ro_test.false_v1 VALUES (1)")
    spark.sql("DROP TABLE ro_test.false_v1")
  }
}
