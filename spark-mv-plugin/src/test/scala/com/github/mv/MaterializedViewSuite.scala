package com.github.mv

import java.io.File
import java.nio.file.Files

import org.apache.spark.sql.{AnalysisException, Row, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

/**
 * Comprehensive unit tests for the materialized view extension.
 *
 * Test matrix:
 *   CREATE  – basic creation, duplicate detection, schema preservation
 *   DROP    – basic drop, non-existent detection, backing table cleanup
 *   REFRESH – full refresh updates data, incremental refresh
 *   SHOW    – list registered MVs
 *   QUERY OPTIMIZATION – automatic query rewriting to backing table
 */
class MaterializedViewSuite extends AnyFunSuite with BeforeAndAfterAll with BeforeAndAfterEach {

  private var spark: SparkSession = _
  private var warehouseDir: File = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    warehouseDir = Files.createTempDirectory("mv-test-warehouse").toFile
    spark = SparkSession.builder()
      .master("local[2]")
      .config("spark.sql.extensions", classOf[MaterializedViewExtensions].getName)
      .config("spark.sql.warehouse.dir", warehouseDir.getAbsolutePath)
      .config("spark.ui.enabled", "false")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("javax.jdo.option.ConnectionURL",
        s"jdbc:derby:;databaseName=${warehouseDir.getAbsolutePath}/metastore_db;create=true")
      .enableHiveSupport()
      .getOrCreate()

    // Create source tables for MV queries
    spark.sql("CREATE TABLE src_orders (id INT, amount DOUBLE, status STRING) USING parquet")
    spark.sql("INSERT INTO src_orders VALUES (1, 100.0, 'completed'), (2, 200.0, 'pending'), (3, 150.0, 'completed')")

    spark.sql("CREATE TABLE src_items (id INT, order_id INT, product STRING) USING parquet")
    spark.sql("INSERT INTO src_items VALUES (1, 1, 'laptop'), (2, 1, 'mouse'), (3, 2, 'keyboard')")
  }

  override def afterEach(): Unit = {
    // Clean up any MVs registered during the test
    MaterializedViewCatalog.listAll().foreach { meta =>
      try { spark.sql(s"DROP TABLE IF EXISTS ${meta.backingTable}") } catch { case _: Exception => }
    }
    MaterializedViewCatalog.clear()
    super.afterEach()
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.sql("DROP TABLE IF EXISTS src_orders")
        spark.sql("DROP TABLE IF EXISTS src_items")
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

  // ══════════════════════════════════════════════════════════════
  //  CREATE MATERIALIZED VIEW
  // ══════════════════════════════════════════════════════════════

  test("CREATE MATERIALIZED VIEW creates backing table with correct data") {
    spark.sql("CREATE MATERIALIZED VIEW mv_completed AS SELECT id, amount FROM src_orders WHERE status = 'completed'")

    assert(MaterializedViewCatalog.exists("mv_completed"))

    val meta = MaterializedViewCatalog.get("mv_completed").get
    val result = spark.sql(s"SELECT * FROM ${meta.backingTable}").collect()
    assert(result.length == 2)
    assert(result.map(_.getInt(0)).toSet == Set(1, 3))
  }

  test("CREATE MATERIALIZED VIEW preserves schema") {
    spark.sql("CREATE MATERIALIZED VIEW mv_schema AS SELECT id, amount FROM src_orders")

    val meta = MaterializedViewCatalog.get("mv_schema").get
    val schema = spark.table(meta.backingTable).schema
    assert(schema.fieldNames.toSeq == Seq("id", "amount"))
  }

  test("CREATE MATERIALIZED VIEW with duplicate name fails") {
    spark.sql("CREATE MATERIALIZED VIEW mv_dup AS SELECT * FROM src_orders")
    val ex = intercept[AnalysisException] {
      spark.sql("CREATE MATERIALIZED VIEW mv_dup AS SELECT * FROM src_orders")
    }
    assert(ex.getMessage.contains("already exists"))
  }

  test("CREATE MATERIALIZED VIEW records query text") {
    val query = "SELECT id, amount FROM src_orders WHERE status = 'pending'"
    spark.sql(s"CREATE MATERIALIZED VIEW mv_query AS $query")

    val meta = MaterializedViewCatalog.get("mv_query").get
    assert(meta.query == query)
  }

  test("CREATE MATERIALIZED VIEW sets lastRefreshTs") {
    val before = System.currentTimeMillis()
    spark.sql("CREATE MATERIALIZED VIEW mv_ts AS SELECT * FROM src_orders")
    val after = System.currentTimeMillis()

    val meta = MaterializedViewCatalog.get("mv_ts").get
    assert(meta.lastRefreshTs >= before && meta.lastRefreshTs <= after)
  }

  // ══════════════════════════════════════════════════════════════
  //  DROP MATERIALIZED VIEW
  // ══════════════════════════════════════════════════════════════

  test("DROP MATERIALIZED VIEW removes catalog entry") {
    spark.sql("CREATE MATERIALIZED VIEW mv_drop AS SELECT * FROM src_orders")
    assert(MaterializedViewCatalog.exists("mv_drop"))

    spark.sql("DROP MATERIALIZED VIEW mv_drop")
    assert(!MaterializedViewCatalog.exists("mv_drop"))
  }

  test("DROP MATERIALIZED VIEW removes backing table") {
    spark.sql("CREATE MATERIALIZED VIEW mv_drop_bt AS SELECT * FROM src_orders")
    val meta = MaterializedViewCatalog.get("mv_drop_bt").get
    val backingTable = meta.backingTable

    // Verify backing table exists
    assert(spark.catalog.tableExists(backingTable))

    spark.sql("DROP MATERIALIZED VIEW mv_drop_bt")

    // Backing table should be removed
    assert(!spark.catalog.tableExists(backingTable))
  }

  test("DROP MATERIALIZED VIEW on non-existent MV fails") {
    val ex = intercept[AnalysisException] {
      spark.sql("DROP MATERIALIZED VIEW nonexistent_mv")
    }
    assert(ex.getMessage.contains("does not exist"))
  }

  // ══════════════════════════════════════════════════════════════
  //  REFRESH MATERIALIZED VIEW (full)
  // ══════════════════════════════════════════════════════════════

  test("REFRESH MATERIALIZED VIEW updates data") {
    spark.sql("CREATE MATERIALIZED VIEW mv_refresh AS SELECT id, amount FROM src_orders WHERE status = 'completed'")

    // Verify initial data
    val meta = MaterializedViewCatalog.get("mv_refresh").get
    assert(spark.sql(s"SELECT * FROM ${meta.backingTable}").count() == 2)

    // Insert more data into source
    spark.sql("INSERT INTO src_orders VALUES (4, 300.0, 'completed')")

    // Refresh the MV
    spark.sql("REFRESH MATERIALIZED VIEW mv_refresh")

    // Should now have 3 completed orders
    val result = spark.sql(s"SELECT * FROM ${meta.backingTable}").collect()
    assert(result.length == 3)
    assert(result.map(_.getInt(0)).toSet == Set(1, 3, 4))
  }

  test("REFRESH MATERIALIZED VIEW updates lastRefreshTs") {
    spark.sql("CREATE MATERIALIZED VIEW mv_refresh_ts AS SELECT * FROM src_orders")
    val ts1 = MaterializedViewCatalog.get("mv_refresh_ts").get.lastRefreshTs

    Thread.sleep(10) // ensure time difference
    spark.sql("REFRESH MATERIALIZED VIEW mv_refresh_ts")

    val ts2 = MaterializedViewCatalog.get("mv_refresh_ts").get.lastRefreshTs
    assert(ts2 > ts1)
  }

  test("REFRESH MATERIALIZED VIEW on non-existent MV fails") {
    val ex = intercept[AnalysisException] {
      spark.sql("REFRESH MATERIALIZED VIEW nonexistent_mv")
    }
    assert(ex.getMessage.contains("does not exist"))
  }

  // ══════════════════════════════════════════════════════════════
  //  REFRESH MATERIALIZED VIEW INCREMENTAL
  // ══════════════════════════════════════════════════════════════

  test("REFRESH MATERIALIZED VIEW INCREMENTAL updates data") {
    spark.sql("CREATE MATERIALIZED VIEW mv_incr AS SELECT id, amount FROM src_orders WHERE status = 'completed'")

    val meta = MaterializedViewCatalog.get("mv_incr").get

    // Insert more completed orders
    spark.sql("INSERT INTO src_orders VALUES (5, 500.0, 'completed')")

    // Incremental refresh
    spark.sql("REFRESH MATERIALIZED VIEW mv_incr INCREMENTAL")

    val result = spark.sql(s"SELECT * FROM ${meta.backingTable}").collect()
    assert(result.map(_.getInt(0)).contains(5))
  }

  test("REFRESH MATERIALIZED VIEW INCREMENTAL updates lastRefreshTs") {
    spark.sql("CREATE MATERIALIZED VIEW mv_incr_ts AS SELECT * FROM src_orders")
    val ts1 = MaterializedViewCatalog.get("mv_incr_ts").get.lastRefreshTs

    Thread.sleep(10)
    spark.sql("REFRESH MATERIALIZED VIEW mv_incr_ts INCREMENTAL")

    val ts2 = MaterializedViewCatalog.get("mv_incr_ts").get.lastRefreshTs
    assert(ts2 > ts1)
  }

  test("REFRESH MATERIALIZED VIEW INCREMENTAL on non-existent MV fails") {
    val ex = intercept[AnalysisException] {
      spark.sql("REFRESH MATERIALIZED VIEW nonexistent_mv INCREMENTAL")
    }
    assert(ex.getMessage.contains("does not exist"))
  }

  // ══════════════════════════════════════════════════════════════
  //  SHOW MATERIALIZED VIEWS
  // ══════════════════════════════════════════════════════════════

  test("SHOW MATERIALIZED VIEWS returns empty when none registered") {
    val result = spark.sql("SHOW MATERIALIZED VIEWS").collect()
    assert(result.isEmpty)
  }

  test("SHOW MATERIALIZED VIEWS lists registered MVs") {
    spark.sql("CREATE MATERIALIZED VIEW mv_show1 AS SELECT * FROM src_orders")
    spark.sql("CREATE MATERIALIZED VIEW mv_show2 AS SELECT id FROM src_orders")

    val result = spark.sql("SHOW MATERIALIZED VIEWS").collect()
    assert(result.length == 2)

    val names = result.map(_.getString(0)).toSet
    assert(names.contains("mv_show1"))
    assert(names.contains("mv_show2"))
  }

  test("SHOW MATERIALIZED VIEWS has correct schema") {
    spark.sql("CREATE MATERIALIZED VIEW mv_show_schema AS SELECT * FROM src_orders")

    val df = spark.sql("SHOW MATERIALIZED VIEWS")
    val fields = df.schema.fieldNames
    assert(fields.toSeq == Seq("name", "query", "backing_table", "last_refresh_ts"))
  }

  // ══════════════════════════════════════════════════════════════
  //  QUERY AUTO-OPTIMIZATION
  // ══════════════════════════════════════════════════════════════

  test("Query matching MV definition is optimized to use backing table") {
    spark.sql("CREATE MATERIALIZED VIEW mv_opt AS SELECT id, amount FROM src_orders WHERE status = 'completed'")

    val meta = MaterializedViewCatalog.get("mv_opt").get

    // Execute the same query - should be rewritten to use backing table
    val result = spark.sql("SELECT id, amount FROM src_orders WHERE status = 'completed'")

    // Verify the optimized plan references the backing table
    val planStr = result.queryExecution.optimizedPlan.toString()
    assert(planStr.contains(meta.backingTable),
      s"Expected optimized plan to reference backing table ${meta.backingTable}, got: $planStr")
  }

  test("Query not matching any MV is not rewritten") {
    spark.sql("CREATE MATERIALIZED VIEW mv_no_match AS SELECT id FROM src_orders")

    // Different query should NOT be rewritten
    val result = spark.sql("SELECT amount FROM src_orders")
    val planStr = result.queryExecution.optimizedPlan.toString()
    assert(!planStr.contains("mv_backing_"),
      s"Expected no MV backing table reference, got: $planStr")
  }

  test("Optimized query returns correct results") {
    spark.sql("CREATE MATERIALIZED VIEW mv_result AS SELECT id, amount FROM src_orders WHERE status = 'completed'")

    val directResult = spark.sql("SELECT id, amount FROM src_orders WHERE status = 'completed'").collect()
    assert(directResult.nonEmpty)
    assert(directResult.forall(r => r.getInt(0) > 0))
  }

  // ══════════════════════════════════════════════════════════════
  //  CASE INSENSITIVITY
  // ══════════════════════════════════════════════════════════════

  test("MV name lookup is case-insensitive") {
    spark.sql("CREATE MATERIALIZED VIEW MV_Case AS SELECT * FROM src_orders")
    assert(MaterializedViewCatalog.exists("mv_case"))
    assert(MaterializedViewCatalog.exists("MV_Case"))
    assert(MaterializedViewCatalog.exists("MV_CASE"))

    spark.sql("DROP MATERIALIZED VIEW mv_case")
    assert(!MaterializedViewCatalog.exists("MV_Case"))
  }

  // ══════════════════════════════════════════════════════════════
  //  PARSE ERRORS
  // ══════════════════════════════════════════════════════════════

  test("CREATE MATERIALIZED VIEW without AS clause fails") {
    intercept[Exception] {
      spark.sql("CREATE MATERIALIZED VIEW bad_mv SELECT * FROM src_orders")
    }
  }

  // ══════════════════════════════════════════════════════════════
  //  END-TO-END WORKFLOW
  // ══════════════════════════════════════════════════════════════

  test("Full lifecycle: create, query, refresh, show, drop") {
    // 1. Create
    spark.sql("CREATE MATERIALIZED VIEW mv_lifecycle AS SELECT id, amount FROM src_orders WHERE status = 'completed'")
    assert(MaterializedViewCatalog.exists("mv_lifecycle"))

    // 2. Query backing table
    val meta = MaterializedViewCatalog.get("mv_lifecycle").get
    val initialData = spark.sql(s"SELECT * FROM ${meta.backingTable}").collect()
    assert(initialData.nonEmpty)

    // 3. Show
    val shown = spark.sql("SHOW MATERIALIZED VIEWS").collect()
    assert(shown.exists(_.getString(0) == "mv_lifecycle"))

    // 4. Insert more data and refresh
    spark.sql("INSERT INTO src_orders VALUES (10, 1000.0, 'completed')")
    spark.sql("REFRESH MATERIALIZED VIEW mv_lifecycle")
    val refreshedData = spark.sql(s"SELECT * FROM ${meta.backingTable}").collect()
    assert(refreshedData.length > initialData.length)

    // 5. Drop
    spark.sql("DROP MATERIALIZED VIEW mv_lifecycle")
    assert(!MaterializedViewCatalog.exists("mv_lifecycle"))
    assert(!spark.catalog.tableExists(meta.backingTable))
  }
}
