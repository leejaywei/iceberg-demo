package com.github.mv

import java.io.File
import java.nio.file.Files

import org.apache.spark.sql.{AnalysisException, Row, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

/**
 * Tests for aggregate query rewriting and roll-up optimizations.
 *
 * Covers StarRocks-style MV rewrite capabilities:
 *   Aggregate rewrite   – exact GROUP BY + aggregate function match
 *   Roll-up (上卷)       – coarser GROUP BY served by re-aggregation
 *   Predicate compensation – extra WHERE filters applied on MV scan
 *   Column subset        – query selects fewer columns than MV
 *   AVG derivation       – AVG recomputed from SUM/COUNT
 */
class AggregateRewriteSuite extends AnyFunSuite with BeforeAndAfterAll with BeforeAndAfterEach {

  private var spark: SparkSession = _
  private var warehouseDir: File = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    warehouseDir = Files.createTempDirectory("mv-agg-test-warehouse").toFile
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

    // Create a sales fact table
    spark.sql(
      """CREATE TABLE sales (
        |  region STRING,
        |  city STRING,
        |  product STRING,
        |  amount DOUBLE,
        |  quantity INT,
        |  sale_date STRING
        |) USING parquet""".stripMargin)

    spark.sql(
      """INSERT INTO sales VALUES
        |  ('east', 'new_york', 'laptop',  1000.0, 2, '2024-01'),
        |  ('east', 'new_york', 'mouse',    50.0, 10, '2024-01'),
        |  ('east', 'boston',   'laptop',   900.0, 1, '2024-01'),
        |  ('west', 'sf',      'laptop',  1100.0, 3, '2024-02'),
        |  ('west', 'sf',      'mouse',     40.0, 5, '2024-02'),
        |  ('west', 'la',      'keyboard', 200.0, 8, '2024-02'),
        |  ('east', 'new_york', 'laptop',  1200.0, 4, '2024-03'),
        |  ('west', 'la',      'laptop',  1050.0, 2, '2024-03')""".stripMargin)
  }

  override def afterEach(): Unit = {
    MaterializedViewCatalog.listAll().foreach { meta =>
      try { spark.sql(s"DROP TABLE IF EXISTS ${meta.backingTable}") } catch { case _: Exception => }
    }
    MaterializedViewCatalog.clear()
    super.afterEach()
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.sql("DROP TABLE IF EXISTS sales")
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
  //  AGGREGATE REWRITE – exact GROUP BY match
  // ══════════════════════════════════════════════════════════════

  test("Aggregate rewrite: exact GROUP BY + SUM match redirects to MV") {
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_region_sum AS
        |SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_region_sum").get

    val result = spark.sql(
      """SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region""".stripMargin)

    val planStr = result.queryExecution.analyzed.toString()
    assert(planStr.contains(meta.backingTable),
      s"Expected analyzed plan to reference ${meta.backingTable}, got: $planStr")
  }

  test("Aggregate rewrite: correct results for SUM aggregate") {
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_region_sum2 AS
        |SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region""".stripMargin)

    val rows = spark.sql(
      """SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region
        |ORDER BY region""".stripMargin).collect()

    assert(rows.length == 2)
    val eastRow = rows.find(_.getString(0) == "east").get
    val westRow = rows.find(_.getString(0) == "west").get
    // east: 1000 + 50 + 900 + 1200 = 3150
    assert(eastRow.getDouble(1) == 3150.0)
    // west: 1100 + 40 + 200 + 1050 = 2390
    assert(westRow.getDouble(1) == 2390.0)
  }

  test("Aggregate rewrite: multiple aggregate functions (SUM, COUNT, MIN, MAX)") {
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_multi_agg AS
        |SELECT region, SUM(amount) AS total, COUNT(amount) AS cnt,
        |       MIN(amount) AS mn, MAX(amount) AS mx
        |FROM sales
        |GROUP BY region""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_multi_agg").get
    val result = spark.sql(
      """SELECT region, SUM(amount) AS total, COUNT(amount) AS cnt,
        |       MIN(amount) AS mn, MAX(amount) AS mx
        |FROM sales
        |GROUP BY region""".stripMargin)

    val planStr = result.queryExecution.analyzed.toString()
    assert(planStr.contains(meta.backingTable),
      s"Expected plan to reference ${meta.backingTable}, got: $planStr")
  }

  // ══════════════════════════════════════════════════════════════
  //  ROLL-UP (上卷) REWRITE
  // ══════════════════════════════════════════════════════════════

  test("Roll-up: query groups by subset of MV GROUP BY columns (SUM)") {
    // MV groups by (region, city), query groups by (region) only
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_city_sum AS
        |SELECT region, city, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region, city""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_city_sum").get

    val result = spark.sql(
      """SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region""".stripMargin)

    val planStr = result.queryExecution.analyzed.toString()
    assert(planStr.contains(meta.backingTable),
      s"Roll-up plan should reference ${meta.backingTable}, got: $planStr")
  }

  test("Roll-up: correct results for SUM roll-up") {
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_city_sum2 AS
        |SELECT region, city, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region, city""".stripMargin)

    val rows = spark.sql(
      """SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region
        |ORDER BY region""".stripMargin).collect()

    assert(rows.length == 2)
    val eastRow = rows.find(_.getString(0) == "east").get
    val westRow = rows.find(_.getString(0) == "west").get
    assert(eastRow.getDouble(1) == 3150.0)
    assert(westRow.getDouble(1) == 2390.0)
  }

  test("Roll-up: COUNT is rolled up via SUM") {
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_city_count AS
        |SELECT region, city, COUNT(amount) AS cnt
        |FROM sales
        |GROUP BY region, city""".stripMargin)

    val rows = spark.sql(
      """SELECT region, COUNT(amount) AS cnt
        |FROM sales
        |GROUP BY region
        |ORDER BY region""".stripMargin).collect()

    assert(rows.length == 2)
    val eastRow = rows.find(_.getString(0) == "east").get
    val westRow = rows.find(_.getString(0) == "west").get
    // east: new_york has 3, boston has 1 => 4
    assert(eastRow.getLong(1) == 4L)
    // west: sf has 2, la has 2 => 4
    assert(westRow.getLong(1) == 4L)
  }

  test("Roll-up: MIN/MAX are preserved correctly") {
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_city_minmax AS
        |SELECT region, city, MIN(amount) AS mn, MAX(amount) AS mx
        |FROM sales
        |GROUP BY region, city""".stripMargin)

    val rows = spark.sql(
      """SELECT region, MIN(amount) AS mn, MAX(amount) AS mx
        |FROM sales
        |GROUP BY region
        |ORDER BY region""".stripMargin).collect()

    assert(rows.length == 2)
    val eastRow = rows.find(_.getString(0) == "east").get
    val westRow = rows.find(_.getString(0) == "west").get
    // east min: 50.0, max: 1200.0
    assert(eastRow.getDouble(1) == 50.0)
    assert(eastRow.getDouble(2) == 1200.0)
    // west min: 40.0, max: 1100.0
    assert(westRow.getDouble(1) == 40.0)
    assert(westRow.getDouble(2) == 1100.0)
  }

  // ══════════════════════════════════════════════════════════════
  //  AVG DERIVATION FROM SUM/COUNT
  // ══════════════════════════════════════════════════════════════

  test("Roll-up: AVG derived from SUM and COUNT columns in MV") {
    // MV stores SUM and COUNT; query asks for AVG at coarser granularity
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_avg_base AS
        |SELECT region, city, SUM(amount) AS total, COUNT(amount) AS cnt
        |FROM sales
        |GROUP BY region, city""".stripMargin)

    val rows = spark.sql(
      """SELECT region, AVG(amount) AS avg_amount
        |FROM sales
        |GROUP BY region
        |ORDER BY region""".stripMargin).collect()

    assert(rows.length == 2)
    val eastRow = rows.find(_.getString(0) == "east").get
    val westRow = rows.find(_.getString(0) == "west").get
    // east: 3150.0 / 4 = 787.5
    assert(math.abs(eastRow.getDouble(1) - 787.5) < 0.01)
    // west: 2390.0 / 4 = 597.5
    assert(math.abs(westRow.getDouble(1) - 597.5) < 0.01)
  }

  // ══════════════════════════════════════════════════════════════
  //  PREDICATE COMPENSATION
  // ══════════════════════════════════════════════════════════════

  test("Predicate compensation: extra WHERE applied on top of MV scan") {
    // MV has no WHERE; query adds a WHERE filter
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_no_filter AS
        |SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |GROUP BY region""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_no_filter").get

    val rows = spark.sql(
      """SELECT region, SUM(amount) AS total_amount
        |FROM sales
        |WHERE region = 'east'
        |GROUP BY region""".stripMargin).collect()

    // Should return only east
    assert(rows.length == 1)
    assert(rows(0).getString(0) == "east")
    assert(rows(0).getDouble(1) == 3150.0)
  }

  // ══════════════════════════════════════════════════════════════
  //  COLUMN SUBSET
  // ══════════════════════════════════════════════════════════════

  test("Column subset: query selects fewer aggregate columns than MV") {
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_full AS
        |SELECT region, SUM(amount) AS total, COUNT(amount) AS cnt,
        |       MIN(amount) AS mn, MAX(amount) AS mx
        |FROM sales
        |GROUP BY region""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_full").get

    // Query only needs SUM and region
    val result = spark.sql(
      """SELECT region, SUM(amount) AS total
        |FROM sales
        |GROUP BY region""".stripMargin)

    val planStr = result.queryExecution.analyzed.toString()
    assert(planStr.contains(meta.backingTable),
      s"Expected plan to reference ${meta.backingTable}, got: $planStr")

    val rows = result.collect()
    assert(rows.length == 2)
  }

  // ══════════════════════════════════════════════════════════════
  //  NEGATIVE TESTS
  // ══════════════════════════════════════════════════════════════

  test("No rewrite: query groups by column NOT in MV GROUP BY") {
    // MV groups by region, but query groups by city
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_by_region AS
        |SELECT region, SUM(amount) AS total
        |FROM sales
        |GROUP BY region""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_by_region").get

    val result = spark.sql(
      """SELECT city, SUM(amount) AS total
        |FROM sales
        |GROUP BY city""".stripMargin)

    val planStr = result.queryExecution.analyzed.toString()
    assert(!planStr.contains(meta.backingTable),
      s"Should NOT reference ${meta.backingTable} when GROUP BY doesn't match, got: $planStr")
  }

  test("No rewrite: different base table") {
    // MV on one table, query on a different table
    spark.sql("CREATE TABLE other_sales (region STRING, amount DOUBLE) USING parquet")
    spark.sql("INSERT INTO other_sales VALUES ('east', 100.0)")

    spark.sql(
      """CREATE MATERIALIZED VIEW mv_sales_only AS
        |SELECT region, SUM(amount) AS total
        |FROM sales
        |GROUP BY region""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_sales_only").get

    val result = spark.sql(
      """SELECT region, SUM(amount) AS total
        |FROM other_sales
        |GROUP BY region""".stripMargin)

    val planStr = result.queryExecution.analyzed.toString()
    assert(!planStr.contains(meta.backingTable),
      s"Should NOT match MV for different base table, got: $planStr")

    spark.sql("DROP TABLE IF EXISTS other_sales")
  }

  test("No rewrite: MV has restrictive WHERE that query doesn't have") {
    // MV filters by region='east', but query wants all regions
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_east_only AS
        |SELECT region, SUM(amount) AS total
        |FROM sales
        |WHERE region = 'east'
        |GROUP BY region""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_east_only").get

    val result = spark.sql(
      """SELECT region, SUM(amount) AS total
        |FROM sales
        |GROUP BY region""".stripMargin)

    val planStr = result.queryExecution.analyzed.toString()
    assert(!planStr.contains(meta.backingTable),
      s"Should NOT use MV with more restrictive WHERE, got: $planStr")
  }

  // ══════════════════════════════════════════════════════════════
  //  INTERACTION WITH EXISTING EXACT-MATCH REWRITE
  // ══════════════════════════════════════════════════════════════

  test("Exact match rewrite still works for non-aggregate queries") {
    // Use the sales table that's available in this suite
    spark.sql(
      """CREATE MATERIALIZED VIEW mv_exact AS
        |SELECT region, amount FROM sales WHERE region = 'east'""".stripMargin)

    val meta = MaterializedViewCatalog.get("mv_exact").get

    val result = spark.sql("SELECT region, amount FROM sales WHERE region = 'east'")
    val planStr = result.queryExecution.analyzed.toString()
    assert(planStr.contains(meta.backingTable),
      s"Exact match should still work, got: $planStr")
  }
}
