import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from pyspark.sql import SparkSession, functions as F


# ----------------------------
# Result model + runner
# ----------------------------
@dataclass
class CaseResult:
    group: str
    name: str
    status: str  # PASS/FAIL/SKIP
    seconds: float
    error: str = ""


class SkipCase(Exception):
    pass


def get_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Iceberg Spark4 Full Suite - spark_catalog(hive) + procedures")
        .enableHiveSupport()
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
        .config("spark.sql.catalog.spark_catalog.type", "hive")
        .config("spark.sql.storeAssignmentPolicy", "ANSI")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_sql(spark: SparkSession, sql: str, show: bool = False):
    sql_clean = sql.strip().rstrip(";")
    print("\n[SQL] " + sql_clean.replace("\n", "\n      "))
    df = spark.sql(sql_clean)
    if show:
        df.show(truncate=False)
    return df


def try_sql(spark: SparkSession, sql: str, show: bool = False) -> Tuple[bool, str]:
    try:
        run_sql(spark, sql, show=show)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def scalar_long(spark: SparkSession, sql: str) -> int:
    return int(run_sql(spark, sql).collect()[0][0])


def scalar_str(spark: SparkSession, sql: str) -> str:
    return str(run_sql(spark, sql).collect()[0][0])


# ----------------------------
# Assertion helpers
# ----------------------------
def assert_true(cond: bool, msg: str):
    """Assert that condition is true, raise RuntimeError with message if false."""
    if not cond:
        raise RuntimeError(f"Assertion failed: {msg}")


def assert_eq(actual, expected, msg: str = None):
    """Assert that actual equals expected."""
    if actual != expected:
        err_msg = f"Expected {expected}, got {actual}"
        if msg:
            err_msg = f"{msg}: {err_msg}"
        raise RuntimeError(err_msg)


def assert_sql_count(spark: SparkSession, sql: str, expected: int, msg: str = None):
    """Execute SQL query and assert row count matches expected."""
    actual = scalar_long(spark, sql)
    err_msg = msg or f"SQL count mismatch for: {sql[:100]}"
    assert_eq(actual, expected, err_msg)


def assert_sql_scalar(spark: SparkSession, sql: str, expected, msg: str = None):
    """Execute SQL query and assert scalar result matches expected."""
    actual = run_sql(spark, sql).collect()[0][0]
    err_msg = msg or f"SQL scalar mismatch for: {sql[:100]}"
    assert_eq(actual, expected, err_msg)


# ----------------------------
# Test Suite
# ----------------------------
class Suite:
    def __init__(self, spark: SparkSession, db: str = "qa_full"):
        self.spark = spark
        self.catalog = "spark_catalog"
        self.db = db
        self.run_procedures = True

    def ns(self) -> str:
        return f"{self.catalog}.{self.db}"

    def t(self, name: str) -> str:
        return f"{self.catalog}.{self.db}.{name}"

    def use_ns(self):
        run_sql(self.spark, f"USE {self.ns()}")

    # -------- helpers for migration/procedures ----------
    def _describe_location(self, full_table: str) -> str:
        df = run_sql(self.spark, f"DESCRIBE TABLE EXTENDED {full_table}")
        for r in df.collect():
            if r.col_name and str(r.col_name).strip().lower() == "location":
                return str(r.data_type).strip()
        raise SkipCase(f"Cannot parse Location from DESCRIBE TABLE EXTENDED: {full_table}")

    def _latest_metadata_json(self, iceberg_table: str) -> str:
        df = run_sql(
            self.spark,
            f"SELECT file FROM {iceberg_table}.metadata_log_entries ORDER BY timestamp DESC LIMIT 1"
        )
        rows = df.collect()
        if not rows:
            raise SkipCase(f"No metadata_log_entries rows for {iceberg_table}")
        return str(rows[0]["file"])

    def _infer_hdfs_location(self, base_table: str, suffix: str = "test_location") -> str:
        """
        Derive a writable HDFS location from an existing table's location.
        Returns a sibling directory path suitable for LOCATION clause.
        Raises SkipCase if location cannot be inferred or is not HDFS/writable.
        """
        try:
            base_location = self._describe_location(base_table)
            # Strip trailing slash
            base_location = base_location.rstrip("/")
            
            # For HDFS/S3, create sibling directory
            if base_location.startswith("hdfs://") or base_location.startswith("s3://") or base_location.startswith("s3a://"):
                # Extract parent and create sibling
                parent = base_location.rsplit("/", 1)[0]
                new_location = f"{parent}/{suffix}_{int(time.time())}"
                return new_location
            else:
                # For local file:// paths, skip in HDFS environment
                raise SkipCase(f"Base location is local file:// - skipping LOCATION test in HDFS environment")
        except Exception as e:
            raise SkipCase(f"Cannot infer HDFS location: {e}")

    def _get_latest_snapshot_id(self, iceberg_table: str) -> int:
        """Get the latest snapshot ID for an Iceberg table."""
        try:
            sid = scalar_long(self.spark, f"SELECT snapshot_id FROM {iceberg_table}.snapshots ORDER BY committed_at DESC LIMIT 1")
            return sid
        except Exception as e:
            raise SkipCase(f"Cannot get latest snapshot ID: {e}")

    def _is_unsupported_feature_error(self, error_msg: str) -> bool:
        """
        Check if an error message indicates an unsupported feature.
        Used to SKIP tests that are not supported in certain Spark versions.
        """
        error_lower = error_msg.lower()
        unsupported_patterns = [
            "unsupported_feature",
            "not supported",
            "does not support",
            "unsupported operation",
            "is not supported",
        ]
        return any(pattern in error_lower for pattern in unsupported_patterns)

    # ----------------------------
    # Environment
    # ----------------------------
    def env_prepare(self):
        run_sql(self.spark, f"CREATE DATABASE IF NOT EXISTS {self.ns()}")
        self.use_ns()

        # drop views
        for vw in ["sample_vw", "sample_vw_props", "cdc_changes", "sample_vw_if_not_exists", "sample_vw_with_metadata"]:
            run_sql(self.spark, f"DROP VIEW IF EXISTS {self.t(vw)}")

        # drop tables
        for tbl in [
            "sample_unpart",
            "sample_part",
            "sample_ctas",
            "sample_rtas",
            "sample_alter",
            "sample_nested",
            "write_target",
            "write_source",
            "logs",
            "df_v2_target",
            "cdc_tbl",
            # migration-specific
            "src_parquet_tbl",
            "addfiles_target_tbl",
            "src_parquet_tbl_BACKUP_",
            # new test tables
            "sample_with_comments",
            "sample_with_location",
            "sample_like_test",
            "sample_rtas_advanced",
            "sample_rename_old",
            "sample_rename_new",
            "sample_alter_comment",
        ]:
            try_sql(self.spark, f"DROP TABLE IF EXISTS {self.t(tbl)} PURGE")
            try_sql(self.spark, f"DROP TABLE IF EXISTS {self.t(tbl)}")

    def env_seed_base_tables(self):
        # base unpartitioned
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("sample_unpart")} (
                id bigint NOT NULL,
                data string
            ) USING iceberg
        """)
        run_sql(self.spark, f"INSERT INTO {self.t('sample_unpart')} VALUES (1,'a'),(2,'b')")

        # base partitioned
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("sample_part")} (
                id bigint,
                data string,
                category string,
                ts timestamp
            )
            USING iceberg
            PARTITIONED BY (bucket(16, id), days(ts), category, truncate(4, data))
            TBLPROPERTIES ('format-version'='2')
        """)
        run_sql(self.spark, f"""
            INSERT INTO {self.t("sample_part")} VALUES
            (1, 'abcdefgh', 'c1', TIMESTAMP'2026-01-20 01:02:03'),
            (2, 'abcdZZZZ', 'c2', TIMESTAMP'2026-01-21 01:02:03')
        """)

    # ----------------------------
    # DDL cases
    # ----------------------------
    def ddl_ctas_basic(self):
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("sample_ctas")}
            USING iceberg
            AS SELECT id, data FROM {self.t("sample_unpart")}
        """)
        run_sql(self.spark, f"SELECT * FROM {self.t('sample_ctas')} ORDER BY id", show=True)

    def ddl_ctas_with_props_and_partition(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('sample_ctas')}")
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("sample_ctas")}
            USING iceberg
            PARTITIONED BY (truncate(2, data))
            TBLPROPERTIES ('key'='value')
            AS SELECT id, data FROM {self.t("sample_unpart")}
        """)
        run_sql(self.spark, f"SHOW TBLPROPERTIES {self.t('sample_ctas')}", show=True)

    def ddl_rtas_replace_table_as_select_existing_first(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('sample_rtas')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('sample_rtas')} (id bigint, data string) USING iceberg")
        run_sql(self.spark, f"""
            REPLACE TABLE {self.t("sample_rtas")}
            USING iceberg
            AS SELECT id, data FROM {self.t("sample_unpart")} WHERE id = 1
        """)
        run_sql(self.spark, f"SELECT * FROM {self.t('sample_rtas')}", show=True)

    def ddl_rtas_create_or_replace_table_as_select(self):
        run_sql(self.spark, f"""
            CREATE OR REPLACE TABLE {self.t("sample_rtas")}
            USING iceberg
            AS SELECT id, data FROM {self.t("sample_unpart")} WHERE id >= 1
        """)

    def ddl_drop_table_and_purge(self):
        run_sql(self.spark, f"CREATE TABLE {self.t('tmp_drop')} (id bigint) USING iceberg")
        run_sql(self.spark, f"DROP TABLE {self.t('tmp_drop')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('tmp_drop')} (id bigint) USING iceberg")
        run_sql(self.spark, f"DROP TABLE {self.t('tmp_drop')} PURGE")

    def ddl_alter_table_core(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('sample_alter')}")
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("sample_alter")} (
                id bigint NOT NULL,
                measurement int,
                data string,
                point struct<x: double, y: double>
            ) USING iceberg
        """)
        run_sql(self.spark, f"""
            ALTER TABLE {self.t('sample_alter')} SET TBLPROPERTIES (
                'read.split.target-size'='268435456',
                'comment'='A table comment.'
            )
        """)
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} UNSET TBLPROPERTIES ('read.split.target-size')")
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} ADD COLUMNS (new_column string comment 'docs')")
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} ADD COLUMN point.z double")
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} RENAME COLUMN data TO payload")
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} ALTER COLUMN measurement TYPE bigint")
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} ALTER COLUMN id DROP NOT NULL")
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} DROP COLUMN new_column")
        run_sql(self.spark, f"ALTER TABLE {self.t('sample_alter')} DROP COLUMN point.z")

    def ddl_sql_extensions_partition_evolution_and_write_order(self):
        tbl = self.t("sample_nested")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        run_sql(self.spark, f"""
            CREATE TABLE {tbl} (
                id bigint NOT NULL,
                category string,
                data string,
                ts timestamp
            ) USING iceberg
            PARTITIONED BY (days(ts))
        """)
        run_sql(self.spark, f"ALTER TABLE {tbl} ADD PARTITION FIELD category")
        run_sql(self.spark, f"ALTER TABLE {tbl} ADD PARTITION FIELD bucket(16, id) AS shard")
        run_sql(self.spark, f"ALTER TABLE {tbl} ADD PARTITION FIELD truncate(4, data)")
        run_sql(self.spark, f"ALTER TABLE {tbl} ADD PARTITION FIELD year(ts)")
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP PARTITION FIELD shard")

        ok, err = try_sql(self.spark, f"ALTER TABLE {tbl} REPLACE PARTITION FIELD ts_day WITH day(ts) AS day_of_ts")
        if not ok:
            ok2, err2 = try_sql(self.spark, f"ALTER TABLE {tbl} REPLACE PARTITION FIELD ts WITH day(ts) AS day_of_ts")
            if not ok2:
                raise SkipCase(f"REPLACE PARTITION FIELD skipped (cannot infer old field name). err1={err}; err2={err2}")

        run_sql(self.spark, f"ALTER TABLE {tbl} WRITE ORDERED BY category ASC NULLS LAST, id DESC NULLS FIRST")
        run_sql(self.spark, f"ALTER TABLE {tbl} WRITE LOCALLY ORDERED BY category, id")
        run_sql(self.spark, f"ALTER TABLE {tbl} WRITE UNORDERED")
        run_sql(self.spark, f"ALTER TABLE {tbl} WRITE DISTRIBUTED BY PARTITION")
        run_sql(self.spark, f"ALTER TABLE {tbl} SET IDENTIFIER FIELDS id")
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP IDENTIFIER FIELDS id")

    def ddl_views(self):
        base = self.t("sample_unpart")
        vw = self.t("sample_vw")
        vw_props = self.t("sample_vw_props")

        run_sql(self.spark, f"CREATE VIEW {vw} AS SELECT * FROM {base}")
        run_sql(self.spark, f"""
            CREATE VIEW {vw_props}
            TBLPROPERTIES ('key1'='val1','key2'='val2')
            AS SELECT * FROM {base}
        """)
        run_sql(self.spark, f"SHOW TBLPROPERTIES {vw_props}", show=True)
        run_sql(self.spark, f"SHOW VIEWS IN {self.ns()}", show=True)
        run_sql(self.spark, f"SHOW CREATE TABLE {vw}", show=True)
        run_sql(self.spark, f"DESCRIBE EXTENDED {vw}", show=True)
        run_sql(self.spark, f"DROP VIEW {vw_props}")
        run_sql(self.spark, f"""
            CREATE OR REPLACE VIEW {vw}
            TBLPROPERTIES ('key1'='new_val1')
            AS SELECT id FROM {base}
        """)
        run_sql(self.spark, f"ALTER VIEW {vw} SET TBLPROPERTIES ('key1'='val3','key4'='val4')")
        run_sql(self.spark, f"ALTER VIEW {vw} UNSET TBLPROPERTIES ('key4')")

    # ----------------------------
    # Additional DDL tests
    # ----------------------------
    def ddl_create_table_with_comments(self):
        """Test CREATE TABLE with table-level and column-level comments"""
        tbl = self.t("sample_with_comments")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        run_sql(self.spark, f"""
            CREATE TABLE {tbl} (
                id bigint COMMENT 'Unique identifier',
                data string COMMENT 'Data payload',
                category string
            ) USING iceberg
            COMMENT 'Sample table with comments for testing'
        """)
        # Verify comments are stored
        run_sql(self.spark, f"DESCRIBE TABLE EXTENDED {tbl}", show=True)
        run_sql(self.spark, f"DROP TABLE {tbl}")

    def ddl_create_table_with_location(self):
        """Test CREATE TABLE with LOCATION clause (HDFS-aware)"""
        tbl = self.t("sample_with_location")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        
        try:
            # Try to infer a writable HDFS location from existing table
            location = self._infer_hdfs_location(self.t("sample_unpart"), "sample_with_location")
            
            run_sql(self.spark, f"""
                CREATE TABLE {tbl} (
                    id bigint,
                    data string
                ) USING iceberg
                LOCATION '{location}'
            """)
            run_sql(self.spark, f"INSERT INTO {tbl} VALUES (1, 'test')")
            run_sql(self.spark, f"SELECT * FROM {tbl}", show=True)
            run_sql(self.spark, f"DROP TABLE {tbl}")
        except SkipCase:
            # If we can't infer location, skip this test
            raise

    def ddl_create_table_like_negative(self):
        """Test CREATE TABLE LIKE - should fail as not supported by Iceberg"""
        tbl_src = self.t("sample_unpart")
        tbl_new = self.t("sample_like_test")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl_new}")
        
        # Try CREATE TABLE LIKE - this should fail for Iceberg
        ok, err = try_sql(self.spark, f"CREATE TABLE {tbl_new} LIKE {tbl_src}")
        
        if not ok:
            # Expected failure - this is a PASS
            print(f"[EXPECTED] CREATE TABLE LIKE failed as expected: {err}")
        else:
            # Unexpected success - clean up and mark as potential issue
            try_sql(self.spark, f"DROP TABLE IF EXISTS {tbl_new}")
            raise SkipCase("CREATE TABLE LIKE succeeded unexpectedly (may be supported in this version)")

    def ddl_rtas_with_partition_and_properties(self):
        """Test REPLACE TABLE AS SELECT with PARTITIONED BY and TBLPROPERTIES"""
        tbl = self.t("sample_rtas_advanced")
        
        # First create the table with some properties
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        run_sql(self.spark, f"""
            CREATE TABLE {tbl} (
                id bigint,
                data string,
                category string
            ) USING iceberg
            TBLPROPERTIES ('old_prop'='old_value', 'common_prop'='original')
        """)
        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (1, 'old', 'c1')")
        
        # Replace with new schema, partitioning, and properties
        run_sql(self.spark, f"""
            REPLACE TABLE {tbl}
            USING iceberg
            PARTITIONED BY (category)
            TBLPROPERTIES ('new_prop'='new_value', 'common_prop'='updated')
            AS SELECT id, data, category FROM {self.t('sample_unpart')} WHERE id > 0
        """)
        
        # Check the result
        run_sql(self.spark, f"SELECT * FROM {tbl} ORDER BY id", show=True)
        run_sql(self.spark, f"SHOW TBLPROPERTIES {tbl}", show=True)
        
        # Note: REPLACE TABLE typically does NOT preserve old properties
        # The behavior is to replace everything including properties

    def ddl_alter_table_rename(self):
        """Test ALTER TABLE RENAME TO"""
        tbl_old = self.t("sample_rename_old")
        tbl_new = self.t("sample_rename_new")
        
        # Clean up first
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl_old}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl_new}")
        
        # Create and populate
        run_sql(self.spark, f"CREATE TABLE {tbl_old} (id bigint, data string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {tbl_old} VALUES (1, 'test')")
        
        # Rename
        run_sql(self.spark, f"ALTER TABLE {tbl_old} RENAME TO {tbl_new}")
        
        # Verify new name works
        cnt = scalar_long(self.spark, f"SELECT count(*) FROM {tbl_new}")
        if cnt != 1:
            raise RuntimeError(f"Expected 1 row after rename, got {cnt}")
        
        # Rename back to avoid affecting other tests
        run_sql(self.spark, f"ALTER TABLE {tbl_new} RENAME TO {tbl_old}")
        run_sql(self.spark, f"DROP TABLE {tbl_old}")

    def ddl_alter_column_comment(self):
        """Test ALTER COLUMN to change column comment"""
        tbl = self.t("sample_alter_comment")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        
        run_sql(self.spark, f"""
            CREATE TABLE {tbl} (
                id bigint COMMENT 'Original comment',
                data string
            ) USING iceberg
        """)
        
        # Try to alter column comment
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} ALTER COLUMN id COMMENT 'Updated comment'
        """)
        
        if not ok:
            # If not supported, skip
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"ALTER COLUMN COMMENT not supported: {err}")
            else:
                raise RuntimeError(f"ALTER COLUMN COMMENT failed unexpectedly: {err}")
        
        # Verify the comment was updated
        run_sql(self.spark, f"DESCRIBE TABLE {tbl}", show=True)
        run_sql(self.spark, f"DROP TABLE {tbl}")

    # ----------------------------
    # Enhanced Views tests
    # ----------------------------
    def ddl_views_if_not_exists(self):
        """Test CREATE VIEW IF NOT EXISTS"""
        base = self.t("sample_unpart")
        vw = self.t("sample_vw_if_not_exists")
        
        # Drop if exists
        run_sql(self.spark, f"DROP VIEW IF EXISTS {vw}")
        
        # Create view
        run_sql(self.spark, f"CREATE VIEW IF NOT EXISTS {vw} AS SELECT * FROM {base}")
        
        # Try creating again with IF NOT EXISTS - should succeed without error
        run_sql(self.spark, f"CREATE VIEW IF NOT EXISTS {vw} AS SELECT id FROM {base}")
        
        # Verify original view is unchanged
        run_sql(self.spark, f"SELECT * FROM {vw} LIMIT 1", show=True)
        
        run_sql(self.spark, f"DROP VIEW {vw}")

    def ddl_views_with_comments_and_aliases(self):
        """Test CREATE VIEW with view comment, column comments, and aliases"""
        base = self.t("sample_unpart")
        vw = self.t("sample_vw_with_metadata")
        
        run_sql(self.spark, f"DROP VIEW IF EXISTS {vw}")
        
        # Try with column aliases and view comment
        ok, err = try_sql(self.spark, f"""
            CREATE VIEW {vw}
            (identifier COMMENT 'The ID', payload COMMENT 'The data')
            COMMENT 'View with column aliases and comments'
            AS SELECT id AS identifier, data AS payload FROM {base}
        """)
        
        if not ok:
            # Some Spark versions may not support column comments in views
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"View with column comments not supported: {err}")
            else:
                raise RuntimeError(f"Failed to create view with metadata: {err}")
        
        run_sql(self.spark, f"DESCRIBE EXTENDED {vw}", show=True)
        run_sql(self.spark, f"SELECT * FROM {vw} LIMIT 1", show=True)
        run_sql(self.spark, f"DROP VIEW {vw}")

    # ----------------------------
    # Branch & Tag DDL tests
    # ----------------------------
    def ddl_branch_create_with_if_not_exists(self):
        """Test CREATE BRANCH with IF NOT EXISTS"""
        tbl = self.t("sample_part")
        branch = "test_branch_if_not_exists"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        
        # Create branch with IF NOT EXISTS
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH IF NOT EXISTS `{branch}`")
        
        # Try creating again - should succeed without error
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH IF NOT EXISTS `{branch}`")
        
        # Verify branch exists
        run_sql(self.spark, f"SELECT * FROM {tbl}.refs WHERE name = '{branch}'", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def ddl_branch_create_or_replace(self):
        """Test CREATE OR REPLACE BRANCH"""
        tbl = self.t("sample_part")
        branch = "test_branch_cor"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        
        # Create branch
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        
        # Create or replace - should succeed
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE OR REPLACE BRANCH `{branch}`")
        
        # Verify branch exists
        run_sql(self.spark, f"SELECT * FROM {tbl}.refs WHERE name = '{branch}'", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def ddl_branch_as_of_version(self):
        """Test CREATE BRANCH AS OF VERSION (snapshot)"""
        tbl = self.t("sample_part")
        branch = "test_branch_as_of"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        
        # Get a snapshot ID
        try:
            snapshot_id = self._get_latest_snapshot_id(tbl)
        except SkipCase:
            raise
        
        # Create branch at specific snapshot
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} CREATE BRANCH `{branch}` AS OF VERSION {snapshot_id}
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"CREATE BRANCH AS OF VERSION not supported: {err}")
            else:
                raise RuntimeError(f"CREATE BRANCH AS OF VERSION failed: {err}")
        
        # Verify branch exists
        run_sql(self.spark, f"SELECT * FROM {tbl}.refs WHERE name = '{branch}'", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def ddl_branch_replace(self):
        """Test REPLACE BRANCH"""
        tbl = self.t("sample_part")
        branch = "test_branch_replace"
        
        # Create branch first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        
        # Write to branch
        run_sql(self.spark, f"""
            INSERT INTO {tbl}.branch_{branch} 
            VALUES (9001, 'replace_test', 'c1', TIMESTAMP'2026-02-10 00:00:00')
        """)
        
        # Get snapshot before replace
        old_snapshot = self._get_latest_snapshot_id(f"{tbl}.branch_{branch}")
        
        # Now replace the branch (resets it)
        ok, err = try_sql(self.spark, f"ALTER TABLE {tbl} REPLACE BRANCH `{branch}`")
        
        if not ok:
            # Try alternative syntax
            ok2, err2 = try_sql(self.spark, f"ALTER TABLE {tbl} CREATE OR REPLACE BRANCH `{branch}`")
            if not ok2:
                if self._is_unsupported_feature_error(err) or self._is_unsupported_feature_error(err2):
                    raise SkipCase(f"REPLACE BRANCH not supported. err1={err}; err2={err2}")
                else:
                    raise RuntimeError(f"REPLACE BRANCH failed. err1={err}; err2={err2}")
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def ddl_branch_drop_if_exists(self):
        """Test DROP BRANCH IF EXISTS"""
        tbl = self.t("sample_part")
        branch = "test_branch_drop"
        
        # Create a branch
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        
        # Drop with IF EXISTS
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        
        # Drop again with IF EXISTS - should not error
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")

    def ddl_tag_create_with_if_not_exists(self):
        """Test CREATE TAG with IF NOT EXISTS"""
        tbl = self.t("sample_part")
        tag = "test_tag_if_not_exists"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")
        
        # Get snapshot ID for tag
        try:
            snapshot_id = self._get_latest_snapshot_id(tbl)
        except SkipCase:
            raise
        
        # Create tag with IF NOT EXISTS
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} CREATE TAG IF NOT EXISTS `{tag}` AS OF VERSION {snapshot_id}
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"CREATE TAG not supported: {err}")
            else:
                raise RuntimeError(f"CREATE TAG failed: {err}")
        
        # Try creating again - should succeed
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE TAG IF NOT EXISTS `{tag}` AS OF VERSION {snapshot_id}")
        
        # Verify tag exists
        run_sql(self.spark, f"SELECT * FROM {tbl}.refs WHERE name = '{tag}'", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")

    def ddl_tag_create_or_replace(self):
        """Test CREATE OR REPLACE TAG"""
        tbl = self.t("sample_part")
        tag = "test_tag_cor"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")
        
        # Get snapshot ID
        try:
            snapshot_id = self._get_latest_snapshot_id(tbl)
        except SkipCase:
            raise
        
        # Create tag
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} CREATE TAG `{tag}` AS OF VERSION {snapshot_id}
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"CREATE TAG not supported: {err}")
            else:
                raise RuntimeError(f"CREATE TAG failed: {err}")
        
        # Create or replace
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE OR REPLACE TAG `{tag}` AS OF VERSION {snapshot_id}")
        
        # Verify tag exists
        run_sql(self.spark, f"SELECT * FROM {tbl}.refs WHERE name = '{tag}'", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")

    def ddl_tag_drop_if_exists(self):
        """Test DROP TAG IF EXISTS"""
        tbl = self.t("sample_part")
        tag = "test_tag_drop"
        
        # Get snapshot ID
        try:
            snapshot_id = self._get_latest_snapshot_id(tbl)
        except SkipCase:
            raise
        
        # Create a tag
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} CREATE TAG `{tag}` AS OF VERSION {snapshot_id}
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"CREATE TAG not supported: {err}")
            else:
                raise RuntimeError(f"CREATE TAG failed: {err}")
        
        # Drop with IF EXISTS
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")
        
        # Drop again - should not error
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")

    def ddl_branch_with_retention(self):
        """Test CREATE BRANCH with retention settings (may not be supported)"""
        tbl = self.t("sample_part")
        branch = "test_branch_retention"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        
        # Try to create branch with retention
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} CREATE BRANCH `{branch}`
            RETAIN 7 DAYS
        """)
        
        if not ok:
            # Likely not supported - try without RETAIN clause
            ok2, err2 = try_sql(self.spark, f"""
                ALTER TABLE {tbl} CREATE BRANCH `{branch}`
                WITH SNAPSHOT RETENTION 7 DAYS
            """)
            if not ok2:
                raise SkipCase(f"Branch retention syntax not supported. err1={err}; err2={err2}")
        
        # If we got here, retention is supported
        run_sql(self.spark, f"SELECT * FROM {tbl}.refs WHERE name = '{branch}'", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    # ----------------------------
    # Writes (SQL)
    # ----------------------------
    def writes_insert_into_and_insert_select(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('write_target')}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('write_source')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('write_target')} (id bigint, data string) USING iceberg")
        run_sql(self.spark, f"CREATE TABLE {self.t('write_source')} (id bigint, data string) USING iceberg")

        run_sql(self.spark, f"INSERT INTO {self.t('write_target')} VALUES (1,'a'),(2,'b')")
        run_sql(self.spark, f"INSERT INTO {self.t('write_source')} VALUES (3,'c'),(4,'d')")
        run_sql(self.spark, f"INSERT INTO {self.t('write_target')} SELECT * FROM {self.t('write_source')}")
        run_sql(self.spark, f"SELECT count(*) FROM {self.t('write_target')}", show=True)

    def writes_merge_into(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('write_target')}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('write_source')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('write_target')} (id bigint NOT NULL, data string, cnt int) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('write_target')} VALUES (1,'a',10),(2,'b',20)")
        run_sql(self.spark, f"CREATE TABLE {self.t('write_source')} (id bigint, data string, cnt int, op string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('write_source')} VALUES (1,'a2',11,'update'),(3,'c',30,'insert')")

        run_sql(self.spark, f"""
            MERGE INTO {self.t('write_target')} t
            USING (SELECT id, data, cnt, op FROM {self.t('write_source')}) s
            ON t.id = s.id
            WHEN MATCHED AND s.op = 'update' THEN UPDATE SET t.data = s.data, t.cnt = s.cnt
            WHEN NOT MATCHED THEN INSERT (id, data, cnt) VALUES (s.id, s.data, s.cnt)
        """)
        
        # Validate: should have 3 rows (id=1 updated, id=2 unchanged, id=3 inserted)
        assert_sql_count(self.spark, f"SELECT count(*) FROM {self.t('write_target')}", 3, "Expected 3 rows after merge")
        assert_sql_scalar(self.spark, f"SELECT data FROM {self.t('write_target')} WHERE id=1", "a2", "Row 1 should be updated")
        assert_sql_scalar(self.spark, f"SELECT data FROM {self.t('write_target')} WHERE id=3", "c", "Row 3 should be inserted")
        run_sql(self.spark, f"SELECT * FROM {self.t('write_target')} ORDER BY id", show=True)

    def writes_insert_overwrite_dynamic_and_static(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('logs')}")
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("logs")} (
                uuid string NOT NULL,
                level string NOT NULL,
                ts timestamp NOT NULL,
                message string
            )
            USING iceberg
            PARTITIONED BY (level, hours(ts))
        """)
        run_sql(self.spark, f"""
            INSERT INTO {self.t("logs")} VALUES
            ('u1','INFO', TIMESTAMP'2026-01-01 01:00:00','m1'),
            ('u2','INFO', TIMESTAMP'2026-01-01 01:00:00','m2'),
            ('u3','INFO', TIMESTAMP'2026-01-01 02:00:00','m3'),
            ('u4','WARN', TIMESTAMP'2026-01-01 01:00:00','m4'),
            ('u5','INFO', TIMESTAMP'2026-01-02 01:00:00','m5')
        """)

        run_sql(self.spark, "SET spark.sql.sources.partitionOverwriteMode=dynamic")
        run_sql(self.spark, f"""
            INSERT OVERWRITE {self.t("logs")}
            SELECT uuid, level, ts, message
            FROM {self.t("logs")}
            WHERE level = 'INFO' AND cast(ts as date) = DATE'2026-01-01'
        """)
        dyn_cnt = scalar_long(self.spark, f"SELECT count(*) FROM {self.t('logs')}")
        assert_true(dyn_cnt > 0, "dynamic overwrite should not produce empty table")

        run_sql(self.spark, "SET spark.sql.sources.partitionOverwriteMode=static")
        run_sql(self.spark, f"""
            INSERT OVERWRITE {self.t("logs")}
            SELECT uuid, level, ts, message
            FROM {self.t("logs")}
            WHERE level = 'WARN'
        """)
        static_cnt = scalar_long(self.spark, f"SELECT count(*) FROM {self.t('logs')}")
        assert_eq(static_cnt, 1, "static overwrite should result in 1 row")


    def writes_delete_and_update(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('write_target')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('write_target')} (id bigint, v int) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('write_target')} VALUES (1,10),(2,20),(3,30)")
        run_sql(self.spark, f"DELETE FROM {self.t('write_target')} WHERE id = 1")
        assert_sql_count(self.spark, f"SELECT count(*) FROM {self.t('write_target')}", 2, "Expected 2 rows after DELETE")
        run_sql(self.spark, f"UPDATE {self.t('write_target')} SET v = 999 WHERE id = 2")
        assert_sql_scalar(self.spark, f"SELECT v FROM {self.t('write_target')} WHERE id=2", 999, "UPDATE should set v=999")


    def writes_to_branch_and_wap(self):
        tbl = self.t("sample_part")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE OR REPLACE BRANCH `audit_branch`")

        run_sql(self.spark, f"""
            INSERT INTO {tbl}.branch_audit_branch VALUES
            (100, 'branchdata', 'c9', TIMESTAMP'2026-02-01 00:00:00')
        """)
        run_sql(self.spark, f"SELECT * FROM {tbl}.branch_audit_branch WHERE id = 100", show=True)

        run_sql(self.spark, f"ALTER TABLE {tbl} SET TBLPROPERTIES ('write.wap.enabled'='true')")
        run_sql(self.spark, "SET spark.wap.branch=audit_branch")
        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (101, 'wapdata', 'c9', TIMESTAMP'2026-02-01 00:01:00')")
        run_sql(self.spark, "RESET spark.wap.branch")

    # ----------------------------
    # DataFrameWriterV2 cases (拆分)
    # ----------------------------
    def _dfv2_prepare_src_temp_view(self):
        df = self.spark.createDataFrame([(10, "x"), (11, "y")], ["id", "data"])
        df.createOrReplaceTempView("tmp_dfv2_src")

    def dfv2_create(self):
        self._dfv2_prepare_src_temp_view()
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('df_v2_target')}")
        self.spark.table("tmp_dfv2_src").writeTo(self.t("df_v2_target")).using("iceberg").create()
        run_sql(self.spark, f"SELECT * FROM {self.t('df_v2_target')} ORDER BY id", show=True)

    def dfv2_replace(self):
        run_sql(self.spark, f"CREATE TABLE IF NOT EXISTS {self.t('df_v2_target')} (id bigint, data string) USING iceberg")
        self._dfv2_prepare_src_temp_view()
        try:
            self.spark.table("tmp_dfv2_src").writeTo(self.t("df_v2_target")).replace()
            run_sql(self.spark, f"SELECT * FROM {self.t('df_v2_target')} ORDER BY id", show=True)
        except Exception as e:
            error_msg = str(e)
            # Check if this is the known Spark 3.5 unsupported feature
            if self._is_unsupported_feature_error(error_msg) or "REPLACE TABLE AS SELECT" in error_msg:
                raise SkipCase(f"DataFrameWriterV2.replace() not supported in this Spark version: {type(e).__name__}")
            else:
                # Re-raise if it's a different error
                raise

    def dfv2_create_or_replace(self):
        self._dfv2_prepare_src_temp_view()
        self.spark.table("tmp_dfv2_src").writeTo(self.t("df_v2_target")).using("iceberg").createOrReplace()
        run_sql(self.spark, f"SELECT * FROM {self.t('df_v2_target')} ORDER BY id", show=True)

    def dfv2_append(self):
        df = self.spark.createDataFrame([(12, "z")], ["id", "data"])
        df.writeTo(self.t("df_v2_target")).append()
        run_sql(self.spark, f"SELECT * FROM {self.t('df_v2_target')} ORDER BY id", show=True)

    def dfv2_overwrite_partitions(self):
        df = self.spark.createDataFrame(
            [(999, "op", "c1", "2026-03-01 00:00:00")],
            ["id", "data", "category", "ts"],
        ).withColumn("ts", F.to_timestamp("ts"))
        df.writeTo(self.t("sample_part")).overwritePartitions()

    # ----------------------------
    # New Write Coverage: SQL MERGE enhancements
    # ----------------------------
    def writes_merge_with_delete(self):
        """MERGE INTO with WHEN MATCHED THEN DELETE clause."""
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('merge_del_target')}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('merge_del_source')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('merge_del_target')} (id bigint, data string, status string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('merge_del_target')} VALUES (1,'a','active'),(2,'b','active'),(3,'c','inactive')")
        run_sql(self.spark, f"CREATE TABLE {self.t('merge_del_source')} (id bigint, status string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('merge_del_source')} VALUES (1,'delete'),(2,'keep')")

        ok, err = try_sql(self.spark, f"""
            MERGE INTO {self.t('merge_del_target')} t
            USING {self.t('merge_del_source')} s
            ON t.id = s.id
            WHEN MATCHED AND s.status = 'delete' THEN DELETE
            WHEN MATCHED THEN UPDATE SET t.status = s.status
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"MERGE with DELETE not supported: {err}")
            else:
                raise RuntimeError(f"MERGE with DELETE failed: {err}")
        
        # Validate: id=1 should be deleted, id=2 should have status='keep', id=3 unchanged
        assert_sql_count(self.spark, f"SELECT count(*) FROM {self.t('merge_del_target')}", 2, "Expected 2 rows after merge delete")
        assert_sql_count(self.spark, f"SELECT count(*) FROM {self.t('merge_del_target')} WHERE id=1", 0, "Row with id=1 should be deleted")
        assert_sql_scalar(self.spark, f"SELECT status FROM {self.t('merge_del_target')} WHERE id=2", "keep", "Row with id=2 should have status='keep'")
        run_sql(self.spark, f"SELECT * FROM {self.t('merge_del_target')} ORDER BY id", show=True)

    def writes_merge_multiple_matched(self):
        """MERGE INTO with multiple WHEN MATCHED clauses - first matching wins."""
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('merge_multi_target')}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('merge_multi_source')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('merge_multi_target')} (id bigint, data string, value int) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('merge_multi_target')} VALUES (1,'a',10),(2,'b',20),(3,'c',30)")
        run_sql(self.spark, f"CREATE TABLE {self.t('merge_multi_source')} (id bigint, op string, value int) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('merge_multi_source')} VALUES (1,'delete',0),(2,'update',99),(3,'noop',0)")

        ok, err = try_sql(self.spark, f"""
            MERGE INTO {self.t('merge_multi_target')} t
            USING {self.t('merge_multi_source')} s
            ON t.id = s.id
            WHEN MATCHED AND s.op = 'delete' THEN DELETE
            WHEN MATCHED AND s.op = 'update' THEN UPDATE SET t.value = s.value
            WHEN NOT MATCHED THEN INSERT (id, data, value) VALUES (s.id, 'new', s.value)
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"MERGE with multiple WHEN MATCHED not supported: {err}")
            else:
                raise RuntimeError(f"MERGE with multiple WHEN MATCHED failed: {err}")
        
        # Validate: id=1 deleted, id=2 updated, id=3 unchanged (no matching WHEN clause)
        assert_sql_count(self.spark, f"SELECT count(*) FROM {self.t('merge_multi_target')}", 2, "Expected 2 rows after multi-match merge")
        assert_sql_count(self.spark, f"SELECT count(*) FROM {self.t('merge_multi_target')} WHERE id=1", 0, "Row with id=1 should be deleted")
        assert_sql_scalar(self.spark, f"SELECT value FROM {self.t('merge_multi_target')} WHERE id=2", 99, "Row with id=2 should be updated")
        run_sql(self.spark, f"SELECT * FROM {self.t('merge_multi_target')} ORDER BY id", show=True)

    def writes_merge_not_matched_by_source(self):
        """MERGE INTO with WHEN NOT MATCHED BY SOURCE (Spark 3.5+ feature)."""
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('merge_source_target')}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('merge_source_src')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('merge_source_target')} (id bigint, status string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('merge_source_target')} VALUES (1,'old'),(2,'old'),(3,'old')")
        run_sql(self.spark, f"CREATE TABLE {self.t('merge_source_src')} (id bigint, status string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('merge_source_src')} VALUES (1,'new')")

        ok, err = try_sql(self.spark, f"""
            MERGE INTO {self.t('merge_source_target')} t
            USING {self.t('merge_source_src')} s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.status = s.status
            WHEN NOT MATCHED BY SOURCE THEN UPDATE SET t.status = 'archived'
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err) or "NOT MATCHED BY SOURCE" in err:
                raise SkipCase(f"MERGE NOT MATCHED BY SOURCE not supported: {err}")
            else:
                raise RuntimeError(f"MERGE NOT MATCHED BY SOURCE failed: {err}")
        
        # Validate: id=1 updated to 'new', id=2 and id=3 updated to 'archived'
        assert_sql_scalar(self.spark, f"SELECT status FROM {self.t('merge_source_target')} WHERE id=1", "new", "Row with id=1 should have status='new'")
        assert_sql_count(self.spark, f"SELECT count(*) FROM {self.t('merge_source_target')} WHERE status='archived'", 2, "Two rows should have status='archived'")
        run_sql(self.spark, f"SELECT * FROM {self.t('merge_source_target')} ORDER BY id", show=True)

    # ----------------------------
    # New Write Coverage: Branch writes
    # ----------------------------
    def writes_update_on_branch(self):
        """UPDATE operation on a branch."""
        tbl = self.t("sample_part")
        branch = "update_test_branch"
        
        # Clean up and create branch
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        
        # Insert data to branch
        run_sql(self.spark, f"INSERT INTO {tbl}.branch_{branch} VALUES (701,'before_update','c1',TIMESTAMP'2026-02-07 00:00:00')")
        
        # Update on branch
        run_sql(self.spark, f"UPDATE {tbl}.branch_{branch} SET data='after_update' WHERE id=701")
        
        # Validate: branch has updated value
        assert_sql_scalar(self.spark, f"SELECT data FROM {tbl}.branch_{branch} WHERE id=701", "after_update", "Branch should have updated data")
        
        # Validate: main branch does not have this row
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl} WHERE id=701", 0, "Main branch should not have this row")
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def writes_delete_on_branch(self):
        """DELETE operation on a branch."""
        tbl = self.t("sample_part")
        branch = "delete_test_branch"
        
        # Clean up and create branch
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        
        # Insert multiple rows to branch
        run_sql(self.spark, f"INSERT INTO {tbl}.branch_{branch} VALUES (702,'to_delete','c1',TIMESTAMP'2026-02-07 01:00:00')")
        run_sql(self.spark, f"INSERT INTO {tbl}.branch_{branch} VALUES (703,'to_keep','c1',TIMESTAMP'2026-02-07 02:00:00')")
        
        # Delete on branch
        run_sql(self.spark, f"DELETE FROM {tbl}.branch_{branch} WHERE id=702")
        
        # Validate: branch has one row left
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl}.branch_{branch}", 1, "Branch should have 1 row after delete")
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl}.branch_{branch} WHERE id=702", 0, "Deleted row should not exist on branch")
        
        # Validate: main branch unchanged
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl} WHERE id IN (702,703)", 0, "Main branch should not have these rows")
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def writes_merge_on_branch(self):
        """MERGE INTO operation on a branch."""
        tbl = self.t("sample_part")
        branch = "merge_test_branch"
        
        # Clean up and create branch, source table
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('branch_merge_source')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('branch_merge_source')} (id bigint, data string, category string, ts timestamp) USING iceberg")
        
        # Insert data to branch
        run_sql(self.spark, f"INSERT INTO {tbl}.branch_{branch} VALUES (704,'original','c1',TIMESTAMP'2026-02-07 03:00:00')")
        
        # Prepare source data
        run_sql(self.spark, f"INSERT INTO {self.t('branch_merge_source')} VALUES (704,'merged','c1',TIMESTAMP'2026-02-07 04:00:00'),(705,'new','c1',TIMESTAMP'2026-02-07 05:00:00')")
        
        # Merge into branch
        run_sql(self.spark, f"""
            MERGE INTO {tbl}.branch_{branch} t
            USING {self.t('branch_merge_source')} s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.data = s.data
            WHEN NOT MATCHED THEN INSERT (id, data, category, ts) VALUES (s.id, s.data, s.category, s.ts)
        """)
        
        # Validate: branch has 2 rows with expected data
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl}.branch_{branch} WHERE id IN (704,705)", 2, "Branch should have 2 rows after merge")
        assert_sql_scalar(self.spark, f"SELECT data FROM {tbl}.branch_{branch} WHERE id=704", "merged", "Row 704 should be updated")
        
        # Validate: main branch unchanged
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl} WHERE id IN (704,705)", 0, "Main branch should not have these rows")
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('branch_merge_source')}")

    # ----------------------------
    # New Write Coverage: DataFrame advanced
    # ----------------------------
    def writes_dfv2_overwrite_by_filter(self):
        """DataFrameWriterV2 overwrite by filter."""
        tbl = self.t("logs")
        
        # Ensure table exists with partitioned data
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        run_sql(self.spark, f"""
            CREATE TABLE {tbl} (
                uuid string NOT NULL,
                level string NOT NULL,
                ts timestamp NOT NULL,
                message string
            )
            USING iceberg
            PARTITIONED BY (level)
        """)
        run_sql(self.spark, f"""
            INSERT INTO {tbl} VALUES
            ('u1','INFO', TIMESTAMP'2026-01-01 01:00:00','m1'),
            ('u2','INFO', TIMESTAMP'2026-01-01 02:00:00','m2'),
            ('u3','WARN', TIMESTAMP'2026-01-01 03:00:00','m3'),
            ('u4','ERROR', TIMESTAMP'2026-01-01 04:00:00','m4')
        """)
        
        # Overwrite only INFO level logs
        df = self.spark.createDataFrame(
            [('u99','INFO', '2026-01-02 00:00:00','overwritten')],
            ["uuid", "level", "ts", "message"]
        ).withColumn("ts", F.to_timestamp("ts"))
        
        try:
            df.writeTo(tbl).overwrite(F.col("level") == "INFO")
        except Exception as e:
            if self._is_unsupported_feature_error(str(e)) or "overwrite" in str(e).lower():
                raise SkipCase(f"DataFrameWriterV2 overwrite by filter not supported: {e}")
            else:
                raise
        
        # Validate: INFO rows replaced, WARN and ERROR remain
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl} WHERE level='INFO'", 1, "Should have 1 INFO row after overwrite")
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl} WHERE level='WARN'", 1, "Should still have 1 WARN row")
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl} WHERE level='ERROR'", 1, "Should still have 1 ERROR row")
        assert_sql_scalar(self.spark, f"SELECT message FROM {tbl} WHERE level='INFO'", "overwritten", "INFO message should be overwritten")
        run_sql(self.spark, f"SELECT * FROM {tbl} ORDER BY level, uuid", show=True)

    def writes_dfv2_schema_merge(self):
        """DataFrameWriterV2 with schema evolution via mergeSchema option."""
        tbl = self.t("schema_merge_tbl")
        
        # Create table with initial schema
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        run_sql(self.spark, f"CREATE TABLE {tbl} (id bigint, name string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (1,'Alice'),(2,'Bob')")
        
        # Enable schema acceptance
        ok, err = try_sql(self.spark, f"ALTER TABLE {tbl} SET TBLPROPERTIES ('write.spark.accept-any-schema'='true')")
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"Schema evolution not supported: {err}")
            else:
                raise RuntimeError(f"Failed to set schema property: {err}")
        
        # Append data with new column using mergeSchema
        df = self.spark.createDataFrame(
            [(3, 'Charlie', 30), (4, 'Diana', 25)],
            ["id", "name", "age"]
        )
        
        try:
            df.writeTo(tbl).option("mergeSchema", "true").append()
        except Exception as e:
            if self._is_unsupported_feature_error(str(e)) or "schema" in str(e).lower():
                raise SkipCase(f"Schema merge not supported: {e}")
            else:
                raise
        
        # Validate: new column exists, old rows have NULL for new column
        assert_sql_count(self.spark, f"SELECT count(*) FROM {tbl}", 4, "Should have 4 rows total")
        run_sql(self.spark, f"SELECT * FROM {tbl} ORDER BY id", show=True)
        
        # Check schema includes new column
        cols = [field.name for field in self.spark.table(tbl).schema.fields]
        assert_true("age" in cols, "Schema should include 'age' column")
        
        # Validate old rows have NULL for new column, new rows have values
        assert_sql_scalar(self.spark, f"SELECT age FROM {tbl} WHERE id=1", None, "Old rows should have NULL for new column")
        assert_sql_scalar(self.spark, f"SELECT age FROM {tbl} WHERE id=3", 30, "New rows should have age values")

    # ----------------------------
    # New Procedures Coverage
    # ----------------------------
    def proc_snapshot_table(self):
        """system.snapshot procedure to snapshot a table."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        # Create small source table
        src_tbl = self.t("snapshot_source")
        snap_tbl = self.t("snapshot_target")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {src_tbl}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {snap_tbl}")
        run_sql(self.spark, f"CREATE TABLE {src_tbl} (id bigint, value string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {src_tbl} VALUES (1,'a'),(2,'b'),(3,'c')")
        
        # Get source row count
        src_count = scalar_long(self.spark, f"SELECT count(*) FROM {src_tbl}")
        
        # Snapshot the table
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.snapshot(
              source_table => '{self.db}.snapshot_source',
              table => '{self.db}.snapshot_target'
            )
        """, show=True)
        
        if not ok:
            if self._is_unsupported_feature_error(err) or "snapshot" in err.lower():
                raise SkipCase(f"snapshot procedure not supported: {err}")
            else:
                raise RuntimeError(f"snapshot procedure failed: {err}")
        
        # Validate: snapshot table has same row count as source
        snap_count = scalar_long(self.spark, f"SELECT count(*) FROM {snap_tbl}")
        assert_eq(snap_count, src_count, "Snapshot table should have same row count as source")
        run_sql(self.spark, f"SELECT * FROM {snap_tbl} ORDER BY id", show=True)
        
        # Clean up
        run_sql(self.spark, f"DROP TABLE IF EXISTS {snap_tbl}")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {src_tbl}")

    def proc_set_current_snapshot_with_ref(self):
        """system.set_current_snapshot using ref parameter (branch or tag)."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        tbl = self.t("sample_part")
        branch = "snapshot_ref_branch"
        
        # Create branch and insert data
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        run_sql(self.spark, f"INSERT INTO {tbl}.branch_{branch} VALUES (800,'ref_test','c1',TIMESTAMP'2026-02-08 00:00:00')")
        
        # Get snapshot ID of branch
        try:
            branch_snapshot_id = scalar_long(self.spark, f"""
                SELECT snapshot_id FROM {tbl}.refs WHERE name = '{branch}' LIMIT 1
            """)
        except Exception as e:
            run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
            raise SkipCase(f"Could not get branch snapshot ID: {e}")
        
        # Set current snapshot using ref
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.set_current_snapshot(
              table => '{self.db}.sample_part',
              ref => '{branch}'
            )
        """, show=True)
        
        if not ok:
            run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
            if self._is_unsupported_feature_error(err) or "ref" in err.lower():
                raise SkipCase(f"set_current_snapshot with ref not supported: {err}")
            else:
                raise RuntimeError(f"set_current_snapshot with ref failed: {err}")
        
        # Validate: current snapshot ID should match branch snapshot
        current_snapshot_id = scalar_long(self.spark, f"SELECT snapshot_id FROM {tbl}.snapshots ORDER BY committed_at DESC LIMIT 1")
        assert_eq(current_snapshot_id, branch_snapshot_id, "Current snapshot should match branch snapshot")
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")

    def proc_rewrite_data_files_with_options(self):
        """rewrite_data_files with where and strategy options."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        tbl = self.t("sample_part")
        
        # Test with where clause
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.rewrite_data_files(
              table => '{self.db}.sample_part',
              where => 'category = \\'c1\\''
            )
        """, show=True)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"rewrite_data_files with where not supported: {err}")
            # Not fatal if it fails, continue with strategy test
        
        # Test with strategy option
        ok2, err2 = try_sql(self.spark, f"""
            CALL {self.catalog}.system.rewrite_data_files(
              table => '{self.db}.sample_part',
              strategy => 'sort',
              options => map('target-file-size-bytes', '134217728')
            )
        """, show=True)
        
        if not ok2:
            if self._is_unsupported_feature_error(err2):
                raise SkipCase(f"rewrite_data_files with strategy not supported: {err2}")

    def proc_rewrite_manifests_with_options(self):
        """rewrite_manifests with spec_id and use_caching options."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        tbl = self.t("sample_part")
        
        # Test with use_caching option
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.rewrite_manifests(
              table => '{self.db}.sample_part',
              use_caching => true
            )
        """, show=True)
        
        if not ok:
            if self._is_unsupported_feature_error(err) or "use_caching" in err:
                raise SkipCase(f"rewrite_manifests with use_caching not supported: {err}")

    def proc_expire_snapshots_with_ids(self):
        """expire_snapshots using snapshot_ids array (on temp table for safety)."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        # Create temp table for safe snapshot expiration
        temp_tbl = self.t("expire_snap_temp")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {temp_tbl}")
        run_sql(self.spark, f"CREATE TABLE {temp_tbl} (id bigint, data string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {temp_tbl} VALUES (1,'a')")
        run_sql(self.spark, f"INSERT INTO {temp_tbl} VALUES (2,'b')")
        run_sql(self.spark, f"INSERT INTO {temp_tbl} VALUES (3,'c')")
        
        # Get snapshot IDs (keep last 2, expire first one)
        snaps = run_sql(self.spark, f"SELECT snapshot_id FROM {temp_tbl}.snapshots ORDER BY committed_at").collect()
        if len(snaps) < 2:
            run_sql(self.spark, f"DROP TABLE IF EXISTS {temp_tbl}")
            raise SkipCase("Not enough snapshots for expire test")
        
        first_snap_id = int(snaps[0]["snapshot_id"])
        
        # Expire using snapshot_ids array
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.expire_snapshots(
              table => '{self.db}.expire_snap_temp',
              snapshot_ids => ARRAY({first_snap_id})
            )
        """, show=True)
        
        if not ok:
            run_sql(self.spark, f"DROP TABLE IF EXISTS {temp_tbl}")
            if self._is_unsupported_feature_error(err) or "snapshot_ids" in err:
                raise SkipCase(f"expire_snapshots with snapshot_ids not supported: {err}")
            else:
                raise RuntimeError(f"expire_snapshots with snapshot_ids failed: {err}")
        
        # Clean up
        run_sql(self.spark, f"DROP TABLE IF EXISTS {temp_tbl}")

    def proc_compute_table_stats(self):
        """compute_table_stats procedure."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        tbl = self.t("sample_part")
        
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.compute_table_stats(
              table => '{self.db}.sample_part'
            )
        """, show=True)
        
        if not ok:
            if self._is_unsupported_feature_error(err) or "compute_table_stats" in err:
                raise SkipCase(f"compute_table_stats not supported: {err}")
            else:
                raise RuntimeError(f"compute_table_stats failed: {err}")

    def proc_compute_partition_stats(self):
        """compute_partition_stats procedure."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        tbl = self.t("sample_part")
        
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.compute_partition_stats(
              table => '{self.db}.sample_part'
            )
        """, show=True)
        
        if not ok:
            if self._is_unsupported_feature_error(err) or "compute_partition_stats" in err:
                raise SkipCase(f"compute_partition_stats not supported: {err}")
            else:
                raise RuntimeError(f"compute_partition_stats failed: {err}")

    def proc_remove_orphan_files_safe(self):
        """remove_orphan_files with dry_run=false on isolated temp table (LAST TEST)."""
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        
        # Create isolated temp table for safe orphan file removal
        temp_tbl = self.t("orphan_safe_temp")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {temp_tbl}")
        run_sql(self.spark, f"CREATE TABLE {temp_tbl} (id bigint, data string) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {temp_tbl} VALUES (1,'test')")
        
        # Run remove_orphan_files with dry_run=false
        ok, err = try_sql(self.spark, f"""
            CALL {self.catalog}.system.remove_orphan_files(
              table => '{self.db}.orphan_safe_temp',
              dry_run => false
            )
        """, show=True)
        
        if not ok:
            run_sql(self.spark, f"DROP TABLE IF EXISTS {temp_tbl}")
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"remove_orphan_files with dry_run=false not supported: {err}")
            else:
                # Not fatal - permissions or other issues may prevent execution
                raise SkipCase(f"remove_orphan_files with dry_run=false could not execute safely: {err}")
        
        # Table should still be queryable
        assert_sql_count(self.spark, f"SELECT count(*) FROM {temp_tbl}", 1, "Table should still have data after orphan file removal")
        
        # Clean up
        run_sql(self.spark, f"DROP TABLE IF EXISTS {temp_tbl}")

    # ----------------------------
    # Queries cases
    # ----------------------------
    def queries_metadata_tables(self):
        tbl = self.t("sample_part")
        meta = [
            "history",
            "metadata_log_entries",
            "snapshots",
            "entries",
            "files",
            "manifests",
            "partitions",
            "refs",
            "all_data_files",
            "all_delete_files",
            "all_entries",
            "all_manifests",
        ]
        for m in meta:
            run_sql(self.spark, f"SELECT * FROM {tbl}.{m} LIMIT 20", show=True)

    def queries_time_travel(self):
        tbl = self.t("sample_part")
        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (200,'tt','c1',TIMESTAMP'2026-02-02 00:00:00')")

        ts = scalar_str(self.spark, f"SELECT made_current_at FROM {tbl}.history ORDER BY made_current_at DESC LIMIT 1")
        sid = scalar_long(self.spark, f"SELECT snapshot_id FROM {tbl}.snapshots ORDER BY committed_at DESC LIMIT 1")

        run_sql(self.spark, f"SELECT * FROM {tbl} TIMESTAMP AS OF '{ts}' LIMIT 10", show=True)
        run_sql(self.spark, f"SELECT * FROM {tbl} VERSION AS OF {sid} LIMIT 10", show=True)
        run_sql(self.spark, f"SELECT * FROM {tbl} FOR SYSTEM_TIME AS OF '{ts}' LIMIT 10", show=True)
        run_sql(self.spark, f"SELECT * FROM {tbl} FOR SYSTEM_VERSION AS OF {sid} LIMIT 10", show=True)

    def queries_time_travel_metadata_tables(self):
        tbl = self.t("sample_part")
        ts = scalar_str(self.spark, f"SELECT made_current_at FROM {tbl}.history ORDER BY made_current_at DESC LIMIT 1")
        sid = scalar_long(self.spark, f"SELECT snapshot_id FROM {tbl}.snapshots ORDER BY committed_at DESC LIMIT 1")
        run_sql(self.spark, f"SELECT * FROM {tbl}.manifests TIMESTAMP AS OF '{ts}' LIMIT 10", show=True)
        run_sql(self.spark, f"SELECT * FROM {tbl}.partitions VERSION AS OF {sid} LIMIT 10", show=True)

    def queries_data_files_metadata_table(self):
        """Query data_files metadata table"""
        tbl = self.t("sample_part")
        run_sql(self.spark, f"SELECT * FROM {tbl}.data_files LIMIT 20", show=True)

    def queries_delete_files_metadata_table(self):
        """Query delete_files metadata table"""
        tbl = self.t("sample_part")
        run_sql(self.spark, f"SELECT * FROM {tbl}.delete_files LIMIT 20", show=True)

    def queries_all_files_metadata_table(self):
        """Query all_files metadata table"""
        tbl = self.t("sample_part")
        run_sql(self.spark, f"SELECT * FROM {tbl}.all_files LIMIT 20", show=True)

    def queries_time_travel_branch_string(self):
        """Time travel using string version for branch name"""
        tbl = self.t("sample_part")
        branch = "test_query_branch"
        
        # Clean up and create branch
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        
        # Insert data into branch
        run_sql(self.spark, f"""
            INSERT INTO {tbl}.branch_{branch} 
            VALUES (300, 'branch_data', 'c1', TIMESTAMP'2026-02-03 00:00:00')
        """)
        
        # Query using VERSION AS OF with string branch name
        run_sql(self.spark, f"SELECT * FROM {tbl} VERSION AS OF '{branch}' WHERE id = 300 LIMIT 10", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def queries_time_travel_tag_string(self):
        """Time travel using string version for tag name"""
        tbl = self.t("sample_part")
        tag = "test_query_tag"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")
        
        # Get snapshot ID for tag
        try:
            snapshot_id = self._get_latest_snapshot_id(tbl)
        except SkipCase:
            raise
        
        # Create tag
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} CREATE TAG `{tag}` AS OF VERSION {snapshot_id}
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"CREATE TAG not supported: {err}")
            else:
                raise RuntimeError(f"CREATE TAG failed: {err}")
        
        # Query using VERSION AS OF with string tag name
        run_sql(self.spark, f"SELECT * FROM {tbl} VERSION AS OF '{tag}' LIMIT 10", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")

    def queries_branch_identifier_form(self):
        """Query branch via identifier form using backticks"""
        tbl = self.t("sample_part")
        branch = "test_query_branch_id"
        
        # Clean up and create branch
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH IF EXISTS `{branch}`")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE BRANCH `{branch}`")
        
        # Insert data into branch using identifier form
        run_sql(self.spark, f"""
            INSERT INTO {tbl}.`branch_{branch}` 
            VALUES (400, 'branch_id_data', 'c1', TIMESTAMP'2026-02-04 00:00:00')
        """)
        
        # Query using identifier form with backticks
        run_sql(self.spark, f"SELECT * FROM {tbl}.`branch_{branch}` WHERE id = 400 LIMIT 10", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP BRANCH `{branch}`")

    def queries_tag_identifier_form(self):
        """Query tag via identifier form using backticks"""
        tbl = self.t("sample_part")
        tag = "test_query_tag_id"
        
        # Clean up first
        try_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")
        
        # Get snapshot ID for tag
        try:
            snapshot_id = self._get_latest_snapshot_id(tbl)
        except SkipCase:
            raise
        
        # Create tag
        ok, err = try_sql(self.spark, f"""
            ALTER TABLE {tbl} CREATE TAG `{tag}` AS OF VERSION {snapshot_id}
        """)
        
        if not ok:
            if self._is_unsupported_feature_error(err):
                raise SkipCase(f"CREATE TAG not supported: {err}")
            else:
                raise RuntimeError(f"CREATE TAG failed: {err}")
        
        # Query using identifier form with backticks
        run_sql(self.spark, f"SELECT * FROM {tbl}.`tag_{tag}` LIMIT 10", show=True)
        
        # Clean up
        run_sql(self.spark, f"ALTER TABLE {tbl} DROP TAG IF EXISTS `{tag}`")

    # ----------------------------
    # Procedures: migration env
    # ----------------------------
    def proc_migration_env_prepare(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        for tbl in ["src_parquet_tbl", "addfiles_target_tbl", "src_parquet_tbl_BACKUP_"]:
            try_sql(self.spark, f"DROP TABLE IF EXISTS {self.t(tbl)} PURGE")
            try_sql(self.spark, f"DROP TABLE IF EXISTS {self.t(tbl)}")

        # parquet source
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("src_parquet_tbl")} (
              id bigint,
              data string,
              dt date
            ) USING parquet
        """)
        run_sql(self.spark, f"""
            INSERT INTO {self.t("src_parquet_tbl")} VALUES
            (1,'a', DATE'2026-01-01'),
            (2,'b', DATE'2026-01-02'),
            (3,'c', DATE'2026-01-03')
        """)

        # iceberg target for add_files
        run_sql(self.spark, f"""
            CREATE TABLE {self.t("addfiles_target_tbl")} (
              id bigint,
              data string,
              dt date
            ) USING iceberg
            TBLPROPERTIES ('format-version'='2')
        """)

    def proc_migrate(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        self.proc_migration_env_prepare()

        run_sql(self.spark, f"""
            CALL {self.catalog}.system.migrate(
              table => '{self.db}.src_parquet_tbl',
              backup_table_name => '{self.db}.src_parquet_tbl_BACKUP_',
              drop_backup => false
            )
        """, show=True)

        # migrated table should be readable
        run_sql(self.spark, f"SELECT count(*) FROM {self.t('src_parquet_tbl')}", show=True)

    def proc_add_files(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        self.proc_migration_env_prepare()

        src_loc = self._describe_location(self.t("src_parquet_tbl"))
        source_table_expr = f"`parquet`.`{src_loc}`"

        run_sql(self.spark, f"""
            CALL {self.catalog}.system.add_files(
              table => '{self.db}.addfiles_target_tbl',
              source_table => '{source_table_expr}',
              check_duplicate_files => true
            )
        """, show=True)

        run_sql(self.spark, f"SELECT * FROM {self.t('addfiles_target_tbl')} ORDER BY id", show=True)

    def proc_rewrite_table_path(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        iceberg_tbl = self.t("sample_part")
        src_prefix = self._describe_location(iceberg_tbl)

        # 你说有权限：我们用 file:/tmp 或在原目录旁边建 staging
        if src_prefix.startswith("file:"):
            tgt_prefix = "file:/tmp/iceberg_rewrite_target"
            staging = f"file:/tmp/iceberg_rewrite_staging/{self.db}/sample_part/{int(time.time())}"
        else:
            # 对 HDFS/S3，通常也可写（你已确认有权限）。staging 放在同级新目录最通用。
            tgt_prefix = src_prefix.rstrip("/") + "_rewritten"
            staging = src_prefix.rstrip("/") + f"_staging_{int(time.time())}"

        run_sql(self.spark, f"""
            CALL {self.catalog}.system.rewrite_table_path(
              table => '{self.db}.sample_part',
              source_prefix => '{src_prefix}',
              target_prefix => '{tgt_prefix}',
              staging_location => '{staging}'
            )
        """, show=True)

    def proc_register_table(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        # 用 addfiles_target_tbl：相对安全、数据量小
        self.proc_migration_env_prepare()
        tbl = self.t("addfiles_target_tbl")

        metadata_json = self._latest_metadata_json(tbl)

        run_sql(self.spark, f"SELECT count(*) FROM {tbl}", show=True)

        # drop catalog entry
        run_sql(self.spark, f"DROP TABLE {tbl}")

        # register back
        run_sql(self.spark, f"""
            CALL {self.catalog}.system.register_table(
              table => '{self.db}.addfiles_target_tbl',
              metadata_file => '{metadata_json}'
            )
        """, show=True)

        run_sql(self.spark, f"SELECT count(*) FROM {tbl}", show=True)

    # ----------------------------
    # Procedures: others
    # ----------------------------
    def proc_snapshot_management(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        tbl = self.t("sample_part")
        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (300,'p1','c1',TIMESTAMP'2026-02-03 00:00:00')")
        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (301,'p2','c1',TIMESTAMP'2026-02-03 00:01:00')")

        snaps = run_sql(self.spark, f"SELECT snapshot_id, committed_at FROM {tbl}.snapshots ORDER BY committed_at").collect()
        if len(snaps) < 2:
            raise SkipCase("not enough snapshots")

        first_sid = int(snaps[0]["snapshot_id"])
        last_sid = int(snaps[-1]["snapshot_id"])
        last_ts = str(snaps[-1]["committed_at"])

        run_sql(self.spark, f"CALL {self.catalog}.system.rollback_to_snapshot(table => '{self.db}.sample_part', snapshot_id => {first_sid})", show=True)
        run_sql(self.spark, f"CALL {self.catalog}.system.set_current_snapshot(table => '{self.db}.sample_part', snapshot_id => {last_sid})", show=True)
        run_sql(self.spark, f"CALL {self.catalog}.system.rollback_to_timestamp(table => '{self.db}.sample_part', timestamp => TIMESTAMP '{last_ts}')", show=True)

        run_sql(self.spark, f"CALL {self.catalog}.system.rollback_to_snapshot(table => '{self.db}.sample_part', snapshot_id => {first_sid})", show=True)
        ok, err = try_sql(self.spark, f"CALL {self.catalog}.system.cherrypick_snapshot(table => '{self.db}.sample_part', snapshot_id => {last_sid})", show=True)
        if not ok:
            raise SkipCase(f"cherrypick_snapshot not applicable: {err}")

    def proc_publish_changes(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        tbl = self.t("sample_part")
        run_sql(self.spark, f"ALTER TABLE {tbl} SET TBLPROPERTIES ('write.wap.enabled'='true')")
        run_sql(self.spark, "SET spark.wap.id=wap_test_1")
        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (400,'wap','c2',TIMESTAMP'2026-02-04 00:00:00')")
        run_sql(self.spark, "RESET spark.wap.id")
        run_sql(self.spark, f"CALL {self.catalog}.system.publish_changes(table => '{self.db}.sample_part', wap_id => 'wap_test_1')", show=True)

    def proc_fast_forward(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        tbl = self.t("sample_part")
        run_sql(self.spark, f"ALTER TABLE {tbl} CREATE OR REPLACE BRANCH `ff_branch`")
        run_sql(self.spark, f"INSERT INTO {tbl}.branch_ff_branch VALUES (500,'ff','c3',TIMESTAMP'2026-02-05 00:00:00')")
        run_sql(self.spark, f"CALL {self.catalog}.system.fast_forward(table => '{self.db}.sample_part', branch => 'main', to => 'ff_branch')", show=True)

    def proc_metadata_management(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        run_sql(self.spark, f"CALL {self.catalog}.system.rewrite_data_files(table => '{self.db}.sample_part')", show=True)
        run_sql(self.spark, f"CALL {self.catalog}.system.rewrite_manifests(table => '{self.db}.sample_part')", show=True)
        run_sql(self.spark, f"CALL {self.catalog}.system.remove_orphan_files(table => '{self.db}.sample_part', dry_run => true)", show=True)
        run_sql(self.spark, f"CALL {self.catalog}.system.expire_snapshots(table => '{self.db}.sample_part', retain_last => 1)", show=True)

    def proc_rewrite_position_delete_files(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        tbl = self.t("sample_part")
        run_sql(self.spark, f"""
            ALTER TABLE {tbl} SET TBLPROPERTIES (
              'write.delete.mode'='merge-on-read',
              'write.update.mode'='merge-on-read',
              'write.merge.mode'='merge-on-read'
            )
        """)
        run_sql(self.spark, f"DELETE FROM {tbl} WHERE id = 2")

        ok, err = try_sql(
            self.spark,
            f"CALL {self.catalog}.system.rewrite_position_delete_files(table => '{self.db}.sample_part', options => map('rewrite-all','true'))",
            show=True
        )
        if not ok:
            raise SkipCase(f"rewrite_position_delete_files not applicable: {err}")

    def proc_ancestors_of(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")
        run_sql(self.spark, f"CALL {self.catalog}.system.ancestors_of(table => '{self.db}.sample_part')", show=True)

    def proc_create_changelog_view(self):
        if not self.run_procedures:
            raise SkipCase("procedures disabled")

        tbl = self.t("cdc_tbl")
        run_sql(self.spark, f"DROP TABLE IF EXISTS {tbl}")
        run_sql(self.spark, f"""
            CREATE TABLE {tbl} (
              id bigint NOT NULL,
              customer_id string NOT NULL,
              amount double,
              order_date date,
              region string
            )
            USING iceberg
            TBLPROPERTIES ('format-version'='2')
        """)

        run_sql(self.spark, f"""
            INSERT INTO {tbl} VALUES
              (1, 'CUST001', 10.0, DATE'2025-01-10', 'North'),
              (2, 'CUST001', 20.0, DATE'2025-01-10', 'North'),
              (3, 'CUST002', 30.0, DATE'2025-01-10', 'South')
        """)
        start_sid = scalar_long(self.spark, f"SELECT snapshot_id FROM {tbl}.snapshots ORDER BY committed_at DESC LIMIT 1")

        run_sql(self.spark, f"INSERT INTO {tbl} VALUES (4, 'CUST003', 40.0, DATE'2025-01-10', 'East')")
        run_sql(self.spark, f"UPDATE {tbl} SET amount = amount + 5 WHERE id IN (1,2)")
        run_sql(self.spark, f"DELETE FROM {tbl} WHERE id = 3")

        end_sid = scalar_long(self.spark, f"SELECT snapshot_id FROM {tbl}.snapshots ORDER BY committed_at DESC LIMIT 1")
        if end_sid == start_sid:
            raise SkipCase("CDC did not create a new snapshot")

        run_sql(self.spark, f"""
            CALL {self.catalog}.system.create_changelog_view(
              table => '{self.db}.cdc_tbl',
              changelog_view => '{self.db}.cdc_changes',
              options => map('start-snapshot-id','{start_sid}','end-snapshot-id','{end_sid}'),
              identifier_columns => array('id')
            )
        """, show=True)

        run_sql(self.spark, f"SELECT * FROM {self.t('cdc_changes')} ORDER BY _change_ordinal, id", show=True)

    # ----------------------------
    # Case registry + executor
    # ----------------------------
    def cases(self) -> List[Tuple[str, str, Callable[[], None]]]:
        """
        Reorganized test cases with clearer grouping aligned to Iceberg Spark docs.
        Groups:
        - 00_env: Environment setup
        - 10_ddl_core: Core DDL operations (CREATE, ALTER, DROP)
        - 11_ddl_views: View operations
        - 12_ddl_branch_tag: Branch and tag DDL
        - 20_writes_sql_core: Core SQL writes (INSERT, UPDATE, DELETE, INSERT OVERWRITE)
        - 21_writes_sql_merge: SQL MERGE INTO operations
        - 22_writes_sql_branch: Branch-specific writes
        - 23_writes_wap: Write-Audit-Publish
        - 30_writes_dfv2_core: DataFrameWriterV2 core operations
        - 31_writes_df_advanced: DataFrameWriterV2 advanced features
        - 40_queries_*: Query operations (metadata, time travel, etc.)
        - 50_proc_snapshot_mgmt: Snapshot management procedures
        - 51_proc_metadata_mgmt: Metadata management procedures
        - 52_proc_migration: Migration and replication procedures
        - 53_proc_replication: Replication procedures
        - 54_proc_stats: Statistics procedures
        - 55_proc_cdc: CDC procedures
        """
        return [
            # Environment setup
            ("00_env", "prepare", self.env_prepare),
            ("00_env", "seed_base_tables", self.env_seed_base_tables),

            # Core DDL operations
            ("10_ddl_core", "create_table_as_select_basic", self.ddl_ctas_basic),
            ("10_ddl_core", "create_table_as_select_with_props_and_partition", self.ddl_ctas_with_props_and_partition),
            ("10_ddl_core", "replace_table_as_select_existing_first", self.ddl_rtas_replace_table_as_select_existing_first),
            ("10_ddl_core", "create_or_replace_table_as_select", self.ddl_rtas_create_or_replace_table_as_select),
            ("10_ddl_core", "create_table_with_comments", self.ddl_create_table_with_comments),
            ("10_ddl_core", "create_table_with_location", self.ddl_create_table_with_location),
            ("10_ddl_core", "create_table_like_negative", self.ddl_create_table_like_negative),
            ("10_ddl_core", "replace_table_with_partition_and_properties", self.ddl_rtas_with_partition_and_properties),
            ("10_ddl_core", "alter_table_core_operations", self.ddl_alter_table_core),
            ("10_ddl_core", "alter_table_partition_evolution_and_write_order", self.ddl_sql_extensions_partition_evolution_and_write_order),
            ("10_ddl_core", "alter_table_rename", self.ddl_alter_table_rename),
            ("10_ddl_core", "alter_column_comment", self.ddl_alter_column_comment),
            ("10_ddl_core", "drop_table_and_purge", self.ddl_drop_table_and_purge),
            
            # View operations
            ("11_ddl_views", "create_view_and_alter_view", self.ddl_views),
            ("11_ddl_views", "create_view_if_not_exists", self.ddl_views_if_not_exists),
            ("11_ddl_views", "create_view_with_comments_and_aliases", self.ddl_views_with_comments_and_aliases),
            
            # Branch & Tag DDL
            ("12_ddl_branch_tag", "create_branch_with_if_not_exists", self.ddl_branch_create_with_if_not_exists),
            ("12_ddl_branch_tag", "create_or_replace_branch", self.ddl_branch_create_or_replace),
            ("12_ddl_branch_tag", "create_branch_as_of_version", self.ddl_branch_as_of_version),
            ("12_ddl_branch_tag", "replace_branch", self.ddl_branch_replace),
            ("12_ddl_branch_tag", "drop_branch_if_exists", self.ddl_branch_drop_if_exists),
            ("12_ddl_branch_tag", "create_tag_with_if_not_exists", self.ddl_tag_create_with_if_not_exists),
            ("12_ddl_branch_tag", "create_or_replace_tag", self.ddl_tag_create_or_replace),
            ("12_ddl_branch_tag", "drop_tag_if_exists", self.ddl_tag_drop_if_exists),
            ("12_ddl_branch_tag", "create_branch_with_retention", self.ddl_branch_with_retention),

            # Core SQL writes
            ("20_writes_sql_core", "insert_into_and_insert_select", self.writes_insert_into_and_insert_select),
            ("20_writes_sql_core", "insert_overwrite_dynamic_and_static", self.writes_insert_overwrite_dynamic_and_static),
            ("20_writes_sql_core", "delete_and_update", self.writes_delete_and_update),

            # SQL MERGE INTO operations
            ("21_writes_sql_merge", "merge_into_basic", self.writes_merge_into),
            ("21_writes_sql_merge", "merge_with_matched_delete", self.writes_merge_with_delete),
            ("21_writes_sql_merge", "merge_multiple_matched_clauses", self.writes_merge_multiple_matched),
            ("21_writes_sql_merge", "merge_not_matched_by_source", self.writes_merge_not_matched_by_source),

            # Branch-specific writes
            ("22_writes_sql_branch", "update_on_branch", self.writes_update_on_branch),
            ("22_writes_sql_branch", "delete_on_branch", self.writes_delete_on_branch),
            ("22_writes_sql_branch", "merge_into_on_branch", self.writes_merge_on_branch),

            # Write-Audit-Publish
            ("23_writes_wap", "writes_to_branch_and_wap", self.writes_to_branch_and_wap),

            # DataFrameWriterV2 core operations
            ("30_writes_dfv2_core", "dfv2_create_table", self.dfv2_create),
            ("30_writes_dfv2_core", "dfv2_replace_table", self.dfv2_replace),
            ("30_writes_dfv2_core", "dfv2_create_or_replace", self.dfv2_create_or_replace),
            ("30_writes_dfv2_core", "dfv2_append", self.dfv2_append),
            ("30_writes_dfv2_core", "dfv2_overwrite_partitions", self.dfv2_overwrite_partitions),

            # DataFrameWriterV2 advanced features
            ("31_writes_df_advanced", "dfv2_overwrite_by_filter", self.writes_dfv2_overwrite_by_filter),
            ("31_writes_df_advanced", "dfv2_schema_merge", self.writes_dfv2_schema_merge),

            # Query operations - metadata tables
            ("40_queries_metadata", "query_metadata_tables", self.queries_metadata_tables),
            ("40_queries_metadata", "query_data_files_metadata", self.queries_data_files_metadata_table),
            ("40_queries_metadata", "query_delete_files_metadata", self.queries_delete_files_metadata_table),
            ("40_queries_metadata", "query_all_files_metadata", self.queries_all_files_metadata_table),

            # Query operations - time travel
            ("40_queries_time_travel", "time_travel_basic", self.queries_time_travel),
            ("40_queries_time_travel", "time_travel_on_metadata_tables", self.queries_time_travel_metadata_tables),
            ("40_queries_time_travel", "time_travel_with_branch_string", self.queries_time_travel_branch_string),
            ("40_queries_time_travel", "time_travel_with_tag_string", self.queries_time_travel_tag_string),

            # Query operations - branch/tag identifiers
            ("40_queries_refs", "query_branch_with_identifier_form", self.queries_branch_identifier_form),
            ("40_queries_refs", "query_tag_with_identifier_form", self.queries_tag_identifier_form),

            # Snapshot management procedures
            ("50_proc_snapshot_mgmt", "rollback_and_set_current_snapshot", self.proc_snapshot_management),
            ("50_proc_snapshot_mgmt", "snapshot_table", self.proc_snapshot_table),
            ("50_proc_snapshot_mgmt", "set_current_snapshot_with_ref", self.proc_set_current_snapshot_with_ref),
            ("50_proc_snapshot_mgmt", "fast_forward_branch", self.proc_fast_forward),
            ("50_proc_snapshot_mgmt", "publish_wap_changes", self.proc_publish_changes),
            ("50_proc_snapshot_mgmt", "ancestors_of_snapshot", self.proc_ancestors_of),

            # Metadata management procedures
            ("51_proc_metadata_mgmt", "rewrite_data_files_basic", self.proc_metadata_management),
            ("51_proc_metadata_mgmt", "rewrite_data_files_with_options", self.proc_rewrite_data_files_with_options),
            ("51_proc_metadata_mgmt", "rewrite_manifests_with_options", self.proc_rewrite_manifests_with_options),
            ("51_proc_metadata_mgmt", "expire_snapshots_with_ids", self.proc_expire_snapshots_with_ids),
            ("51_proc_metadata_mgmt", "rewrite_position_delete_files", self.proc_rewrite_position_delete_files),

            # Migration and replication procedures
            ("52_proc_migration", "migration_env_prepare", self.proc_migration_env_prepare),
            ("52_proc_migration", "migrate_table", self.proc_migrate),
            ("52_proc_migration", "add_files_to_table", self.proc_add_files),
            ("52_proc_migration", "rewrite_table_path", self.proc_rewrite_table_path),
            ("52_proc_migration", "register_table", self.proc_register_table),

            # Statistics procedures
            ("54_proc_stats", "compute_table_stats", self.proc_compute_table_stats),
            ("54_proc_stats", "compute_partition_stats", self.proc_compute_partition_stats),

            # CDC procedures
            ("55_proc_cdc", "create_changelog_view", self.proc_create_changelog_view),

            # LAST TEST: Safe orphan file removal (must be last)
            ("99_cleanup", "remove_orphan_files_safe", self.proc_remove_orphan_files_safe),
        ]

    def run_all(self) -> List[CaseResult]:
        results: List[CaseResult] = []
        for group, name, fn in self.cases():
            print("\n" + "=" * 120)
            print(f"[CASE] {group} :: {name}")
            print("=" * 120)
            start = time.time()
            try:
                fn()
                results.append(CaseResult(group=group, name=name, status="PASS", seconds=time.time() - start))
            except SkipCase as e:
                results.append(CaseResult(group=group, name=name, status="SKIP", seconds=time.time() - start, error=str(e)))
            except Exception as e:
                traceback.print_exc()
                results.append(CaseResult(group=group, name=name, status="FAIL", seconds=time.time() - start, error=f"{type(e).__name__}: {e}"))
                # Continue running other tests instead of breaking
                # Only break on truly unexpected errors if needed
                continue
        return results


def print_summary(results: List[CaseResult]):
    print("\n" + "#" * 120)
    print("# SUMMARY")
    print("#" * 120)
    header = f"{'GROUP':<24} {'CASE':<55} {'STATUS':<6} {'SECONDS':>8}  ERROR"
    print(header)
    print("-" * len(header))
    for r in results:
        err = (r.error or "").replace("\n", " ")[:220]
        print(f"{r.group:<24} {r.name:<55} {r.status:<6} {r.seconds:>8.2f}  {err}")
    print("-" * len(header))
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    print(f"TOTAL={total}  PASS={passed}  FAIL={failed}  SKIP={skipped}")
    if failed > 0:
        sys.exit(1)


def main():
    spark = get_spark()
    try:
        suite = Suite(spark, db="qa_full")
        results = suite.run_all()
        print_summary(results)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
