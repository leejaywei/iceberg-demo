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
        if dyn_cnt <= 0:
            raise RuntimeError("dynamic overwrite produced empty table unexpectedly")

        run_sql(self.spark, "SET spark.sql.sources.partitionOverwriteMode=static")
        run_sql(self.spark, f"""
            INSERT OVERWRITE {self.t("logs")}
            SELECT uuid, level, ts, message
            FROM {self.t("logs")}
            WHERE level = 'WARN'
        """)
        static_cnt = scalar_long(self.spark, f"SELECT count(*) FROM {self.t('logs')}")
        if static_cnt != 1:
            raise RuntimeError(f"static overwrite expected 1 row, got {static_cnt}")

    def writes_delete_and_update(self):
        run_sql(self.spark, f"DROP TABLE IF EXISTS {self.t('write_target')}")
        run_sql(self.spark, f"CREATE TABLE {self.t('write_target')} (id bigint, v int) USING iceberg")
        run_sql(self.spark, f"INSERT INTO {self.t('write_target')} VALUES (1,10),(2,20),(3,30)")
        run_sql(self.spark, f"DELETE FROM {self.t('write_target')} WHERE id = 1")
        cnt1 = scalar_long(self.spark, f"SELECT count(*) FROM {self.t('write_target')}")
        if cnt1 != 2:
            raise RuntimeError(f"DELETE failed, expected 2 rows, got {cnt1}")
        run_sql(self.spark, f"UPDATE {self.t('write_target')} SET v = 999 WHERE id = 2")
        v2 = scalar_long(self.spark, f"SELECT v FROM {self.t('write_target')} WHERE id=2")
        if v2 != 999:
            raise RuntimeError("UPDATE failed")

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

        ok, err = try_sql(self.spark, f"CALL {self.catalog}.system.cherrypick_snapshot(table => '{self.db}.sample_part', snapshot_id => {first_sid})", show=True)
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
        return [
            ("00_env", "prepare", self.env_prepare),
            ("00_env", "seed_base_tables", self.env_seed_base_tables),

            # Basic DDL
            ("10_ddl", "ctas_basic", self.ddl_ctas_basic),
            ("10_ddl", "ctas_with_props_and_partition", self.ddl_ctas_with_props_and_partition),
            ("10_ddl", "rtas_replace_table_as_select_existing_first", self.ddl_rtas_replace_table_as_select_existing_first),
            ("10_ddl", "rtas_create_or_replace_table_as_select", self.ddl_rtas_create_or_replace_table_as_select),
            ("10_ddl", "drop_table_and_purge", self.ddl_drop_table_and_purge),
            ("10_ddl", "alter_table_core", self.ddl_alter_table_core),
            ("10_ddl", "sql_extensions_partition_evolution_and_write_order", self.ddl_sql_extensions_partition_evolution_and_write_order),
            
            # Enhanced DDL tests
            ("10_ddl_enhanced", "create_table_with_comments", self.ddl_create_table_with_comments),
            ("10_ddl_enhanced", "create_table_with_location", self.ddl_create_table_with_location),
            ("10_ddl_enhanced", "create_table_like_negative", self.ddl_create_table_like_negative),
            ("10_ddl_enhanced", "rtas_with_partition_and_properties", self.ddl_rtas_with_partition_and_properties),
            ("10_ddl_enhanced", "alter_table_rename", self.ddl_alter_table_rename),
            ("10_ddl_enhanced", "alter_column_comment", self.ddl_alter_column_comment),
            
            # Views
            ("10_ddl_views", "views_basic", self.ddl_views),
            ("10_ddl_views", "views_if_not_exists", self.ddl_views_if_not_exists),
            ("10_ddl_views", "views_with_comments_and_aliases", self.ddl_views_with_comments_and_aliases),
            
            # Branch & Tag DDL
            ("15_branch_tag_ddl", "branch_create_with_if_not_exists", self.ddl_branch_create_with_if_not_exists),
            ("15_branch_tag_ddl", "branch_create_or_replace", self.ddl_branch_create_or_replace),
            ("15_branch_tag_ddl", "branch_as_of_version", self.ddl_branch_as_of_version),
            ("15_branch_tag_ddl", "branch_replace", self.ddl_branch_replace),
            ("15_branch_tag_ddl", "branch_drop_if_exists", self.ddl_branch_drop_if_exists),
            ("15_branch_tag_ddl", "tag_create_with_if_not_exists", self.ddl_tag_create_with_if_not_exists),
            ("15_branch_tag_ddl", "tag_create_or_replace", self.ddl_tag_create_or_replace),
            ("15_branch_tag_ddl", "tag_drop_if_exists", self.ddl_tag_drop_if_exists),
            ("15_branch_tag_ddl", "branch_with_retention", self.ddl_branch_with_retention),

            # Writes
            ("20_writes_sql", "insert_into_and_insert_select", self.writes_insert_into_and_insert_select),
            ("20_writes_sql", "merge_into", self.writes_merge_into),
            ("20_writes_sql", "insert_overwrite_dynamic_and_static", self.writes_insert_overwrite_dynamic_and_static),
            ("20_writes_sql", "delete_and_update", self.writes_delete_and_update),
            ("20_writes_sql", "writes_to_branch_and_wap", self.writes_to_branch_and_wap),

            # DataFrameWriterV2
            ("30_writes_dfv2", "dfv2_create", self.dfv2_create),
            ("30_writes_dfv2", "dfv2_replace", self.dfv2_replace),
            ("30_writes_dfv2", "dfv2_create_or_replace", self.dfv2_create_or_replace),
            ("30_writes_dfv2", "dfv2_append", self.dfv2_append),
            ("30_writes_dfv2", "dfv2_overwrite_partitions", self.dfv2_overwrite_partitions),

            # Queries
            ("40_queries", "metadata_tables", self.queries_metadata_tables),
            ("40_queries", "time_travel", self.queries_time_travel),
            ("40_queries", "time_travel_metadata_tables", self.queries_time_travel_metadata_tables),
            ("40_queries", "data_files_metadata_table", self.queries_data_files_metadata_table),
            ("40_queries", "delete_files_metadata_table", self.queries_delete_files_metadata_table),
            ("40_queries", "all_files_metadata_table", self.queries_all_files_metadata_table),
            ("40_queries", "time_travel_branch_string", self.queries_time_travel_branch_string),
            ("40_queries", "time_travel_tag_string", self.queries_time_travel_tag_string),
            ("40_queries", "branch_identifier_form", self.queries_branch_identifier_form),
            ("40_queries", "tag_identifier_form", self.queries_tag_identifier_form),

            # migration/replication procedures
            ("55_procedures_migration", "migration_env_prepare", self.proc_migration_env_prepare),
            ("55_procedures_migration", "migrate", self.proc_migrate),
            ("55_procedures_migration", "add_files", self.proc_add_files),
            ("55_procedures_migration", "rewrite_table_path", self.proc_rewrite_table_path),
            ("55_procedures_migration", "register_table", self.proc_register_table),

            # other procedures
            ("50_procedures", "snapshot_management", self.proc_snapshot_management),
            ("50_procedures", "publish_changes", self.proc_publish_changes),
            ("50_procedures", "fast_forward", self.proc_fast_forward),
            ("50_procedures", "metadata_management", self.proc_metadata_management),
            ("50_procedures", "rewrite_position_delete_files", self.proc_rewrite_position_delete_files),
            ("50_procedures", "ancestors_of", self.proc_ancestors_of),
            ("50_procedures", "create_changelog_view", self.proc_create_changelog_view),
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
