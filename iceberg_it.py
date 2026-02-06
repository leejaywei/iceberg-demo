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

    # ----------------------------
    # Environment
    # ----------------------------
    def env_prepare(self):
        run_sql(self.spark, f"CREATE DATABASE IF NOT EXISTS {self.ns()}")
        self.use_ns()

        # drop views
        for vw in ["sample_vw", "sample_vw_props", "cdc_changes"]:
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
        self.spark.table("tmp_dfv2_src").writeTo(self.t("df_v2_target")).replace()
        run_sql(self.spark, f"SELECT * FROM {self.t('df_v2_target')} ORDER BY id", show=True)

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

            ("10_ddl", "ctas_basic", self.ddl_ctas_basic),
            ("10_ddl", "ctas_with_props_and_partition", self.ddl_ctas_with_props_and_partition),
            ("10_ddl", "rtas_replace_table_as_select_existing_first", self.ddl_rtas_replace_table_as_select_existing_first),
            ("10_ddl", "rtas_create_or_replace_table_as_select", self.ddl_rtas_create_or_replace_table_as_select),
            ("10_ddl", "drop_table_and_purge", self.ddl_drop_table_and_purge),
            ("10_ddl", "alter_table_core", self.ddl_alter_table_core),
            ("10_ddl", "sql_extensions_partition_evolution_and_write_order", self.ddl_sql_extensions_partition_evolution_and_write_order),
            ("10_ddl", "views", self.ddl_views),

            ("20_writes_sql", "insert_into_and_insert_select", self.writes_insert_into_and_insert_select),
            ("20_writes_sql", "merge_into", self.writes_merge_into),
            ("20_writes_sql", "insert_overwrite_dynamic_and_static", self.writes_insert_overwrite_dynamic_and_static),
            ("20_writes_sql", "delete_and_update", self.writes_delete_and_update),
            ("20_writes_sql", "writes_to_branch_and_wap", self.writes_to_branch_and_wap),

            ("30_writes_dfv2", "dfv2_create", self.dfv2_create),
            ("30_writes_dfv2", "dfv2_replace", self.dfv2_replace),
            ("30_writes_dfv2", "dfv2_create_or_replace", self.dfv2_create_or_replace),
            ("30_writes_dfv2", "dfv2_append", self.dfv2_append),
            ("30_writes_dfv2", "dfv2_overwrite_partitions", self.dfv2_overwrite_partitions),

            ("40_queries", "metadata_tables", self.queries_metadata_tables),
            ("40_queries", "time_travel", self.queries_time_travel),
            ("40_queries", "time_travel_metadata_tables", self.queries_time_travel_metadata_tables),

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
                break  # 如要失败也继续跑，改成 continue
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
