# iceberg-demo

Comprehensive Apache Iceberg integration test suite for Spark.

## Overview

This repository contains `iceberg_it.py`, a comprehensive integration test suite for Apache Iceberg with Spark. The suite validates DDL operations, write operations, queries, procedures, and more across different Spark versions.

## Features

- **59 Test Cases** covering all major Iceberg operations
- **Spark Version Agnostic**: Compatible with both Spark 3.5 and Spark 4
- **HDFS Environment Support**: Tests automatically adapt to HDFS/S3 storage
- **Fail-Safe Design**: Unsupported features are SKIPped instead of failing
- **Comprehensive DDL Coverage**: Aligned with Iceberg documentation

## Test Groups

### Environment Setup (00_env)
- Database and table preparation
- Base table seeding

### DDL Operations (10_ddl, 10_ddl_enhanced)
- CREATE TABLE (basic, with props, partitioned, with comments, with LOCATION)
- CREATE TABLE AS SELECT (CTAS)
- REPLACE TABLE AS SELECT (RTAS) with properties merge
- CREATE TABLE LIKE (negative test)
- ALTER TABLE (columns, properties, partitions, write ordering)
- ALTER TABLE RENAME TO
- ALTER COLUMN COMMENT
- DROP TABLE / PURGE

### Views (10_ddl_views)
- CREATE VIEW / CREATE OR REPLACE VIEW
- CREATE VIEW IF NOT EXISTS
- Views with TBLPROPERTIES
- Views with column comments and aliases
- ALTER VIEW (set/unset properties)

### Branch & Tag DDL (15_branch_tag_ddl)
- CREATE BRANCH / TAG with IF NOT EXISTS
- CREATE OR REPLACE BRANCH / TAG
- CREATE BRANCH AS OF VERSION
- REPLACE BRANCH / TAG
- DROP BRANCH / TAG IF EXISTS
- Branch retention syntax (when supported)

### Write Operations (20_writes_sql)
- INSERT INTO / INSERT SELECT
- MERGE INTO
- INSERT OVERWRITE (dynamic and static)
- DELETE / UPDATE
- Write to branches and WAP (Write-Audit-Publish)

### DataFrameWriterV2 (30_writes_dfv2)
- create()
- replace() - with Spark 3.5 compatibility
- createOrReplace()
- append()
- overwritePartitions()

### Query Operations (40_queries)
- Metadata tables (history, snapshots, files, manifests, data_files, delete_files, all_files, etc.)
- Time travel (TIMESTAMP AS OF, VERSION AS OF with snapshot IDs)
- Time travel with string versions (branch/tag names)
- Time travel on metadata tables
- Branch/tag queries via identifier form with backticks

### Procedures (50_procedures, 55_procedures_migration)
- Migration (migrate, add_files, register_table)
- Snapshot management (rollback, set_current_snapshot, cherrypick)
- Metadata management (rewrite_data_files, rewrite_manifests, expire_snapshots)
- WAP publishing (publish_changes)
- Branch operations (fast_forward)
- CDC (create_changelog_view)
- Other utilities (ancestors_of, remove_orphan_files)

## Key Enhancements

### Helper Methods
- `_infer_hdfs_location()`: Derives writable HDFS paths from existing tables
- `_get_latest_snapshot_id()`: Retrieves snapshot IDs for branch/tag operations
- `_is_unsupported_feature_error()`: Detects version-specific unsupported features

### Spark 3.5 & 4 Compatibility
- Tests that are unsupported in certain versions are automatically SKIPped
- Suite continues execution even if individual tests fail
- `dfv2_replace` handles Spark 3.5 UNSUPPORTED_FEATURE gracefully

### HDFS Environment Support
- LOCATION clauses use inferred HDFS paths, not hardcoded local paths
- Tests skip gracefully if HDFS paths cannot be determined
- Compatible with S3, HDFS, and local file systems

## Running the Suite

```bash
# Basic execution
python iceberg_it.py

# With specific Spark configuration
spark-submit \
  --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0 \
  iceberg_it.py
```

## Test Results

The suite produces a comprehensive summary showing:
- Total test cases executed
- PASS/FAIL/SKIP counts
- Execution time per test
- Error messages for failures

Example output:
```
GROUP                    CASE                                                 STATUS SECONDS  ERROR
00_env                   prepare                                              PASS      0.52  
00_env                   seed_base_tables                                     PASS      2.31  
10_ddl                   ctas_basic                                           PASS      1.45  
...
30_writes_dfv2           dfv2_replace                                         SKIP      0.03  DataFrameWriterV2.replace() not supported in this Spark version
...
TOTAL=59  PASS=54  FAIL=0  SKIP=5
```

## Requirements

- PySpark 3.5+ or 4.0+
- Apache Iceberg Spark runtime
- Hive metastore (configured)
- HDFS/S3 storage (or local for testing)

## Architecture

The suite uses:
- **CaseResult**: Data class for tracking test results
- **SkipCase**: Exception for gracefully skipping unsupported tests
- **Suite**: Main test orchestrator with helper methods
- **try_sql()**: Safe SQL execution with error capture

## Contributing

When adding new test cases:
1. Follow the naming convention: `{group}_{operation}`
2. Use `try_sql()` for version-dependent features
3. Raise `SkipCase` for unsupported operations
4. Clean up test artifacts in teardown
5. Register new tests in `cases()` method

## License

This is a demonstration/testing suite for Apache Iceberg.