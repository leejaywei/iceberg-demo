# iceberg-demo

Comprehensive Apache Iceberg integration test suite for Spark.

## Overview

This repository contains `iceberg_it.py`, a comprehensive integration test suite for Apache Iceberg with Spark. The suite validates DDL operations, write operations, queries, procedures, and more across different Spark versions.

## Features

- **96 Test Cases** covering all major Iceberg operations including comprehensive data type testing
- **Spark Version Agnostic**: Compatible with both Spark 3.5 and Spark 4
- **HDFS Environment Support**: Tests automatically adapt to HDFS/S3 storage
- **Fail-Safe Design**: Unsupported features are SKIPped instead of failing
- **Comprehensive Coverage**: Aligned with Iceberg Spark Writes and Procedures documentation
- **Built-in Assertions**: Validates expected outcomes with assert_true, assert_eq, assert_sql_count

## Test Groups

### Environment Setup (00_env)
- Database and table preparation
- Base table seeding

### DDL Operations (10_ddl_core)
- CREATE TABLE (basic, with props, partitioned, with comments, with LOCATION)
- CREATE TABLE AS SELECT (CTAS)
- REPLACE TABLE AS SELECT (RTAS) with properties merge
- CREATE TABLE LIKE (negative test)
- ALTER TABLE (columns, properties, partitions, write ordering, rename)
- ALTER COLUMN COMMENT
- DROP TABLE / PURGE

### Views (11_ddl_views)
- CREATE VIEW / CREATE OR REPLACE VIEW
- CREATE VIEW IF NOT EXISTS
- Views with TBLPROPERTIES
- Views with column comments and aliases
- ALTER VIEW (set/unset properties)

### Branch & Tag DDL (12_ddl_branch_tag)
- CREATE BRANCH / TAG with IF NOT EXISTS
- CREATE OR REPLACE BRANCH / TAG
- CREATE BRANCH AS OF VERSION
- REPLACE BRANCH / TAG
- DROP BRANCH / TAG IF EXISTS
- Branch retention syntax (when supported)

### Data Type Tests (15_ddl_data_types)
- Boolean, Byte (tinyint), Short (smallint), Integer, Long (bigint)
- Float, Double, Decimal with precision and scale
- Date, Timestamp
- Char, Varchar, String
- Binary, UUID (when supported), Fixed-length binary
- Struct (nested complex type)
- Array/List collections
- Map collections
- Variant type (when supported)
- Time type (when supported)
- Each type has dedicated table creation, insertion, and validation tests

### Core SQL Writes (20_writes_sql_core)
- INSERT INTO / INSERT SELECT
- INSERT OVERWRITE (dynamic and static)
- DELETE / UPDATE

### SQL MERGE Operations (21_writes_sql_merge)
- Basic MERGE INTO (update + insert)
- MERGE with WHEN MATCHED THEN DELETE
- MERGE with multiple WHEN MATCHED clauses
- MERGE with WHEN NOT MATCHED BY SOURCE (Spark 3.5+)

### Branch-Specific Writes (22_writes_sql_branch)
- UPDATE on branch
- DELETE on branch
- MERGE INTO on branch

### Write-Audit-Publish (23_writes_wap)
- Write to branches with WAP enabled

### DataFrameWriterV2 Core (30_writes_dfv2_core)
- create()
- replace() - with Spark 3.5 compatibility
- createOrReplace()
- append()
- overwritePartitions()

### DataFrameWriterV2 Advanced (31_writes_df_advanced)
- overwrite() with filter condition
- Schema evolution with mergeSchema option

### Query Operations (40_queries_*)
- Metadata tables (history, snapshots, files, manifests, data_files, delete_files, all_files, etc.)
- Time travel (TIMESTAMP AS OF, VERSION AS OF with snapshot IDs)
- Time travel with string versions (branch/tag names)
- Time travel on metadata tables
- Branch/tag queries via identifier form with backticks

### Procedures

#### Snapshot Management (50_proc_snapshot_mgmt)
- rollback_to_snapshot / set_current_snapshot / rollback_to_timestamp
- snapshot (copy table snapshot)
- set_current_snapshot with ref parameter (branch/tag)
- cherrypick_snapshot
- fast_forward (branch)
- publish_changes (WAP)
- ancestors_of

#### Metadata Management (51_proc_metadata_mgmt)
- rewrite_data_files (basic and with where/strategy options)
- rewrite_manifests (with use_caching option)
- expire_snapshots (with snapshot_ids array)
- rewrite_position_delete_files

#### Migration & Replication (52_proc_migration)
- migrate (with backup)
- add_files
- rewrite_table_path
- register_table

#### Statistics (54_proc_stats)
- compute_table_stats
- compute_partition_stats

#### CDC (55_proc_cdc)
- create_changelog_view

#### Cleanup (99_cleanup)
- remove_orphan_files (safe, dry_run=false, last test)

## Key Enhancements

### Assertion Helpers
- `assert_true(cond, msg)`: Assert boolean condition
- `assert_eq(actual, expected, msg)`: Assert equality
- `assert_sql_count(sql, expected, msg)`: Assert query row count
- `assert_sql_scalar(sql, expected, msg)`: Assert scalar query result

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

### Run All Tests

```bash
# Basic execution - runs all tests
python iceberg_it.py

# With specific Spark configuration
spark-submit \
  --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0 \
  iceberg_it.py
```

### Run Specific Tests

The suite supports filtering to run individual tests or test groups:

```bash
# List all available test groups and cases
python iceberg_it.py --list

# Run all tests in a specific group
python iceberg_it.py --group 10_ddl_core

# Run a specific test case (supports partial name matching)
python iceberg_it.py --case create_table_as_select_basic

# Run tests matching a pattern (e.g., all merge-related tests)
python iceberg_it.py --case merge

# Combine group and case filters
python iceberg_it.py --group 20_writes_sql_core --case insert

# Use a custom database name
python iceberg_it.py --db my_test_db --group 10_ddl_core
```

### Command-Line Options

- `--list`: List all available test groups and cases, then exit
- `--group GROUP`: Run only tests in the specified group (e.g., `10_ddl_core`, `20_writes_sql_core`)
- `--case CASE`: Run only tests matching the specified case name (supports partial matching)
- `--db DB`: Database name to use for tests (default: `qa_full`)
- `-h, --help`: Show help message with examples

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
10_ddl_core              create_table_as_select_basic                         PASS      1.45  
...
15_ddl_data_types        create_boolean_table                                 PASS      0.89
15_ddl_data_types        create_struct_table                                  PASS      1.02
...
30_writes_dfv2_core      dfv2_replace_table                                   SKIP      0.03  DataFrameWriterV2.replace() not supported in this Spark version
...
TOTAL=96  PASS=89  FAIL=0  SKIP=7
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