# Test Filtering Examples

This document demonstrates the new test filtering functionality added to `iceberg_it.py`.

## Overview

The test suite now supports running individual tests or groups of tests using command-line arguments:
- `--list`: List all available tests
- `--group`: Filter by test group
- `--case`: Filter by test case name (supports partial matching)
- `--db`: Specify custom database name

## Example 1: List All Available Tests

```bash
python iceberg_it.py --list
```

**Output:**
```
========================================================================================================================
AVAILABLE TEST GROUPS AND CASES
========================================================================================================================

[00_env] (2 tests)
  - prepare
  - seed_base_tables

[10_ddl_core] (13 tests)
  - create_table_as_select_basic
  - create_table_as_select_with_props_and_partition
  - replace_table_as_select_existing_first
  - create_or_replace_table_as_select
  - create_table_with_comments
  - create_table_with_location
  - create_table_like_negative
  - replace_table_with_partition_and_properties
  - alter_table_core_operations
  - alter_table_partition_evolution_and_write_order
  - alter_table_rename
  - alter_column_comment
  - drop_table_and_purge

[11_ddl_views] (3 tests)
  - create_view_and_alter_view
  - create_view_if_not_exists
  - create_view_with_comments_and_aliases

... (truncated for brevity)

========================================================================================================================
TOTAL TESTS: 75
TOTAL GROUPS: 17
========================================================================================================================
```

## Example 2: Run All Tests (Default Behavior)

```bash
python iceberg_it.py
```

**Output:**
```
[INFO] Running 75 test(s) out of 75 total

========================================================================================================================
[CASE] 00_env :: prepare
========================================================================================================================
[SQL] CREATE DATABASE IF NOT EXISTS spark_catalog.qa_full
...
```

## Example 3: Run a Specific Group

```bash
python iceberg_it.py --group 10_ddl_core
```

**Output:**
```
[INFO] Running 13 test(s) out of 75 total

========================================================================================================================
[CASE] 10_ddl_core :: create_table_as_select_basic
========================================================================================================================
...
========================================================================================================================
[CASE] 10_ddl_core :: create_table_as_select_with_props_and_partition
========================================================================================================================
...
```

## Example 4: Run a Specific Test Case

```bash
python iceberg_it.py --case create_table_as_select_basic
```

**Output:**
```
[INFO] Running 1 test(s) out of 75 total

========================================================================================================================
[CASE] 10_ddl_core :: create_table_as_select_basic
========================================================================================================================
...
```

## Example 5: Run Tests Matching a Pattern

```bash
python iceberg_it.py --case merge
```

**Output:**
```
[INFO] Running 4 test(s) out of 75 total

========================================================================================================================
[CASE] 21_writes_sql_merge :: merge_into_basic
========================================================================================================================
...
========================================================================================================================
[CASE] 21_writes_sql_merge :: merge_with_matched_delete
========================================================================================================================
...
========================================================================================================================
[CASE] 21_writes_sql_merge :: merge_multiple_matched_clauses
========================================================================================================================
...
========================================================================================================================
[CASE] 22_writes_sql_branch :: merge_into_on_branch
========================================================================================================================
...
```

## Example 6: Combine Group and Case Filters

```bash
python iceberg_it.py --group 20_writes_sql_core --case insert
```

**Output:**
```
[INFO] Running 2 test(s) out of 75 total

========================================================================================================================
[CASE] 20_writes_sql_core :: insert_into_and_insert_select
========================================================================================================================
...
========================================================================================================================
[CASE] 20_writes_sql_core :: insert_overwrite_dynamic_and_static
========================================================================================================================
...
```

## Example 7: No Matching Tests

```bash
python iceberg_it.py --group nonexistent_group
```

**Output:**
```
[WARNING] No tests match the filters: group=nonexistent_group, case=None

########################################################################################################################
# SUMMARY
########################################################################################################################
GROUP                    CASE                                                 STATUS SECONDS  ERROR
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
TOTAL=0  PASS=0  FAIL=0  SKIP=0
```

## Design Rationale

The filtering feature was designed with the following principles:

1. **Minimal Changes**: The implementation adds minimal code to the existing structure
2. **Backward Compatibility**: Running without arguments behaves exactly as before (runs all tests)
3. **Flexible Matching**: Case filter supports partial matching for convenience (e.g., `--case merge` matches all merge-related tests)
4. **Composable Filters**: Group and case filters can be combined for precise test selection
5. **User-Friendly**: The `--list` option helps users discover available tests
6. **Clear Feedback**: Informative messages show how many tests will run

## Use Cases

- **Development**: Run only the tests related to the feature you're working on
- **Debugging**: Quickly re-run a failing test without waiting for the full suite
- **CI/CD**: Split tests across multiple jobs for parallel execution
- **Exploration**: Use `--list` to understand test organization
- **Iterative Testing**: Run a small group of tests repeatedly while debugging
