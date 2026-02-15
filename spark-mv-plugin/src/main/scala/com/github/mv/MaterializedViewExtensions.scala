package com.github.mv

import org.apache.spark.sql.SparkSessionExtensions

/**
 * Entry point for the materialized view plugin.
 *
 * Register via Spark config:
 * {{{
 *   spark.sql.extensions=com.github.mv.MaterializedViewExtensions
 * }}}
 *
 * Provides:
 *  - CREATE MATERIALIZED VIEW name AS query
 *  - DROP MATERIALIZED VIEW name
 *  - REFRESH MATERIALIZED VIEW name          (full refresh)
 *  - REFRESH MATERIALIZED VIEW name INCREMENTAL  (incremental refresh)
 *  - SHOW MATERIALIZED VIEWS
 *  - Automatic query optimization that rewrites matching queries
 *    to read from the materialized view's backing table.
 */
class MaterializedViewExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    // Custom parser for MV DDL commands
    extensions.injectParser((session, parser) => new MaterializedViewParser(parser))
    // Post-hoc resolution rule for automatic query rewriting
    extensions.injectPostHocResolutionRule(session => new MaterializedViewOptimizationRule(session))
  }
}
