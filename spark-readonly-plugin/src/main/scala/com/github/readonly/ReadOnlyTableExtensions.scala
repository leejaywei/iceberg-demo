package com.github.readonly

import org.apache.spark.sql.SparkSessionExtensions

/**
 * Entry point for the read-only table protection plugin.
 *
 * Register via Spark config:
 * {{{
 *   spark.sql.extensions=com.github.readonly.ReadOnlyTableExtensions
 * }}}
 *
 * When a table has the property `hive-ext.readOnly` set to `true`,
 * all write and DDL operations on that table will be rejected with
 * an [[org.apache.spark.sql.AnalysisException]].
 */
class ReadOnlyTableExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectCheckRule(session => new ReadOnlyTableCheckRule(session))
  }
}
