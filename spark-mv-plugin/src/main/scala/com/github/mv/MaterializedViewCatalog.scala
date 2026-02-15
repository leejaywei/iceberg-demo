package com.github.mv

import java.util.concurrent.ConcurrentHashMap

import scala.collection.JavaConverters._

/**
 * Metadata for a single materialized view.
 *
 * @param name          Fully qualified name (e.g. "db.mv_name")
 * @param query         The defining SQL query
 * @param backingTable  The physical table storing the MV data
 * @param lastRefreshTs Epoch millis of last refresh, -1 if never refreshed
 */
case class MaterializedViewMeta(
    name: String,
    query: String,
    backingTable: String,
    lastRefreshTs: Long = -1L)

/**
 * In-memory catalog that tracks materialized view metadata.
 *
 * In production this would be backed by a persistent metastore; here we use
 * a thread-safe in-memory map so the plugin is self-contained and requires
 * no external dependencies.
 */
object MaterializedViewCatalog {

  private val views = new ConcurrentHashMap[String, MaterializedViewMeta]()

  /** Normalize name to lowercase for case-insensitive lookup. */
  private def key(name: String): String = name.toLowerCase

  def register(meta: MaterializedViewMeta): Unit =
    views.put(key(meta.name), meta)

  def drop(name: String): Boolean =
    views.remove(key(name)) != null

  def get(name: String): Option[MaterializedViewMeta] =
    Option(views.get(key(name)))

  def exists(name: String): Boolean =
    views.containsKey(key(name))

  def listAll(): Seq[MaterializedViewMeta] =
    views.values().asScala.toSeq

  def updateRefreshTs(name: String, ts: Long): Unit =
    get(name).foreach(m => views.put(key(name), m.copy(lastRefreshTs = ts)))

  /** Remove all entries – useful for tests. */
  def clear(): Unit = views.clear()
}
