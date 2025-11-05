"""Results caching module for material price searches.

Implements SQLite-based caching to avoid redundant API calls and web searches.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheManager:
    """SQLite-based cache for material search results."""
    
    db_path: Path
    cache_ttl_seconds: int = 86400  # 24 hours default
    
    def __post_init__(self) -> None:
        """Initialize database and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Create cache tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS material_cache (
                    material_name TEXT PRIMARY KEY,
                    pt_name TEXT,
                    search_queries TEXT,
                    key_specs TEXT,
                    best_supplier TEXT,
                    price TEXT,
                    url TEXT,
                    reasoning TEXT,
                    suppliers_json TEXT,
                    timestamp REAL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON material_cache(timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_results_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    results_json TEXT,
                    timestamp REAL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.commit()
            logger.info("Cache database initialized at %s", self.db_path)
    
    def get_material(self, material_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached material data if available and not expired.
        
        Args:
            material_name: Name of the material to look up
        
        Returns:
            Cached data dict or None if not found/expired
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM material_cache 
                WHERE material_name = ? 
                AND timestamp > ?
                """,
                (material_name, time.time() - self.cache_ttl_seconds)
            )
            row = cursor.fetchone()
            
            if row:
                logger.info("Cache hit for material '%s'", material_name)
                return {
                    "material_name": row["material_name"],
                    "pt_name": row["pt_name"],
                    "search_queries": json.loads(row["search_queries"]) if row["search_queries"] else [],
                    "key_specs": json.loads(row["key_specs"]) if row["key_specs"] else [],
                    "best_supplier": row["best_supplier"],
                    "price": row["price"],
                    "url": row["url"],
                    "reasoning": row["reasoning"],
                    "suppliers": json.loads(row["suppliers_json"]) if row["suppliers_json"] else [],
                    "timestamp": row["timestamp"],
                }
            
            logger.debug("Cache miss for material '%s'", material_name)
            return None
    
    def set_material(
        self,
        material_name: str,
        pt_name: str,
        search_queries: list,
        key_specs: list,
        best_supplier: str,
        price: str,
        url: str,
        reasoning: str,
        suppliers: list,
    ) -> None:
        """Store material data in cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO material_cache 
                (material_name, pt_name, search_queries, key_specs, best_supplier, 
                 price, url, reasoning, suppliers_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    material_name,
                    pt_name,
                    json.dumps(search_queries, ensure_ascii=False),
                    json.dumps(key_specs, ensure_ascii=False),
                    best_supplier,
                    price,
                    url,
                    reasoning,
                    json.dumps(suppliers, ensure_ascii=False),
                    time.time(),
                )
            )
            conn.commit()
            logger.info("Cached material '%s'", material_name)
    
    def get_search_results(self, query: str) -> Optional[str]:
        """Retrieve cached search results."""
        query_hash = str(hash(query))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT results_json FROM search_results_cache 
                WHERE query_hash = ? 
                AND timestamp > ?
                """,
                (query_hash, time.time() - self.cache_ttl_seconds)
            )
            row = cursor.fetchone()
            
            if row:
                logger.debug("Cache hit for search query '%s'", query)
                return row["results_json"]
            
            return None
    
    def set_search_results(self, query: str, results: str) -> None:
        """Store search results in cache."""
        query_hash = str(hash(query))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO search_results_cache 
                (query_hash, query, results_json, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (query_hash, query, results, time.time())
            )
            conn.commit()
    
    def clear_expired(self) -> int:
        """Remove expired cache entries."""
        cutoff = time.time() - self.cache_ttl_seconds
        
        with sqlite3.connect(self.db_path) as conn:
            cursor1 = conn.execute(
                "DELETE FROM material_cache WHERE timestamp < ?",
                (cutoff,)
            )
            cursor2 = conn.execute(
                "DELETE FROM search_results_cache WHERE timestamp < ?",
                (cutoff,)
            )
            conn.commit()
            
            total_deleted = cursor1.rowcount + cursor2.rowcount
            logger.info("Cleared %d expired cache entries", total_deleted)
            return total_deleted
    
    def clear_all(self) -> None:
        """Clear all cache data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM material_cache")
            conn.execute("DELETE FROM search_results_cache")
            conn.commit()
            logger.info("Cleared all cache data")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM material_cache")
            materials_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM search_results_cache")
            searches_count = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT COUNT(*) FROM material_cache WHERE timestamp > ?",
                (time.time() - self.cache_ttl_seconds,)
            )
            fresh_materials = cursor.fetchone()[0]
            
            return {
                "total_materials": materials_count,
                "total_searches": searches_count,
                "fresh_materials": fresh_materials,
            }


def main() -> None:
    """CLI for cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage material price cache")
    parser.add_argument("--db-path", default="./cache/materials.db", help="Path to cache database")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear-expired", action="store_true", help="Clear expired entries")
    parser.add_argument("--clear-all", action="store_true", help="Clear all cache data")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    cache = CacheManager(db_path=Path(args.db_path))
    
    if args.stats:
        stats = cache.get_stats()
        print("\nCache Statistics:")
        print(f"  Total materials: {stats['total_materials']}")
        print(f"  Fresh materials: {stats['fresh_materials']}")
        print(f"  Total searches: {stats['total_searches']}")
    
    if args.clear_expired:
        deleted = cache.clear_expired()
        print(f"\nCleared {deleted} expired entries")
    
    if args.clear_all:
        cache.clear_all()
        print("\nCleared all cache data")


if __name__ == "__main__":
    main()
