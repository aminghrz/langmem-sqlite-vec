"""SQLite-vec backed store with vector search capabilities."""

import asyncio
import concurrent.futures as cf
import json
import logging
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Optional, Union

from langchain_core.embeddings import Embeddings

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)


def serialize_f32(vector: list[float]) -> bytes:
    """Serializes a list of floats into a compact "raw bytes" format."""
    return struct.pack("%sf" % len(vector), *vector)


def deserialize_f32(data: bytes, dims: int) -> list[float]:
    """Deserializes bytes back into a list of floats."""
    return list(struct.unpack("%sf" % dims, data))


class SqliteVecStore(BaseStore):
    """SQLite-vec backed store with vector search capabilities.
    
    This store provides persistent storage using SQLite with the sqlite-vec extension
    for efficient vector similarity search.
    
    Example:
        Basic key-value storage:
        ```python
        from langgraph.store.sqlite_vec import SqliteVecStore
        
        store = SqliteVecStore(db_file="my_store.db")
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        ```
        
        Vector search with embeddings:
        ```python
        from langchain.embeddings import init_embeddings
        
        store = SqliteVecStore(
            db_file="my_store.db",
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"],
            }
        )
        
        # Store documents
        store.put(("docs",), "doc1", {"text": "Python tutorial"})
        store.put(("docs",), "doc2", {"text": "TypeScript guide"})
        
        # Search by similarity
        results = store.search(("docs",), query="python programming")
        ```
    """
    
    def __init__(
        self,
        *,
        db_file: str = "langgraph_store.db",
        index: Optional[IndexConfig] = None,
    ) -> None:
        """Initialize SQLite-vec store.
        
        Args:
            db_file: Path to SQLite database file
            index: Optional index configuration for vector search
        """
        try:
            import sqlite_vec  # noqa
        except ImportError:
            raise ImportError(
                "Could not import sqlite-vec python package. "
                "Please install it with `pip install sqlite-vec`."
            )
        
        self.db_file = db_file
        self._connection = self._create_connection()
        self.index_config = index
        
        if self.index_config:
            self.index_config = self.index_config.copy()
            self.embeddings: Optional[Embeddings] = ensure_embeddings(
                self.index_config.get("embed"),
            )
            self.index_config["__tokenized_fields"] = [
                (p, tokenize_path(p)) if p != "$" else (p, p)
                for p in (self.index_config.get("fields") or ["$"])
            ]
        else:
            self.index_config = None
            self.embeddings = None
        
        self._setup_tables()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create SQLite connection with vec extension."""
        import sqlite_vec
        
        connection = sqlite3.connect(self.db_file,check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection
    
    def _setup_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        # Main store table
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS store_items (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            )
        """)
        
        # Create index on namespace for efficient filtering
        self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_namespace 
            ON store_items(namespace)
        """)
        
        if self.index_config:
            dims = self.index_config.get("dims", 1536)
            
            # Vector embeddings table
            self._connection.execute(f"""
                CREATE TABLE IF NOT EXISTS store_vectors (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (namespace, key, path),
                    FOREIGN KEY (namespace, key) 
                        REFERENCES store_items(namespace, key) 
                        ON DELETE CASCADE
                )
            """)
            
            # Virtual table for vector search
            self._connection.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_search USING vec0(
                    namespace TEXT,
                    key TEXT,
                    path TEXT,
                    embedding float[{dims}]
                )
            """)
        
        self._connection.commit()
    
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations."""
        results, put_ops, search_ops = self._prepare_ops(ops)
        
        if search_ops:
            query_embeddings = self._embed_search_queries(search_ops)
            self._batch_search(search_ops, query_embeddings, results)
        
        to_embed = self._extract_texts(put_ops)
        if to_embed and self.index_config and self.embeddings:
            embeddings = self.embeddings.embed_documents(list(to_embed))
            self._insert_vectors(to_embed, embeddings)
        
        self._apply_put_ops(put_ops)
        return results
    
    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously."""
        results, put_ops, search_ops = self._prepare_ops(ops)
        
        if search_ops:
            query_embeddings = await self._aembed_search_queries(search_ops)
            self._batch_search(search_ops, query_embeddings, results)
        
        to_embed = self._extract_texts(put_ops)
        if to_embed and self.index_config and self.embeddings:
            embeddings = await self.embeddings.aembed_documents(list(to_embed))
            self._insert_vectors(to_embed, embeddings)
        
        self._apply_put_ops(put_ops)
        return results
    
    def _prepare_ops(
        self, ops: Iterable[Op]
    ) -> tuple[
        list[Result],
        dict[tuple[tuple[str, ...], str], PutOp],
        dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ]:
        """Prepare operations for batch execution."""
        results: list[Result] = []
        put_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        search_ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]] = {}
        
        for i, op in enumerate(ops):
            if isinstance(op, GetOp):
                item = self._get_item(op.namespace, op.key)
                results.append(item)
            elif isinstance(op, SearchOp):
                search_ops[i] = (op, self._filter_items(op))
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            elif isinstance(op, PutOp):
                put_ops[(op.namespace, op.key)] = op
                results.append(None)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        
        return results, put_ops, search_ops
    
    def _get_item(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        """Get a single item from the store."""
        namespace_str = json.dumps(namespace)
        cursor = self._connection.execute(
            "SELECT * FROM store_items WHERE namespace = ? AND key = ?",
            (namespace_str, key)
        )
        row = cursor.fetchone()
        
        if row:
            return Item(
                value=json.loads(row["value"]),
                key=row["key"],
                namespace=json.loads(row["namespace"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
        return None
    
    def _filter_items(self, op: SearchOp) -> list[tuple[Item, list[list[float]]]]:
        """Filter items by namespace and filter function."""
        namespace_prefix = op.namespace_prefix
        namespace_prefix_str = json.dumps(namespace_prefix)
        
        # Query items with namespace prefix
        query = """
            SELECT * FROM store_items 
            WHERE json_extract(namespace, '$[0]') = json_extract(?, '$[0]')
        """
        params = [namespace_prefix_str]
        
        # Add additional namespace prefix checks
        for i in range(1, len(namespace_prefix)):
            query += f" AND json_extract(namespace, '$[{i}]') = json_extract(?, '$[{i}]')"
            params.append(namespace_prefix_str)
        
        cursor = self._connection.execute(query, params)
        
        filtered = []
        for row in cursor:
            item = Item(
                value=json.loads(row["value"]),
                key=row["key"],
                namespace=json.loads(row["namespace"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            
            # Apply filter
            if op.filter and not self._matches_filter(item, op.filter):
                continue
            
            # Get embeddings if needed
            embeddings = []
            if op.query and self.index_config:
                vec_cursor = self._connection.execute(
                    """
                    SELECT path, embedding FROM store_vectors
                    WHERE namespace = ? AND key = ?
                    """,
                    (row["namespace"], row["key"])
                )
                for vec_row in vec_cursor:
                    embedding = deserialize_f32(
                        vec_row["embedding"], 
                        self.index_config["dims"]
                    )
                    embeddings.append(embedding)
            
            filtered.append((item, embeddings))
        
        return filtered
    
    def _matches_filter(self, item: Item, filter_dict: dict[str, Any]) -> bool:
        """Check if item matches filter criteria."""
        for key, filter_value in filter_dict.items():
            if not self._compare_values(item.value.get(key), filter_value):
                return False
        return True
    
    def _compare_values(self, item_value: Any, filter_value: Any) -> bool:
        """Compare values in a JSONB-like way."""
        if isinstance(filter_value, dict):
            if any(k.startswith("$") for k in filter_value):
                return all(
                    self._apply_operator(item_value, op_key, op_value)
                    for op_key, op_value in filter_value.items()
                )
            if not isinstance(item_value, dict):
                return False
            return all(
                self._compare_values(item_value.get(k), v) 
                for k, v in filter_value.items()
            )
        elif isinstance(filter_value, (list, tuple)):
            return (
                isinstance(item_value, (list, tuple))
                and len(item_value) == len(filter_value)
                and all(
                    self._compare_values(iv, fv) 
                    for iv, fv in zip(item_value, filter_value)
                )
            )
        else:
            return item_value == filter_value
    
    def _apply_operator(self, value: Any, operator: str, op_value: Any) -> bool:
        """Apply a comparison operator."""
        if operator == "$eq":
            return value == op_value
        elif operator == "$gt":
            return float(value) > float(op_value)
        elif operator == "$gte":
            return float(value) >= float(op_value)
        elif operator == "$lt":
            return float(value) < float(op_value)
        elif operator == "$lte":
            return float(value) <= float(op_value)
        elif operator == "$ne":
            return value != op_value
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    def _embed_search_queries(
        self,
        search_ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ) -> dict[str, list[float]]:
        """Embed search queries."""
        query_embeddings = {}
        if self.index_config and self.embeddings and search_ops:
            queries = {op.query for (op, _) in search_ops.values() if op.query}
            
            if queries:
                with cf.ThreadPoolExecutor() as executor:
                    futures = {
                        q: executor.submit(self.embeddings.embed_query, q)
                        for q in list(queries)
                    }
                    for query, future in futures.items():
                        query_embeddings[query] = future.result()
        
        return query_embeddings
    
    async def _aembed_search_queries(
        self,
        search_ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ) -> dict[str, list[float]]:
        """Embed search queries asynchronously."""
        query_embeddings = {}
        if self.index_config and self.embeddings and search_ops:
            queries = {op.query for (op, _) in search_ops.values() if op.query}
            
            if queries:
                coros = [self.embeddings.aembed_query(q) for q in list(queries)]
                results = await asyncio.gather(*coros)
                query_embeddings = dict(zip(queries, results))
        
        return query_embeddings
    
    def _batch_search(
        self,
        ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
        query_embeddings: dict[str, list[float]],
        results: list[Result],
    ) -> None:
        """Perform batch similarity search."""
        for i, (op, candidates) in ops.items():
            if not candidates:
                results[i] = []
                continue
            
            if op.query and query_embeddings and self.index_config:
                # Use sqlite-vec for similarity search
                query_embedding = query_embeddings[op.query]
                
                # Get all unique namespace/key pairs from candidates
                candidate_keys = {
                    (json.dumps(item.namespace), item.key): item 
                    for item, _ in candidates
                }
                
                if not candidate_keys:
                    results[i] = []
                    continue
                
                # For each item, find the best matching vector
                item_scores = []
                
                for (ns_str, key), item in candidate_keys.items():
                    # Get the best score for this item across all its vectors
                    cursor = self._connection.execute(
                        """
                        SELECT 
                            distance
                        FROM vec_search
                        WHERE namespace = ? AND key = ?
                        AND embedding MATCH ?
                        ORDER BY distance
                        LIMIT 1
                        """,
                        (ns_str, key, serialize_f32(query_embedding))
                    )
                    
                    row = cursor.fetchone()
                    if row:
                        item_scores.append((row["distance"], item))
                
                # Sort by distance (ascending - lower is better)
                item_scores.sort(key=lambda x: x[0])
                
                # Convert to SearchItems with offset and limit
                search_results = []
                for distance, item in item_scores[op.offset:op.offset + op.limit]:
                    search_results.append(
                        SearchItem(
                            namespace=item.namespace,
                            key=item.key,
                            value=item.value,
                            created_at=item.created_at,
                            updated_at=item.updated_at,
                            score=1.0 - distance,  # Convert distance to similarity
                        )
                    )
                
                results[i] = search_results
            else:
                # No query, just return filtered items
                results[i] = [
                    SearchItem(
                        namespace=item.namespace,
                        key=item.key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                    for (item, _) in candidates[op.offset:op.offset + op.limit]
                ]

    def _apply_put_ops(self, put_ops: dict[tuple[tuple[str, ...], str], PutOp]) -> None:
        """Apply put operations to the database."""
        with self._connection:
            for (namespace, key), op in put_ops.items():
                namespace_str = json.dumps(namespace)
                
                if op.value is None:
                    # Delete item
                    self._connection.execute(
                        "DELETE FROM store_items WHERE namespace = ? AND key = ?",
                        (namespace_str, key)
                    )
                    if self.index_config:
                        self._connection.execute(
                            "DELETE FROM store_vectors WHERE namespace = ? AND key = ?",
                            (namespace_str, key)
                        )
                        self._connection.execute(
                            "DELETE FROM vec_search WHERE namespace = ? AND key = ?",
                            (namespace_str, key)
                        )
                else:
                    # Insert or update item
                    now = datetime.now(timezone.utc).isoformat()
                    value_str = json.dumps(op.value)
                    
                    self._connection.execute(
                        """
                        INSERT OR REPLACE INTO store_items 
                        (namespace, key, value, created_at, updated_at)
                        VALUES (?, ?, ?, 
                            COALESCE((SELECT created_at FROM store_items 
                                     WHERE namespace = ? AND key = ?), ?),
                            ?)
                        """,
                        (namespace_str, key, value_str, namespace_str, key, now, now)
                    )
    
    def _extract_texts(
        self, put_ops: dict[tuple[tuple[str, ...], str], PutOp]
    ) -> dict[str, list[tuple[tuple[str, ...], str, str]]]:
        """Extract texts to embed from put operations."""
        if put_ops and self.index_config and self.embeddings:
            to_embed = defaultdict(list)
            
            for op in put_ops.values():
                if op.value is not None and op.index is not False:
                    if op.index is None:
                        paths = self.index_config["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]
                    
                    for path, field in paths:
                        texts = get_text_at_path(op.value, field)
                        if texts:
                            if len(texts) > 1:
                                for i, text in enumerate(texts):
                                    to_embed[text].append(
                                        (op.namespace, op.key, f"{path}.{i}")
                                    )
                            else:
                                to_embed[texts[0]].append((op.namespace, op.key, path))
            
            return to_embed
        
        return {}
    
    def _insert_vectors(
        self,
        to_embed: dict[str, list[tuple[tuple[str, ...], str, str]]],
        embeddings: list[list[float]],
    ) -> None:
        """Insert vector embeddings into the database."""
        indices = [index for indices in to_embed.values() for index in indices]
        if len(indices) != len(embeddings):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not"
                f" match number of indices ({len(indices)})"
            )
        
        with self._connection:
            # First, delete existing vectors for the items being updated
            for ns, key, path in indices:
                namespace_str = json.dumps(ns)
                self._connection.execute(
                    """
                    DELETE FROM store_vectors 
                    WHERE namespace = ? AND key = ? AND path = ?
                    """,
                    (namespace_str, key, path)
                )
                self._connection.execute(
                    """
                    DELETE FROM vec_search 
                    WHERE namespace = ? AND key = ? AND path = ?
                    """,
                    (namespace_str, key, path)
                )
            
            # Insert new vectors
            for embedding, (ns, key, path) in zip(embeddings, indices):
                namespace_str = json.dumps(ns)
                embedding_blob = serialize_f32(embedding)
                
                # Insert into store_vectors
                self._connection.execute(
                    """
                    INSERT INTO store_vectors (namespace, key, path, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (namespace_str, key, path, embedding_blob)
                )
                
                # Insert into vec_search virtual table
                self._connection.execute(
                    """
                    INSERT INTO vec_search (namespace, key, path, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (namespace_str, key, path, embedding_blob)
                )
    
    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Handle list namespaces operation."""
        query = "SELECT DISTINCT namespace FROM store_items"
        cursor = self._connection.execute(query)
        
        all_namespaces = [json.loads(row["namespace"]) for row in cursor]
        
        namespaces = all_namespaces
        if op.match_conditions:
            namespaces = [
                ns
                for ns in namespaces
                if all(self._does_match(condition, ns) for condition in op.match_conditions)
            ]
        
        if op.max_depth is not None:
            namespaces = sorted({tuple(ns[:op.max_depth]) for ns in namespaces})
        else:
            namespaces = sorted(namespaces)
        
        return namespaces[op.offset:op.offset + op.limit]
    
    def _does_match(self, match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
        """Check if a namespace key matches a match condition."""
        match_type = match_condition.match_type
        path = match_condition.path
        
        if len(key) < len(path):
            return False
        
        if match_type == "prefix":
            for k_elem, p_elem in zip(key, path):
                if p_elem == "*":
                    continue
                if k_elem != p_elem:
                    return False
            return True
        elif match_type == "suffix":
            for k_elem, p_elem in zip(reversed(key), reversed(path)):
                if p_elem == "*":
                    continue
                if k_elem != p_elem:
                    return False
            return True
        else:
            raise ValueError(f"Unsupported match type: {match_type}")
    
    def __del__(self):
        """Close database connection on deletion."""
        if hasattr(self, '_connection'):
            self._connection.close()