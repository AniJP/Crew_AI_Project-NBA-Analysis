"""
NBA Data Analysis Utilities - Consolidated Functions

This module contains all functions from the NBA Analysis project consolidated into a single file.
All functions can be imported and used from a Jupyter notebook.
"""

import os
import re
import time
import hashlib
import threading
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from contextvars import ContextVar
import json
import traceback

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from pydantic import BaseModel, Field


class SearchNbaDataInput(BaseModel):
    """Optional filters — any subset may be omitted by the agent."""

    query: str = Field(default="", description="Optional text to search across all columns")
    column: str = Field(default="", description="Optional column name to filter")
    value: str = Field(default="", description="Optional value to match in that column")
    limit: int = Field(default=50, description="Max rows to return (default 50)")


class LookupPlayerGamesInput(BaseModel):
    """Player lookup — only player is required."""

    player: str = Field(description="Player name or fragment (e.g. Giannis, Fox, LeBron)")
    pts: Optional[float] = Field(default=None, description="Optional exact PTS filter; omit if unused")
    min_pts: Optional[float] = Field(default=None, description="Optional minimum PTS; omit if unused")
    limit: int = Field(default=20, description="Max rows to return (default 20)")

# ============================================================================
# CONFIGURATION
# ============================================================================

def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from .env into os.environ (does not override existing)."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()

# NBA Data Configuration
NBA_DATA_PATH = "nba24-25.csv"

# OpenAI Configuration (ONLY PROVIDER)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ask reliability / caching
ASK_CACHE_TTL_S = int(os.getenv("ASK_CACHE_TTL_S", "3600"))
ASK_MAX_RETRIES = int(os.getenv("ASK_MAX_RETRIES", "2"))
ASK_MAX_TOOL_CALLS = int(os.getenv("ASK_MAX_TOOL_CALLS", "8"))
ASK_GROUNDEDNESS_MIN = float(os.getenv("ASK_GROUNDEDNESS_MIN", "0.5"))


def get_llm() -> LLM:
    """
    Create and return a CrewAI LLM instance configured for OpenAI.
    
    Returns:
        LLM: Configured CrewAI LLM instance for OpenAI
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    model = os.getenv("OPENAI_MODEL") or OPENAI_MODEL
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it using: export OPENAI_API_KEY='your-api-key'"
        )
    return LLM(
        model=model,
        api_key=api_key
    )


# ============================================================================
# VECTOR DATABASE
# ============================================================================

class NBAVectorDB:
    """
    Manages vector embeddings and semantic search for NBA data.
    Uses sentence-transformers for embeddings and ChromaDB for storage.
    """
    
    def __init__(self, csv_path: str, collection_name: str = "nba_data", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector database.
        
        Args:
            csv_path: Path to the NBA CSV file
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
        """
        self.csv_path = csv_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Lazy imports — avoids breaking FastAPI startup when transformers versions clash
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings
        
        # Initialize embedding model (open-source, runs locally)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")
        
        # Initialize ChromaDB client
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "NBA 2024-25 season data"}
        )
        
        # Check if collection is empty and needs indexing
        if self.collection.count() == 0:
            print("Vector database is empty. Indexing CSV data...")
            self._index_csv()
        else:
            print(f"Vector database loaded with {self.collection.count()} records")
    
    def _create_text_representation(self, row: pd.Series) -> str:
        """Convert a DataFrame row to a text representation for embedding."""
        parts = []
        
        if 'Player' in row:
            parts.append(f"Player: {row['Player']}")
        if 'Tm' in row:
            parts.append(f"Team: {row['Tm']}")
        if 'Opp' in row:
            parts.append(f"Opponent: {row['Opp']}")
        if 'Res' in row:
            parts.append(f"Result: {'Win' if row['Res'] == 'W' else 'Loss'}")
        if 'PTS' in row and pd.notna(row['PTS']):
            parts.append(f"Points: {row['PTS']}")
        if 'AST' in row and pd.notna(row['AST']):
            parts.append(f"Assists: {row['AST']}")
        if 'TRB' in row and pd.notna(row['TRB']):
            parts.append(f"Rebounds: {row['TRB']}")
        if 'FG%' in row and pd.notna(row['FG%']):
            parts.append(f"Field Goal Percentage: {row['FG%']:.3f}")
        if '3P%' in row and pd.notna(row['3P%']):
            parts.append(f"Three Point Percentage: {row['3P%']:.3f}")
        if 'Data' in row:
            parts.append(f"Date: {row['Data']}")
        
        return ". ".join(parts)
    
    def _index_csv(self):
        """Read CSV file, create embeddings, and store in ChromaDB."""
        print(f"Reading CSV from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        print(f"Creating embeddings for {len(df)} records...")
        
        # Process in batches for efficiency
        batch_size = 100
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            batch_texts = []
            batch_metadatas = []
            batch_ids = []
            
            for idx, row in batch_df.iterrows():
                text = self._create_text_representation(row)
                batch_texts.append(text)
                
                metadata = {
                    'row_index': int(idx),
                    'player': str(row.get('Player', '')),
                    'team': str(row.get('Tm', '')),
                    'opponent': str(row.get('Opp', '')),
                    'result': str(row.get('Res', '')),
                    'points': float(row.get('PTS', 0)) if pd.notna(row.get('PTS')) else 0.0,
                    'date': str(row.get('Data', '')),
                }
                batch_metadatas.append(metadata)
                batch_ids.append(f"row_{idx}")
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} records)...")
            embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        print(f"Successfully indexed {len(df)} records in vector database!")
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Perform semantic search on the NBA data.
        
        Args:
            query: Natural language query
            n_results: Number of results to return
        
        Returns:
            List of dictionaries containing search results with metadata
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
        
        return formatted_results
    
    def get_original_row(self, row_index: int) -> Optional[pd.Series]:
        """Retrieve the original CSV row by index."""
        try:
            df = pd.read_csv(self.csv_path)
            if 0 <= row_index < len(df):
                return df.iloc[row_index]
        except Exception as e:
            print(f"Error retrieving row {row_index}: {e}")
        return None


# Global vector DB instance
_vector_db_instance: Optional[NBAVectorDB] = None


def get_vector_db(csv_path: str) -> NBAVectorDB:
    """
    Get or create the global vector database instance.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        NBAVectorDB instance
    """
    global _vector_db_instance
    if _vector_db_instance is None or _vector_db_instance.csv_path != csv_path:
        _vector_db_instance = NBAVectorDB(csv_path)
    return _vector_db_instance


# ============================================================================
# TOOLS
# ============================================================================

_tool_trace_ctx: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "nba_tool_trace", default=None
)
_tool_budget_ctx: ContextVar[Optional[Dict[str, int]]] = ContextVar(
    "nba_tool_budget", default=None
)

# In-memory response cache: key -> {expires_at, payload}
_ask_cache: Dict[str, Dict[str, Any]] = {}
_ask_cache_lock = threading.Lock()
_ask_cache_stats = {"hits": 0, "misses": 0}


def begin_tool_trace(max_tool_calls: int = ASK_MAX_TOOL_CALLS) -> List[Dict[str, Any]]:
    """Start collecting tool calls for the current request and reset the tool budget."""
    trace: List[Dict[str, Any]] = []
    _tool_trace_ctx.set(trace)
    _tool_budget_ctx.set({"count": 0, "max": max(1, int(max_tool_calls))})
    return trace


def get_tool_trace() -> List[Dict[str, Any]]:
    """Return the active tool trace (empty list if none)."""
    return list(_tool_trace_ctx.get() or [])


def clear_ask_cache() -> Dict[str, int]:
    """Clear the in-memory ask cache. Returns previous size + hit/miss counters."""
    with _ask_cache_lock:
        size = len(_ask_cache)
        _ask_cache.clear()
        stats = dict(_ask_cache_stats)
        _ask_cache_stats["hits"] = 0
        _ask_cache_stats["misses"] = 0
    return {"cleared": size, **stats}


def get_ask_cache_stats() -> Dict[str, Any]:
    with _ask_cache_lock:
        return {
            "size": len(_ask_cache),
            "hits": _ask_cache_stats["hits"],
            "misses": _ask_cache_stats["misses"],
            "ttl_s": ASK_CACHE_TTL_S,
        }


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().casefold())


def _cache_key(question: str, data_path: str) -> str:
    raw = f"{_normalize_question(question)}::{os.path.abspath(data_path)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _citation_from_record(record: Dict[str, Any], source_tool: str) -> Dict[str, Any]:
    keys = ("Player", "Tm", "Opp", "PTS", "AST", "TRB", "FG%", "3P", "Data", "Res", "MP")
    snippet = {k: record[k] for k in keys if k in record and record[k] is not None}
    return {"source": "tool", "tool": source_tool, "snippet": snippet or record}


def _record_tool_call(payload: Dict[str, Any]) -> None:
    trace = _tool_trace_ctx.get()
    if trace is None:
        return

    tool_name = payload.get("tool", "unknown")
    citations: List[Dict[str, Any]] = []
    for key in ("records", "sample", "results"):
        items = payload.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items[:5]:
            if isinstance(item, dict) and "record" in item and isinstance(item["record"], dict):
                citations.append(_citation_from_record(item["record"], tool_name))
            elif isinstance(item, dict):
                citations.append(_citation_from_record(item, tool_name))

    # Scalar / series results still need citation snippets for groundedness.
    if isinstance(payload.get("data"), dict) and payload["data"]:
        citations.append({"source": "tool", "tool": tool_name, "snippet": payload["data"]})
    if "value" in payload and payload.get("ok", True):
        citations.append({
            "source": "tool",
            "tool": tool_name,
            "snippet": {"value": payload.get("value"), "result_type": payload.get("result_type")},
        })

    summary_bits = []
    if "match_count" in payload:
        summary_bits.append(f"match_count={payload['match_count']}")
    if "returned" in payload:
        summary_bits.append(f"returned={payload['returned']}")
    if "total_rows" in payload:
        summary_bits.append(f"total_rows={payload['total_rows']}")
    if "result_type" in payload:
        summary_bits.append(f"result_type={payload['result_type']}")
    if "error" in payload:
        summary_bits.append(f"error={payload['error']}")

    # Compact evidence blob used by the groundedness gate.
    evidence_parts = [json.dumps(payload, ensure_ascii=False, default=str)[:4000]]

    trace.append({
        "tool": tool_name,
        "ok": bool(payload.get("ok", True)),
        "summary": ", ".join(summary_bits) if summary_bits else ("ok" if payload.get("ok", True) else "error"),
        "citations": citations,
        "evidence": "\n".join(evidence_parts),
    })


def _normalize_text(value: Any) -> str:
    """Lowercase and strip punctuation so DeAaron ~= De'Aaron."""
    text = str(value or "").casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _series_contains_fuzzy(series: pd.Series, needle: str) -> pd.Series:
    """Case-insensitive contains; also matches ignoring punctuation/apostrophes."""
    needle = str(needle or "")
    if not needle:
        return pd.Series(False, index=series.index)
    as_str = series.astype(str)
    direct = as_str.str.contains(needle, case=False, na=False, regex=False)
    norm_needle = _normalize_text(needle)
    if not norm_needle:
        return direct
    # Vectorized-ish via map for fuzzy punctuation-insensitive match.
    fuzzy = as_str.map(lambda x: norm_needle in _normalize_text(x))
    return direct | fuzzy


def _records_from_df(df: pd.DataFrame, limit: int) -> List[Dict]:
    """Convert a DataFrame slice to JSON-safe dict records."""
    return json.loads(df.head(limit).to_json(orient="records", date_format="iso"))


def _tool_budget_exceeded() -> Optional[str]:
    budget = _tool_budget_ctx.get()
    if budget is None:
        return None
    if budget["count"] >= budget["max"]:
        return f"Tool budget exceeded (max {budget['max']} calls per question)."
    budget["count"] += 1
    return None


def _tool_json(payload: Dict) -> str:
    """Serialize tool responses as compact JSON for the LLM and record the call."""
    _record_tool_call(payload)
    return json.dumps(payload, ensure_ascii=False, default=str)


def _extract_stat_numbers(text: str) -> List[str]:
    """Extract likely stat numbers; skip 4-digit years."""
    nums = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text or "")
    out = []
    for n in nums:
        if re.fullmatch(r"20\d{2}", n):  # years like 2024/2025
            continue
        out.append(n)
    return out


def assess_groundedness(
    answer: str,
    tool_trace: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Check whether numeric claims in the answer appear in tool evidence/citations.

    Returns score in [0, 1], lists of grounded/ungrounded numbers, and whether
    the answer should be refused.
    """
    evidence_parts = [c.get("evidence") or c.get("summary") or "" for c in tool_trace]
    for cite in citations:
        evidence_parts.append(json.dumps(cite.get("snippet") or cite, default=str))
    evidence = "\n".join(evidence_parts)

    answer_nums = _extract_stat_numbers(answer)
    if not answer_nums:
        # No numeric claims — treat as grounded if at least one tool succeeded.
        any_ok = any(bool(t.get("ok")) for t in tool_trace)
        return {
            "score": 1.0 if any_ok else 0.0,
            "grounded_numbers": [],
            "ungrounded_numbers": [],
            "refuse": not any_ok,
            "reason": None if any_ok else "No successful tool calls to ground the answer.",
        }

    grounded, ungrounded = [], []
    for n in answer_nums:
        # Allow "24" to match "24.46" in evidence.
        pattern = rf"(?<![\d.]){re.escape(n)}(?:\.\d+)?(?![\d])"
        if re.search(pattern, evidence):
            grounded.append(n)
        else:
            ungrounded.append(n)

    score = len(grounded) / len(answer_nums) if answer_nums else 1.0
    # Refuse only when grounding is weak overall (configurable threshold).
    refuse = score < ASK_GROUNDEDNESS_MIN
    reason = None
    if refuse:
        reason = (
            "Answer contains numeric claims that were not found in tool results: "
            + ", ".join(ungrounded[:8] or ["(none matched)"])
        )
    return {
        "score": score,
        "grounded_numbers": grounded,
        "ungrounded_numbers": ungrounded,
        "refuse": refuse,
        "reason": reason,
    }


def get_agent_tools(data_path: str):
    """
    Get the list of tools available for agents.

    Every tool returns structured JSON (facts/stats), never raw CSV text dumps.
    """

    def _read_nba_data(limit: int = 10) -> str:
        budget_err = _tool_budget_exceeded()
        if budget_err:
            return _tool_json({"ok": False, "tool": "read_nba_data", "error": budget_err})
        try:
            df = pd.read_csv(data_path)
            limit = min(max(limit, 1), 50)
            return _tool_json({
                "ok": True,
                "tool": "read_nba_data",
                "total_rows": int(len(df)),
                "columns": df.columns.tolist(),
                "sample_limit": limit,
                "sample": _records_from_df(df, limit),
            })
        except Exception as e:
            return _tool_json({"ok": False, "tool": "read_nba_data", "error": str(e)})

    def _search_nba_data(
        query: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[str] = None,
        limit: int = 50,
    ) -> str:
        budget_err = _tool_budget_exceeded()
        if budget_err:
            return _tool_json({"ok": False, "tool": "search_nba_data", "error": budget_err})
        try:
            query = (query or "").strip()
            column = (column or "").strip()
            value = "" if value is None else str(value).strip()
            df = pd.read_csv(data_path)
            matched = len(df)

            # Targeted column filter (fuzzy for Player names / substring elsewhere).
            if column and value:
                if column not in df.columns:
                    return _tool_json({
                        "ok": False,
                        "tool": "search_nba_data",
                        "error": f"Column '{column}' not found",
                        "available_columns": df.columns.tolist(),
                    })
                if column == "Player":
                    df = df[_series_contains_fuzzy(df[column], value)]
                else:
                    df = df[df[column].astype(str).str.contains(str(value), case=False, na=False, regex=False)]

            # Free-text query: search ALL columns as strings (including numeric PTS/AST/...).
            if query:
                mask = pd.Series(False, index=df.index)
                for col in df.columns:
                    if col == "Player":
                        mask |= _series_contains_fuzzy(df[col], query)
                    else:
                        mask |= df[col].astype(str).str.contains(query, case=False, na=False, regex=False)
                # Also match punctuation-stripped query against Player (DeAaron -> De'Aaron).
                if "Player" in df.columns:
                    mask |= _series_contains_fuzzy(df["Player"], query)
                df = df[mask]

            matched = len(df)
            limit = min(max(limit, 1), 50)
            hint = None
            if matched == 0:
                hint = (
                    "No rows matched. Try lookup_player_games(player=..., pts=...) "
                    "or analyze_nba_data with df['Player'].str.contains(...) & (df['PTS']==N)."
                )
            return _tool_json({
                "ok": True,
                "tool": "search_nba_data",
                "filters": {"query": query, "column": column, "value": value},
                "match_count": int(matched),
                "returned": int(min(matched, limit)),
                "records": _records_from_df(df, limit),
                "hint": hint,
            })
        except Exception as e:
            return _tool_json({"ok": False, "tool": "search_nba_data", "error": str(e)})

    def _lookup_player_games(
        player: str,
        pts: Optional[float] = None,
        min_pts: Optional[float] = None,
        limit: int = 20,
    ) -> str:
        """Reliable player/game lookup with fuzzy name match and optional PTS filters."""
        budget_err = _tool_budget_exceeded()
        if budget_err:
            return _tool_json({"ok": False, "tool": "lookup_player_games", "error": budget_err})
        try:
            if not player or not str(player).strip():
                return _tool_json({
                    "ok": False,
                    "tool": "lookup_player_games",
                    "error": "player is required (e.g. 'Giannis' or \"De'Aaron Fox\")",
                })
            df = pd.read_csv(data_path)
            if "Player" not in df.columns:
                return _tool_json({"ok": False, "tool": "lookup_player_games", "error": "Player column missing"})

            # Treat negative sentinels as "no filter" (agents sometimes pass -1).
            if pts is not None and float(pts) < 0:
                pts = None
            if min_pts is not None and float(min_pts) < 0:
                min_pts = None

            out = df[_series_contains_fuzzy(df["Player"], player)]
            if pts is not None and "PTS" in out.columns:
                out = out[out["PTS"] == float(pts)]
            if min_pts is not None and "PTS" in out.columns:
                out = out[out["PTS"] >= float(min_pts)]

            # Prefer highest-scoring games first for inspection.
            if "PTS" in out.columns:
                out = out.sort_values("PTS", ascending=False)

            matched = len(out)
            limit = min(max(int(limit or 20), 1), 50)
            cols = [c for c in ("Player", "Tm", "Opp", "PTS", "AST", "TRB", "3P", "Data", "Res", "MP") if c in out.columns]
            return _tool_json({
                "ok": True,
                "tool": "lookup_player_games",
                "filters": {"player": player, "pts": pts, "min_pts": min_pts},
                "match_count": int(matched),
                "returned": int(min(matched, limit)),
                "records": _records_from_df(out[cols] if cols else out, limit),
            })
        except Exception as e:
            return _tool_json({"ok": False, "tool": "lookup_player_games", "error": str(e)})

    def _get_nba_data_summary() -> str:
        budget_err = _tool_budget_exceeded()
        if budget_err:
            return _tool_json({"ok": False, "tool": "get_nba_data_summary", "error": budget_err})
        try:
            df = pd.read_csv(data_path)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            return _tool_json({
                "ok": True,
                "tool": "get_nba_data_summary",
                "total_rows": int(len(df)),
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "unique_players": int(df["Player"].nunique()) if "Player" in df.columns else None,
                "unique_teams": int(df["Tm"].nunique()) if "Tm" in df.columns else None,
                "date_range": {
                    "min": str(df["Data"].min()) if "Data" in df.columns else None,
                    "max": str(df["Data"].max()) if "Data" in df.columns else None,
                },
                "missing_values": {col: int(v) for col, v in df.isnull().sum().items() if v > 0},
                "sample": _records_from_df(df, 3),
            })
        except Exception as e:
            return _tool_json({"ok": False, "tool": "get_nba_data_summary", "error": str(e)})

    @tool("read_nba_data")
    def read_nba_data(limit: int = 10) -> str:
        """
        Return a JSON sample of NBA rows plus schema metadata.

        Args:
            limit: Number of sample rows (default 10, max 50)
        """
        return _read_nba_data(min(limit, 50))

    @tool("search_nba_data")
    def search_nba_data(
        query: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> str:
        """
        Search/filter NBA data and return matching rows as JSON records.
        Searches numeric columns too (so query='59' can match PTS).
        Player matching ignores punctuation (DeAaron ~= De'Aaron).
        You may pass only query — column/value/limit are optional.

        Args:
            query: Optional text to search across all columns (e.g. 'LeBron James')
            column: Optional column name to filter (e.g. 'Player', 'PTS')
            value: Optional value to match in that column
            limit: Max rows to return (default 50)
        """
        return _search_nba_data(query, column, value, int(limit or 50))

    # CrewAI's @tool marks every arg required unless we override args_schema.
    search_nba_data.args_schema = SearchNbaDataInput

    @tool("lookup_player_games")
    def lookup_player_games(
        player: str,
        pts: Optional[float] = None,
        min_pts: Optional[float] = None,
        limit: Optional[int] = 20,
    ) -> str:
        """
        Look up games for a player with fuzzy name match.
        Only `player` is required. pts/min_pts are optional filters.

        Args:
            player: Player name or fragment (e.g. 'Giannis', 'Fox', 'LeBron')
            pts: Optional exact PTS filter (e.g. 59 or 60). Omit if not filtering.
            min_pts: Optional minimum PTS filter. Omit if not filtering.
            limit: Max rows to return (default 20)
        """
        return _lookup_player_games(player, pts=pts, min_pts=min_pts, limit=int(limit or 20))

    lookup_player_games.args_schema = LookupPlayerGamesInput

    @tool("get_nba_data_summary")
    def get_nba_data_summary() -> str:
        """Return a JSON summary of dataset shape, coverage, and missing values."""
        return _get_nba_data_summary()

    def _semantic_search_nba_data(query: str, n_results: int = 10) -> str:
        budget_err = _tool_budget_exceeded()
        if budget_err:
            return _tool_json({"ok": False, "tool": "semantic_search_nba_data", "error": budget_err})
        try:
            n_results = min(max(n_results, 1), 50)
            vector_db = get_vector_db(data_path)
            results = vector_db.search(query, n_results=n_results)

            if not results:
                return _tool_json({
                    "ok": True,
                    "tool": "semantic_search_nba_data",
                    "query": query,
                    "match_count": 0,
                    "results": [],
                })

            df = pd.read_csv(data_path)
            payload_results = []
            for result in results:
                metadata = result["metadata"]
                row_index = metadata.get("row_index", -1)
                record = None
                if 0 <= row_index < len(df):
                    record = json.loads(
                        df.iloc[[row_index]].to_json(orient="records", date_format="iso")
                    )[0]
                payload_results.append({
                    "similarity": float(result["similarity"]),
                    "row_index": row_index,
                    "document": result["document"],
                    "record": record,
                })

            return _tool_json({
                "ok": True,
                "tool": "semantic_search_nba_data",
                "query": query,
                "match_count": len(payload_results),
                "results": payload_results,
            })
        except Exception as e:
            return _tool_json({"ok": False, "tool": "semantic_search_nba_data", "error": str(e)})

    @tool("semantic_search_nba_data")
    def semantic_search_nba_data(query: str, n_results: int = 10) -> str:
        """
        Semantic search over NBA rows; returns JSON with similarity + full records.

        Args:
            query: Natural language query
            n_results: Number of results (default 10, max 50)
        """
        return _semantic_search_nba_data(query, n_results)

    def _analyze_nba_data(pandas_code: str) -> str:
        budget_err = _tool_budget_exceeded()
        if budget_err:
            return _tool_json({"ok": False, "tool": "analyze_nba_data", "error": budget_err})
        try:
            df = pd.read_csv(data_path)
            namespace = {"pd": pd, "df": df, "__builtins__": __builtins__}
            exec(f"result = {pandas_code}", namespace)
            result = namespace.get("result")

            if isinstance(result, pd.DataFrame):
                truncated = len(result) > 50
                return _tool_json({
                    "ok": True,
                    "tool": "analyze_nba_data",
                    "result_type": "dataframe",
                    "shape": [int(result.shape[0]), int(result.shape[1])],
                    "truncated": truncated,
                    "records": _records_from_df(result, 50),
                })
            if isinstance(result, pd.Series):
                truncated = len(result) > 50
                series = result.head(50)
                return _tool_json({
                    "ok": True,
                    "tool": "analyze_nba_data",
                    "result_type": "series",
                    "length": int(len(result)),
                    "truncated": truncated,
                    "data": json.loads(series.to_json(date_format="iso")),
                })

            value = result
            if isinstance(result, (pd.Timestamp,)):
                value = str(result)
            elif hasattr(result, "item"):
                try:
                    value = result.item()
                except Exception:
                    value = str(result)

            return _tool_json({
                "ok": True,
                "tool": "analyze_nba_data",
                "result_type": type(result).__name__,
                "value": value,
            })
        except Exception as e:
            return _tool_json({
                "ok": False,
                "tool": "analyze_nba_data",
                "error": str(e),
                "hint": "Use 'df' as the DataFrame and write an expression that assigns to result.",
            })

    @tool("analyze_nba_data")
    def analyze_nba_data(pandas_code: str) -> str:
        """
        Run a pandas expression on df and return the result as JSON.

        Args:
            pandas_code: Expression using DataFrame variable 'df' (e.g. df.groupby('Tm')['PTS'].mean())
        """
        return _analyze_nba_data(pandas_code)

    # semantic_search_nba_data is implemented but currently omitted from the default
    # toolset: conda envs often break on sentence-transformers/transformers imports.
    # Re-add after: pip install -U "transformers>=4.41,<5" sentence-transformers
    return [
        read_nba_data,
        search_nba_data,
        lookup_player_games,
        get_nba_data_summary,
        analyze_nba_data,
    ]


# ============================================================================
# AGENTS
# ============================================================================

# Get LLM instance (shared across all agents)
_llm_instance = None

def _get_llm_instance():
    """Get or create shared LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm()
    return _llm_instance


def create_engineer_agent(csv_path: str = None) -> Agent:
    """
    Create the Engineer Agent for data processing and engineering tasks.
    
    Args:
        csv_path: Path to CSV file (defaults to NBA_DATA_PATH)
    
    Returns:
        Agent: Configured Engineer Agent
    """
    data_path = csv_path or NBA_DATA_PATH
    agent_tools = get_agent_tools(data_path)
    
    return Agent(
        role="Data Engineer",
        goal="Process, clean, and prepare data for analysis. Ensure data quality and create structured datasets.",
        backstory="""You are an expert data engineer with years of experience in sports analytics. 
        You specialize in processing large datasets, handling missing values, data validation, 
        and creating clean, analysis-ready datasets. You understand statistics deeply and 
        know how to structure data for optimal analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=_get_llm_instance(),
        tools=agent_tools,
    )


def create_analyst_agent(csv_path: str = None) -> Agent:
    """
    Create the Analyst Agent for data analysis and insights.
    
    Args:
        csv_path: Path to CSV file (defaults to NBA_DATA_PATH)
    
    Returns:
        Agent: Configured Analyst Agent
    """
    data_path = csv_path or NBA_DATA_PATH
    agent_tools = get_agent_tools(data_path)
    
    return Agent(
        role="Data Analyst",
        goal="Analyze data to extract meaningful insights, identify patterns, and provide actionable recommendations.",
        backstory="""You are a seasoned data analyst with a passion for analytics. 
        You excel at finding patterns in data, identifying trends, performing statistical analysis, 
        and translating complex data into clear, actionable insights. You understand performance 
        metrics and can provide strategic recommendations based on data.
        
        CRITICAL: When asked for aggregations, top N lists, totals, or statistical summaries:
        - ALWAYS use the 'analyze_nba_data' tool with pandas groupby operations
        - NEVER invent numbers that are not present in tool JSON
        - Prefer 1-3 tool calls; stop once you have enough evidence
        - For player+points lookups, prefer lookup_player_games(player=..., pts=...)
        - If a search returns 0 rows, retry with a different tool before concluding not found
        - For "top 5 three-point shooters": use analyze_nba_data with groupby('Player')['3P'].sum()
        - Plan your analysis: understand what aggregation is needed, then write the appropriate pandas code""",
        verbose=True,
        allow_delegation=False,
        max_iter=8,
        max_retry_limit=1,
        llm=_get_llm_instance(),
        tools=agent_tools,
    )


def create_storyteller_agent() -> Agent:
    """
    Create the Storyteller Agent for creating engaging headlines and storylines.
    
    Returns:
        Agent: Configured Storyteller Agent
    """
    return Agent(
        role="Sports Storyteller",
        goal="Transform data analysis results into engaging headlines and compelling storylines that bring statistics to life with narrative and context.",
        backstory="""You are a creative sports journalist and storyteller with a talent for turning 
        statistical analysis into captivating headlines and engaging storylines. You know how to make data come alive, 
        creating headlines that grab attention and writing compelling content that tells the story behind the numbers. 
        You understand what makes a great sports story and can transform complex analytics into memorable narratives 
        that connect with readers. You write with flair, using vivid language and storytelling techniques to make 
        statistics human and relatable. Your stories provide context, explain why the data matters, and bring the 
        performance metrics to life with engaging prose.""",
        verbose=True,
        allow_delegation=False,
        llm=_get_llm_instance(),
        tools=[],  # Storyteller doesn't need data tools
    )


# ============================================================================
# TASKS
# ============================================================================

def create_data_engineering_task(engineer_agent, csv_path: str = None) -> Task:
    """
    Create the data engineering task for processing and cleaning data.
    
    Args:
        engineer_agent: The Engineer Agent to assign this task to
        csv_path: Path to CSV file (defaults to NBA_DATA_PATH)
    
    Returns:
        Task: Configured data engineering task
    """
    data_path = csv_path or NBA_DATA_PATH
    
    return Task(
        description=f"""
        Quickly examine the dataset located at {data_path}. 
        
        Your tasks (BE EFFICIENT - use tools only once):
        1. Get a brief summary of the dataset structure (use get_nba_data_summary ONCE)
        2. Note the key columns available
        3. Verify the data is ready for analysis
        
        IMPORTANT: 
        - Use get_nba_data_summary ONCE only - it provides all needed info
        - Do NOT call read_nba_data or analyze_nba_data multiple times
        - Keep your report concise (2-3 sentences)
        - The data is already clean and ready for analysis
        
        Provide a brief confirmation that the dataset is loaded and ready for analysis.
        """,
        agent=engineer_agent,
        expected_output="A brief confirmation (2-3 sentences) that the dataset is loaded and ready for analysis, including key column names."
    )


def create_data_analysis_task(analyst_agent, data_engineering_task: Task) -> Task:
    """
    Create the data analysis task for extracting insights from NBA data.
    
    Args:
        analyst_agent: The Analyst Agent to assign this task to
        data_engineering_task: The data engineering task for context
    
    Returns:
        Task: Configured data analysis task
    """
    return Task(
        description=f"""
        Using the cleaned NBA 2024-25 dataset, perform comprehensive analysis:
        
        Your tasks:
        1. Analyze player performance metrics:
           - Top performers by points, assists, rebounds
           - Shooting efficiency analysis (FG%, 3P%, FT%)
           - Player efficiency ratings
        2. Team performance analysis:
           - Win/loss records by team
           - Team offensive and defensive statistics
           - Team performance trends
        3. Game insights:
           - High-scoring games
           - Close games vs blowouts
           - Performance by date/period
        4. Identify key patterns and trends:
           - Best performing players
           - Most efficient teams
           - Statistical outliers
        5. Provide actionable insights and recommendations
        
        Create a comprehensive analysis report with key findings and insights.
        """,
        agent=analyst_agent,
        expected_output="A detailed analysis report with key insights, statistical findings, top performers, team analysis, and actionable recommendations based on the NBA 2024-25 data.",
        context=[data_engineering_task]
    )


def create_custom_analysis_task(analyst_agent, user_query: str, data_engineering_task: Task = None, csv_path: str = None) -> Task:
    """
    Create a custom data analysis task based on user input.
    
    Args:
        analyst_agent: The Analyst Agent to assign this task to
        user_query: The user's custom analysis query/task
        data_engineering_task: The data engineering task for context (optional for parallel execution)
        csv_path: Path to CSV file (for reference in description)
    
    Returns:
        Task: Configured custom analysis task
    """
    data_path = csv_path or NBA_DATA_PATH
    context = [data_engineering_task] if data_engineering_task else []
    
    return Task(
        description=f"""
        Using the dataset located at {data_path}, answer this user question:

        {user_query}

        CRITICAL RULES:
        1. You MUST call tools to get facts before answering. Do not invent stats.
        2. For averages / totals / top-N, use analyze_nba_data with pandas on `df` (one call).
           Example avg: df[df['Player'].str.contains('LeBron', case=False, na=False)]['PTS'].mean()
        3. For "Did player X score Y?" / "opponent when X scored Y", use
           lookup_player_games with player='X' and pts=Y.
        4. search_nba_data: you may pass only query (column/value optional). Prefer analyze for averages.
        5. Use get_nba_data_summary for unique players/teams / dataset scale.
        6. Prefer 1-2 tool calls. Stop once you have the answer.
        7. If a tool errors or returns 0 matches, retry once with analyze_nba_data — then stop.
        8. Every number in your answer must come from tool JSON output.
        9. Include opponent (Opp), player, and the key stat when available.

        RESPONSE FORMAT:
        - Lead with a direct answer.
        - Include concrete numbers from tool results.
        - End with a short "Sources:" section listing tool name(s) and key facts (player/team/stat/date).
        """,
        agent=analyst_agent,
        expected_output="A grounded answer with concrete statistics from tool results, plus a short Sources section.",
        context=context
    )


def create_storyteller_task(storyteller_agent, analysis_task: Task) -> Task:
    """
    Create a storyteller task that creates headlines and storylines from the analysis results.
    
    Args:
        storyteller_agent: The Storyteller Agent to assign this task to
        analysis_task: The analysis task whose output will be used to create headlines and content
    
    Returns:
        Task: Configured storyteller task
    """
    return Task(
        description="""
        Review the data analysis results and create engaging headlines and compelling storylines that bring the data to life.
        
        Your tasks:
        1. Read and understand the analysis results thoroughly
        2. Identify the most important and interesting findings
        3. Create 3-5 compelling headlines that:
           - Are catchy and attention-grabbing
           - Accurately reflect the key insights
           - Use engaging sports journalism language
           - Are suitable for display to users
        
        4. Write engaging storylines/content for each headline that:
           - Tells a story about the findings
           - Provides context and narrative around the statistics
           - Makes the data come alive with compelling prose
           - Explains why these insights matter
           - Uses vivid language and storytelling techniques
           - Is 2-3 paragraphs per storyline (enough to be engaging but concise)
        
        5. Format your output as follows:
           HEADLINES:
           [List of 3-5 headlines, one per line]
           
           STORYLINES:
           [For each headline, write 2-3 paragraphs of engaging content that tells the story behind the data]
        
        Make both the headlines and storylines exciting, memorable, and true to the data insights. 
        Write like a sports journalist who knows how to make statistics compelling and human.
        """,
        agent=storyteller_agent,
        expected_output="A formatted output with 3-5 engaging headlines followed by detailed storylines (2-3 paragraphs each) that bring the data analysis to life with compelling narrative and context.",
        context=[analysis_task]
    )


# ============================================================================
# CREW CREATION
# ============================================================================

def create_crew() -> Crew:
    """
    Create and configure the CrewAI crew with agents and tasks.
    
    Returns:
        Crew: Configured CrewAI crew ready for execution
    """
    engineer_agent = create_engineer_agent()
    analyst_agent = create_analyst_agent()
    
    data_engineering_task = create_data_engineering_task(engineer_agent)
    data_analysis_task = create_data_analysis_task(analyst_agent, data_engineering_task)
    
    return Crew(
        agents=[engineer_agent, analyst_agent],
        tasks=[data_engineering_task, data_analysis_task],
        process=Process.sequential,
        verbose=True,
    )


def create_crew_with_custom_task(user_query: str, csv_path: str = None) -> Crew:
    """
    Create a CrewAI crew with engineering task, custom analyst task, and storyteller task.
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Optional path to CSV file (if None, uses default from config)
    
    Returns:
        Crew: Configured CrewAI crew ready for execution
    """
    engineer_agent = create_engineer_agent(csv_path)
    analyst_agent = create_analyst_agent(csv_path)
    storyteller_agent = create_storyteller_agent()
    
    data_engineering_task = create_data_engineering_task(engineer_agent, csv_path)
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    storyteller_task = create_storyteller_task(storyteller_agent, custom_analysis_task)
    
    return Crew(
        agents=[engineer_agent, analyst_agent, storyteller_agent],
        tasks=[data_engineering_task, custom_analysis_task, storyteller_task],
        process=Process.sequential,
        verbose=True,
    )


def create_flow_crew(user_query: str, csv_path: str) -> Crew:
    """
    Create a single crew with parallel tasks (Engineer and Analyst) that merge results at the end.
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Path to the uploaded CSV file
    
    Returns:
        Crew: Single crew with parallel tasks that will merge results
    """
    engineer_agent = create_engineer_agent(csv_path)
    analyst_agent = create_analyst_agent(csv_path)
    storyteller_agent = create_storyteller_agent()
    
    data_engineering_task = create_data_engineering_task(engineer_agent, csv_path)
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    storyteller_task = create_storyteller_task(storyteller_agent, custom_analysis_task)
    
    return Crew(
        agents=[engineer_agent, analyst_agent, storyteller_agent],
        tasks=[data_engineering_task, custom_analysis_task, storyteller_task],
        process=Process.sequential,
        verbose=True,
    )


def create_analysis_only_crew(user_query: str, csv_path: str) -> Crew:
    """
    Create a crew with only Analyst and Storyteller agents (no Engineer).
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Path to the uploaded CSV file
    
    Returns:
        Crew: Crew with only analyst and storyteller tasks
    """
    analyst_agent = create_analyst_agent(csv_path)
    storyteller_agent = create_storyteller_agent()
    
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    storyteller_task = create_storyteller_task(storyteller_agent, custom_analysis_task)
    
    return Crew(
        agents=[analyst_agent, storyteller_agent],
        tasks=[custom_analysis_task, storyteller_task],
        process=Process.sequential,
        verbose=True,
    )


def create_analyst_only_crew(user_query: str, csv_path: str) -> Crew:
    """
    Create a crew with only Analyst agent (no Engineer, no Storyteller).
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Path to the uploaded CSV file
    
    Returns:
        Crew: Crew with only analyst task
    """
    analyst_agent = create_analyst_agent(csv_path)
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    
    return Crew(
        agents=[analyst_agent],
        tasks=[custom_analysis_task],
        process=Process.sequential,
        verbose=True,
    )


def run_ask(
    question: str,
    csv_path: Optional[str] = None,
    use_cache: bool = True,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Answer a natural-language NBA question via the analyst crew.

    Features:
    - In-memory TTL cache (normalized question + data path)
    - Tool-call budget per request
    - Crew kickoff retries with exponential backoff
    - Groundedness gate (refuse if numeric claims are ungrounded)
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    data_path = csv_path or NBA_DATA_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"NBA data file not found: {data_path}")

    global OPENAI_API_KEY
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    q = question.strip()
    key = _cache_key(q, data_path)
    retries = ASK_MAX_RETRIES if max_retries is None else max(0, int(max_retries))

    if use_cache:
        with _ask_cache_lock:
            hit = _ask_cache.get(key)
            if hit and hit.get("expires_at", 0) > time.time():
                _ask_cache_stats["hits"] += 1
                cached = dict(hit["payload"])
                cached["cached"] = True
                cached["latency_ms"] = 0
                return cached
            _ask_cache_stats["misses"] += 1

    last_error: Optional[Exception] = None
    answer = ""
    trace: List[Dict[str, Any]] = []
    attempts_used = 0
    started = time.perf_counter()

    for attempt in range(retries + 1):
        attempts_used = attempt + 1
        trace = begin_tool_trace(ASK_MAX_TOOL_CALLS)
        try:
            crew = create_analyst_only_crew(q, data_path)
            result = crew.kickoff()
            answer = str(result)
            if hasattr(result, "raw") and result.raw:
                answer = str(result.raw)
            last_error = None
            break
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.4 * (2 ** attempt))
                continue

    latency_ms = int((time.perf_counter() - started) * 1000)

    if last_error is not None:
        raise RuntimeError(f"ask failed after {attempts_used} attempt(s): {last_error}") from last_error

    citations: List[Dict[str, Any]] = []
    seen = set()
    for entry in trace:
        for cite in entry.get("citations") or []:
            key_c = json.dumps(cite.get("snippet") or cite, sort_keys=True, default=str)
            if key_c in seen:
                continue
            seen.add(key_c)
            citations.append(cite)
            if len(citations) >= 10:
                break
        if len(citations) >= 10:
            break

    grounding = assess_groundedness(answer, trace, citations)
    refused = bool(grounding.get("refuse"))
    if refused:
        reason = grounding.get("reason") or "Insufficient grounded evidence from tools."
        answer = (
            "I cannot confidently answer from the available tool results. "
            f"{reason} "
            "Please rephrase or ask a more specific factual question."
        )

    tool_trace = [
        {"tool": e["tool"], "ok": e["ok"], "summary": e["summary"]}
        for e in trace
    ]

    payload = {
        "question": q,
        "answer": answer,
        "citations": citations,
        "tool_trace": tool_trace,
        "latency_ms": latency_ms,
        "model": os.getenv("OPENAI_MODEL") or OPENAI_MODEL,
        "data_path": data_path,
        "cached": False,
        "groundedness": grounding.get("score"),
        "refused": refused,
        "retries_used": max(0, attempts_used - 1),
    }

    # Only cache successful grounded answers.
    if use_cache and not refused:
        with _ask_cache_lock:
            _ask_cache[key] = {
                "expires_at": time.time() + ASK_CACHE_TTL_S,
                "payload": {**payload, "cached": True},
            }

    return payload


# ============================================================================
# APP FUNCTIONS
# ============================================================================

def process_file_and_analyze(file, user_query: str = "", engineer_result: str = None) -> Tuple[str, str]:
    """
    Process uploaded file and run all agents (Engineer, Analyst, Storyteller), then merge results.
    
    Args:
        file: Uploaded file object
        user_query: The user's analysis query/task (empty for general analysis)
        engineer_result: Previously computed engineer result (if available)
    
    Returns:
        tuple: (merged_results, engineer_result) - engineer_result is stored for reuse
    """
    if file is None:
        return "Please upload a CSV file.", engineer_result or ""
    
    if not user_query or not user_query.strip():
        user_query = "Provide a comprehensive analysis of the dataset including: top performers, key statistics, interesting patterns, and notable insights."
    
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        csv_path = file_path
        
        crew = create_flow_crew(user_query.strip(), csv_path)
        result = crew.kickoff()
        
        merged_output = []
        stored_engineer_result = ""
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 1:
                engineer_output = str(result.tasks_output[0])
                stored_engineer_result = engineer_output
                merged_output.append("## Engineer Agent Results")
                merged_output.append("")
                merged_output.append(engineer_output)
                merged_output.append("")
                merged_output.append("---")
                merged_output.append("")
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 2:
                analyst_output = str(result.tasks_output[1])
                merged_output.append("## Analyst Agent Results")
                merged_output.append("")
                merged_output.append(analyst_output)
                merged_output.append("")
                merged_output.append("---")
                merged_output.append("")
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 3:
                storyteller_output = str(result.tasks_output[2])
                merged_output.append("## Storyteller Agent Results")
                merged_output.append("")
                merged_output.append(storyteller_output)
                merged_output.append("")
        
        if not merged_output:
            merged_output.append("## Complete Analysis Results")
            merged_output.append("")
            merged_output.append(str(result))
        
        return "\n".join(merged_output), stored_engineer_result
    
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        print(error_msg)
        return error_msg, engineer_result or ""


def process_question_only(file, user_query: str) -> str:
    """
    Process a specific user question using only the Analyst agent.
    
    Args:
        file: Uploaded file object
        user_query: The user's specific analysis question
    
    Returns:
        str: Analyst results only
    """
    if file is None:
        return "Please upload a CSV file."
    
    if not user_query or not user_query.strip():
        return "Please enter a question."
    
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        csv_path = file_path
        
        crew = create_analyst_only_crew(user_query.strip(), csv_path)
        result = crew.kickoff()
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 1:
                analyst_output = str(result.tasks_output[0])
                return analyst_output
        
        return str(result)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        print(error_msg)
        return error_msg


def create_app():
    """Create and return the Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="NBA Stats Analysis with CrewAI", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # NBA Stats Analysis with CrewAI
        
        Upload your NBA statistics CSV file to get comprehensive analysis with engaging storylines.
        
        **How it works:**
        - **Engineer Agent**: Examines and validates your dataset
        - **Analyst Agent**: Performs deep analysis (general or based on your question)
        - **Storyteller Agent**: Creates headlines and compelling storylines
        
        All agents work in parallel and results are merged for you!
        """)
        
        engineer_state = gr.State(value="")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                analyze_btn = gr.Button(
                    "Analyze Dataset", 
                    variant="primary", 
                    size="lg",
                    visible=False
                )
                
                gr.Markdown("### Ask a Specific Question")
                
                query_input = gr.Textbox(
                    label="Your Analysis Question",
                    placeholder="e.g., 'Who are the top 5 three-point shooters?' or 'Analyze the best players by assists'",
                    lines=2
                )
                
                question_output = gr.Markdown(
                    value="",
                    label="Answer",
                    visible=False
                )
                
                query_btn = gr.Button(
                    "Analyze with Question", 
                    variant="secondary", 
                    size="lg"
                )
        
        with gr.Row():
            with gr.Column():
                status_output = gr.Markdown(
                    value="",
                    label="Agent Status",
                    visible=False
                )
        
        with gr.Row():
            with gr.Column():
                merged_output = gr.Markdown(
                    value="**Ready to analyze!** Upload a CSV file above, then click 'Analyze Dataset' to get started.",
                    label="Full Analysis Results"
                )
        
        def show_loading_animation(is_question: bool = False):
            """Show loading animation while processing."""
            if is_question:
                return """## Analysis in Progress...

<div style="text-align: center; padding: 20px;">
    <div style="font-size: 18px; margin-bottom: 15px;">
        <strong>Analyzing your question...</strong>
    </div>
    <div style="display: flex; justify-content: center; max-width: 600px; margin: 0 auto;">
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Analyst Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Processing query...</div>
        </div>
    </div>
    <div style="margin-top: 25px; font-size: 14px; color: #888;">
        This may take a moment... Please wait while the agent processes your question.
    </div>
</div>"""
            else:
                return """## Analysis in Progress...

<div style="text-align: center; padding: 20px;">
    <div style="font-size: 18px; margin-bottom: 15px;">
        <strong>Agents are working in parallel...</strong>
    </div>
    <div style="display: flex; justify-content: space-around; max-width: 600px; margin: 0 auto; flex-wrap: wrap;">
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Engineer Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Examining dataset...</div>
        </div>
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Analyst Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Analyzing data...</div>
        </div>
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Storyteller Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Creating storylines...</div>
        </div>
    </div>
    <div style="margin-top: 25px; font-size: 14px; color: #888;">
        This may take a moment... Please wait while the agents process your data.
    </div>
</div>"""
        
        def on_file_upload(file):
            """Handle file upload - show analyze button and reset state."""
            if file is not None:
                return gr.update(visible=True), ""
            return gr.update(visible=False), ""
        
        def start_full_analysis(file, engineer_result: str = ""):
            """Start full analysis and show loading animation."""
            loading_msg = show_loading_animation(is_question=False)
            return gr.update(visible=True, value=loading_msg), gr.update(value="")
        
        def complete_full_analysis(file, engineer_result: str = ""):
            """Complete full analysis and return results."""
            result, new_engineer_result = process_file_and_analyze(file, "", engineer_result)
            if result.startswith("Error:") or result.startswith("Please upload"):
                result = f"### {result}"
            return result, gr.update(visible=False), new_engineer_result
        
        def start_question_analysis(file, user_query: str = ""):
            """Start question analysis and show loading animation."""
            loading_msg = show_loading_animation(is_question=True)
            return gr.update(visible=True, value=loading_msg), gr.update(visible=True, value="")
        
        def complete_question_analysis(file, user_query: str = ""):
            """Complete question analysis and return results."""
            result = process_question_only(file, user_query)
            if result.startswith("Error:") or result.startswith("Please"):
                result = f"### {result}"
            else:
                result = f"""<div style="background-color: #f0f7ff; border: 2px solid #4a90e2; border-radius: 8px; padding: 15px; margin: 10px 0;">
{result}
</div>"""
            return result, gr.update(visible=False)
        
        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[analyze_btn, engineer_state]
        )
        
        analyze_btn.click(
            fn=start_full_analysis,
            inputs=[file_input, engineer_state],
            outputs=[status_output, merged_output]
        ).then(
            fn=complete_full_analysis,
            inputs=[file_input, engineer_state],
            outputs=[merged_output, status_output, engineer_state]
        )
        
        query_btn.click(
            fn=start_question_analysis,
            inputs=[file_input, query_input],
            outputs=[status_output, question_output]
        ).then(
            fn=complete_question_analysis,
            inputs=[file_input, query_input],
            outputs=[question_output, status_output]
        )
        
        query_input.submit(
            fn=start_question_analysis,
            inputs=[file_input, query_input],
            outputs=[status_output, question_output]
        ).then(
            fn=complete_question_analysis,
            inputs=[file_input, query_input],
            outputs=[question_output, status_output]
        )
    
    return app


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the NBA data analysis crew."""
    print("=" * 60)
    print("NBA 2024-25 Data Analysis with CrewAI")
    print("Using LLM Provider: OPENAI")
    print("=" * 60)
    print()
    
    if not os.path.exists(NBA_DATA_PATH):
        print(f"Error: {NBA_DATA_PATH} not found!")
        return
    
    print(f"Loading data from {NBA_DATA_PATH}...")
    try:
        df = pd.read_csv(NBA_DATA_PATH)
        print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Starting CrewAI agents...")
    print("Engineer Agent will process and clean the data...")
    print("Analyst Agent will analyze the data for insights...")
    print()
    print("-" * 60)
    print()
    
    crew = create_crew()
    result = crew.kickoff()
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print(result)

