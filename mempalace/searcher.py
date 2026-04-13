#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Semantic search against the palace.
Returns verbatim text — the actual words, never summaries.
"""

import logging
from pathlib import Path

from .palace import get_collection, get_closets_collection

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


def build_where_filter(wing: str = None, room: str = None) -> dict:
    """Build ChromaDB where filter for wing/room filtering."""
    if wing and room:
        return {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        return {"wing": wing}
    elif room:
        return {"room": room}
    return {}


def search(query: str, palace_path: str, wing: str = None, room: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).
    """
    try:
        col = get_collection(palace_path, create=False)
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        raise SearchError(f"No palace found at {palace_path}")

    where = build_where_filter(wing, room)

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        similarity = round(max(0.0, 1 - dist), 3)
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  {similarity}")
        print()
        # Print the verbatim text, indented
        for line in doc.strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def search_memories(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    max_distance: float = 0.0,
) -> dict:
    """Programmatic search — returns a dict instead of printing.

    Used by the MCP server and other callers that need data.

    Args:
        query: Natural language search query.
        palace_path: Path to the ChromaDB palace directory.
        wing: Optional wing filter.
        room: Optional room filter.
        n_results: Max results to return.
        max_distance: Max cosine distance threshold. The palace collection uses
            cosine distance (hnsw:space=cosine) — 0 = identical, 2 = opposite.
            Results with distance > this value are filtered out. A value of
            0.0 disables filtering. Typical useful range: 0.3–1.0.
    """
    try:
        drawers_col = get_collection(palace_path, create=False)
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    where = build_where_filter(wing, room)

    # Try closet-first search: search the compact index, then hydrate drawers
    closet_hits = []
    try:
        closets_col = get_closets_collection(palace_path, create=False)
        ckwargs = {
            "query_texts": [query],
            "n_results": n_results * 2,  # over-fetch closets to find best drawers
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)
        if closet_results["documents"][0]:
            closet_hits = list(zip(
                closet_results["documents"][0],
                closet_results["metadatas"][0],
                closet_results["distances"][0],
            ))
    except Exception:
        pass  # no closets yet — fall through to direct drawer search

    # If closets found results, hydrate the referenced drawers
    if closet_hits:
        import re
        seen_sources = set()
        hits = []
        for closet_doc, closet_meta, closet_dist in closet_hits:
            source = closet_meta.get("source_file", "")
            if source in seen_sources:
                continue
            seen_sources.add(source)

            # Find drawers for this source file
            try:
                drawer_results = drawers_col.get(
                    where={"source_file": source},
                    include=["documents", "metadatas"],
                )
                if drawer_results.get("ids"):
                    # Combine all drawer content for this file
                    full_text = "\n\n".join(drawer_results["documents"])
                    meta = drawer_results["metadatas"][0]
                    hits.append({
                        "text": full_text,
                        "wing": meta.get("wing", "unknown"),
                        "room": meta.get("room", "unknown"),
                        "source_file": Path(source).name,
                        "similarity": round(max(0.0, 1 - closet_dist), 3),
                        "distance": round(closet_dist, 4),
                        "matched_via": "closet",
                        "closet_preview": closet_doc[:200],
                    })
            except Exception:
                pass

            if len(hits) >= n_results:
                break

        if hits:
            return {
                "query": query,
                "filters": {"wing": wing, "room": room},
                "total_before_filter": len(closet_hits),
                "results": hits,
            }

    # Fallback: direct drawer search (no closets yet, or closets empty)
    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = drawers_col.query(**kwargs)
    except Exception as e:
        return {"error": f"Search error: {e}"}

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        # Filter on raw distance before rounding to avoid precision loss
        if max_distance > 0.0 and dist > max_distance:
            continue
        hits.append(
            {
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "similarity": round(max(0.0, 1 - dist), 3),
                "distance": round(dist, 4),
            }
        )

    return {
        "query": query,
        "filters": {"wing": wing, "room": room},
        "total_before_filter": len(docs),
        "results": hits,
    }
