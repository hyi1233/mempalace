"""
palace.py — Shared palace operations.

Consolidates collection access patterns used by both miners and the MCP server.
"""

import contextlib
import hashlib
import os

from .backends.chroma import ChromaBackend

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}

_DEFAULT_BACKEND = ChromaBackend()


def get_collection(
    palace_path: str,
    collection_name: str = "mempalace_drawers",
    create: bool = True,
):
    """Get the palace collection through the backend layer."""
    return _DEFAULT_BACKEND.get_collection(
        palace_path,
        collection_name=collection_name,
        create=create,
    )


def get_closets_collection(palace_path: str, create: bool = True):
    """Get the closets collection — the searchable index layer."""
    return get_collection(palace_path, collection_name="mempalace_closets", create=create)


CLOSET_CHAR_LIMIT = 1500  # fill closet until ~1500 chars, then start a new one


def build_closet_text(source_file, drawer_ids, content, wing, room):
    """Build a compact closet entry from drawer content.

    Extracts topics, names, and key quotes into an AAAK-style pointer
    that tells the searcher which drawers to open.
    """
    import re
    # Extract proper nouns (capitalized words, 2+ occurrences)
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", content[:5000])
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    entities = sorted([w for w, c in word_freq.items() if c >= 2], key=lambda w: -word_freq[w])[:5]

    # Extract key phrases
    topics = []
    for pattern in [
        r"(?:built|fixed|wrote|added|pushed|tested|created|decided|migrated)\s+[\w\s]{3,30}",
    ]:
        topics.extend(re.findall(pattern, content[:5000], re.IGNORECASE))
    topics = list(dict.fromkeys(t.strip().lower() for t in topics))[:8]

    # Extract first quote
    quotes = re.findall(r'"([^"]{15,100})"', content[:5000])
    quote = quotes[0] if quotes else ""

    # Build pointer lines
    entity_str = ";".join(entities[:5]) if entities else ""
    lines = []
    for topic in topics:
        pointer = f"{topic}|{entity_str}|→{','.join(drawer_ids[:3])}"
        lines.append(pointer)
    if quote:
        lines.append(f'"{quote}"|{entity_str}|→{",".join(drawer_ids[:3])}')
    if not lines:
        lines.append(f"{wing}/{room}|{entity_str}|→{','.join(drawer_ids[:3])}")

    return "\n".join(lines)


def upsert_closet(closets_col, closet_id, closet_text, metadata):
    """Add or update a closet. Respects CLOSET_CHAR_LIMIT."""
    try:
        existing = closets_col.get(ids=[closet_id])
        if existing.get("ids"):
            old_text = existing["documents"][0]
            if len(old_text) + len(closet_text) + 1 <= CLOSET_CHAR_LIMIT:
                closet_text = old_text + "\n" + closet_text
            # else: start fresh — old closet was full
    except Exception:
        pass
    closets_col.upsert(documents=[closet_text], ids=[closet_id], metadatas=[metadata])


@contextlib.contextmanager
def mine_lock(source_file: str):
    """Cross-platform file lock for mine operations.

    Prevents multiple agents from mining the same file simultaneously,
    which causes duplicate drawers when the delete+insert cycle interleaves.
    """
    lock_dir = os.path.join(os.path.expanduser("~"), ".mempalace", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(
        lock_dir, hashlib.sha256(source_file.encode()).hexdigest()[:16] + ".lock"
    )

    lf = open(lock_path, "w")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception:
            pass
        lf.close()


def file_already_mined(collection, source_file: str, check_mtime: bool = False) -> bool:
    """Check if a file has already been filed in the palace.

    When check_mtime=True (used by project miner), returns False if the file
    has been modified since it was last mined, so it gets re-mined.
    When check_mtime=False (used by convo miner), just checks existence.
    """
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        if not results.get("ids"):
            return False
        if check_mtime:
            stored_meta = results.get("metadatas", [{}])[0]
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return abs(float(stored_mtime) - current_mtime) < 0.001
        return True
    except Exception:
        return False
