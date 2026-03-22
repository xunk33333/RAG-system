from __future__ import annotations
from pathlib import Path

import re
import unicodedata
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace/unicode noise while preserving content."""
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text).lower())


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(normalized)]
    return [p for p in parts if p]


def first_sentence(text: str) -> str:
    sentences = split_sentences(text)
    if sentences:
        return sentences[0]
    return normalize_text(text)


def safe_join(items: Iterable[str], sep: str = " ") -> str:
    return sep.join(i for i in items if i).strip()



import json
def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
