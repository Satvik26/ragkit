"""Read a directory of text files and store each one through IngestionService."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragkit.services.ingestion import IngestionService

_SUFFIXES = {".txt", ".md", ".markdown"}
_MAX_CHARS = 100_000
_BATCH_DOCS = 16


class FolderIngestError(ValueError):
    """The folder request cannot be stored. The API turns this into HTTP 400."""


@dataclass(frozen=True, slots=True)
class IngestedFile:
    doc_id: str
    chunks: int


def ingest_folder(
    ingestion: IngestionService, root: Path, relative: str, acl_tags: frozenset[str]
) -> list[IngestedFile]:
    """Store every text or Markdown file under `relative`, which must stay inside `root`."""
    base = root.resolve()
    if Path(relative).is_absolute():
        raise FolderIngestError("path must be relative to the corpus directory")
    target = (base / relative).resolve()
    if not _inside(base, target):
        raise FolderIngestError("path escapes the corpus directory")
    if not target.is_dir():
        raise FolderIngestError("path is not a directory inside the corpus")

    pending: list[tuple[str, str]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
            continue
        resolved = path.resolve()
        if not _inside(base, resolved):
            continue
        relative_path = resolved.relative_to(base)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        text = _read(resolved)
        doc_id = relative_path.as_posix()
        pending.append((doc_id, text))

    stored: list[IngestedFile] = []
    for start in range(0, len(pending), _BATCH_DOCS):
        batch = pending[start : start + _BATCH_DOCS]
        for doc_id, chunks in ingestion.ingest_many(batch, acl_tags):
            stored.append(IngestedFile(doc_id, len(chunks)))
    return stored


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FolderIngestError(f"{path.name} is not UTF-8 text") from exc
    if len(text) > _MAX_CHARS:
        raise FolderIngestError(f"{path.name} is longer than {_MAX_CHARS} characters")
    return text
