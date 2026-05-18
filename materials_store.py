"""
File-backed materials store for grounding chatbot responses in uploaded resources.

The store keeps the original files on disk, extracts searchable text when possible,
and exposes a lightweight retrieval function for prompt grounding.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from docx import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None


BASE_DIR = Path(__file__).resolve().parent
MATERIALS_DIR = BASE_DIR / "materials"
UPLOADS_DIR = MATERIALS_DIR / "uploads"
TEXT_DIR = MATERIALS_DIR / "text"
INDEX_PATH = MATERIALS_DIR / "index.json"

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".xml", ".py", ".log"}


def ensure_storage() -> None:
    """Create the on-disk folders used by the materials store."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secure_filename(filename: str) -> str:
    """Lightweight filename sanitizer that does not require Werkzeug."""
    filename = Path(filename).name.strip().replace(" ", "_")
    filename = re.sub(r"[^A-Za-z0-9._-]", "", filename)
    return filename or "material"


def _load_index() -> list[dict]:
    ensure_storage()
    if not INDEX_PATH.exists():
        return []

    try:
        with INDEX_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_index(records: list[dict]) -> None:
    ensure_storage()
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def _text_path(material_id: str) -> Path:
    return TEXT_DIR / f"{material_id}.txt"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _chunk_text(text: str, size: int = 1100, overlap: int = 180) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    if len(cleaned) <= size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _extract_text_from_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in TEXT_EXTENSIONS:
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    if extension == ".pdf" and PdfReader is not None:
        try:
            reader = PdfReader(str(file_path))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages).strip()
        except Exception:
            return ""

    if extension == ".docx" and Document is not None:
        try:
            document = Document(str(file_path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        except Exception:
            return ""

    return ""


def _determine_type(filename: str) -> str:
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension in {"jpg", "jpeg", "png", "gif", "webp"}:
        return "image"
    if extension in {"mp4", "webm", "mov", "avi"}:
        return "video"
    if extension == "pdf":
        return "pdf"
    if extension in {"doc", "docx", "txt", "md", "csv", "json", "html", "htm"}:
        return "document"
    return "other"


def _public_record(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "original_name": record.get("original_name"),
        "size": record.get("size", 0),
        "file_type": record.get("file_type", "other"),
        "source": record.get("source", "student"),
        "uploaded_at": record.get("uploaded_at"),
        "stored_name": record.get("stored_name"),
    }


def list_materials(source: Optional[str] = None) -> list[dict]:
    """Return stored material metadata, optionally filtered by source page."""
    records = _load_index()
    if source:
        records = [record for record in records if record.get("source") == source]
    return [_public_record(record) for record in records]


def delete_material(material_id: str) -> bool:
    """Delete a material and its extracted text from disk."""
    records = _load_index()
    remaining_records: list[dict] = []
    deleted_record: Optional[dict] = None

    for record in records:
        if record.get("id") == material_id:
            deleted_record = record
        else:
            remaining_records.append(record)

    if not deleted_record:
        return False

    upload_path = UPLOADS_DIR / deleted_record.get("stored_name", "")
    if upload_path.exists():
        try:
            upload_path.unlink()
        except Exception:
            pass

    text_path = _text_path(material_id)
    if text_path.exists():
        try:
            text_path.unlink()
        except Exception:
            pass

    _save_index(remaining_records)
    return True


def store_materials(uploaded_files: Iterable, source: str = "student") -> list[dict]:
    """Save uploaded files to disk and update the searchable index."""
    ensure_storage()

    records = _load_index()
    saved_records: list[dict] = []

    for uploaded_file in uploaded_files:
        original_name = uploaded_file.filename or "material"
        safe_name = _secure_filename(original_name)
        file_extension = Path(safe_name).suffix.lower()
        material_id = uuid.uuid4().hex
        stored_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{material_id}{file_extension}"
        stored_path = UPLOADS_DIR / stored_name

        uploaded_file.save(str(stored_path))
        extracted_text = _extract_text_from_file(stored_path)
        text_path = _text_path(material_id)
        text_path.write_text(extracted_text, encoding="utf-8", errors="ignore")

        record = {
            "id": material_id,
            "original_name": original_name,
            "stored_name": stored_name,
            "text_path": text_path.name,
            "size": stored_path.stat().st_size,
            "mime_type": uploaded_file.mimetype,
            "file_type": _determine_type(original_name),
            "source": source,
            "uploaded_at": _now_iso(),
        }

        records.insert(0, record)
        saved_records.append(_public_record(record))

    _save_index(records)
    return saved_records


def _load_record_text(record: dict) -> str:
    text_file = TEXT_DIR / record.get("text_path", "")
    if not text_file.exists():
        return ""

    try:
        return text_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def retrieve_material_context(query: str, source: Optional[str] = None, limit: int = 3) -> str:
    """Build a grounded context block using the uploaded materials."""
    records = _load_index()
    if source:
        records = [record for record in records if record.get("source") == source]

    if not records:
        return ""

    query_terms = set(_tokenize(query))
    ranked_chunks: list[tuple[int, str, str]] = []

    for record in records:
        text = _load_record_text(record)
        if not text:
            candidate = record.get("original_name", "Untitled material")
            if query_terms and any(term in candidate.lower() for term in query_terms):
                ranked_chunks.append((1, candidate, "No extracted text was available for this file yet."))
            continue

        for chunk in _chunk_text(text):
            lowered = chunk.lower()
            score = 0

            if query.strip() and query.strip().lower() in lowered:
                score += 5

            if query_terms:
                score += sum(1 for term in query_terms if term in lowered)

            if score > 0:
                ranked_chunks.append((score, record.get("original_name", "Untitled material"), chunk))

    if not ranked_chunks:
        for record in records[:limit]:
            text = _load_record_text(record)
            source_name = record.get("original_name", "Untitled material")
            if text:
                fallback_excerpt = _chunk_text(text)[0][:900].strip()
                ranked_chunks.append((0, source_name, fallback_excerpt))
            else:
                ranked_chunks.append((0, source_name, "No extracted text was available for this file yet."))

    if not ranked_chunks:
        return ""

    ranked_chunks.sort(key=lambda item: item[0], reverse=True)
    top_chunks = ranked_chunks[:limit]

    lines = [
        "Use the uploaded materials below as the primary source of truth.",
        "If the answer is not in these materials, say so clearly instead of guessing.",
        "",
        "Relevant material excerpts:",
    ]

    for index, (_, source_name, excerpt) in enumerate(top_chunks, start=1):
        shortened = excerpt[:900].strip()
        lines.append(f"{index}. Source: {source_name}\n   Excerpt: {shortened}")

    return "\n".join(lines)