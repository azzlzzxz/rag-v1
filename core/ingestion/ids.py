"""摄取链路中的稳定 ID 与内容摘要。"""

import hashlib
import re
import unicodedata
from pathlib import Path

from llama_index.core.schema import BaseNode


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_identity_part(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"[^\w.\-]+", "-", normalized).strip("-") or "unnamed"


def build_document_id(category: str, original_filename: str) -> str:
    """业务文档 ID 不使用随机上传前缀，同名文档才能被识别为更新。"""
    identity = f"{normalize_identity_part(category)}/{normalize_identity_part(original_filename)}"
    return f"doc-{sha256_text(identity)[:24]}"


def build_source_id(document_id: str, unit_key: str) -> str:
    """页、Sheet 或段落组是去重与局部更新的最小来源单元。"""
    return f"src-{sha256_text(f'{document_id}:{unit_key}')[:32]}"


def stable_chunk_id(index: int, source: BaseNode) -> str:
    ref_doc_id = source.ref_doc_id or source.id_
    content = source.get_content()
    return f"chk-{sha256_text(f'{ref_doc_id}:{index}:{content}')[:32]}"
