"""面向企业文档的格式感知 LlamaIndex Reader。

本模块只负责“读取和预切分”，不负责 Embedding 和写入向量库。完整调用关系是：

    上传文件
        → EnterpriseDocumentReader.load_data()
        → 按页、Sheet 或段落组生成 LlamaIndex Document
        → IngestionService 中的 SentenceSplitter 继续切成 Node
        → Embedding
        → Chroma

为什么 Reader 不直接生成最终 chunk？
    PDF 页码、Excel Sheet、Word 段落组属于业务语义边界，应在读取文件时保留；
    最终 chunk 大小则属于检索参数，统一交给 IngestionPipeline 配置。这样既不会丢失
    页码等来源信息，也能独立调整 CHUNK_SIZE。

最小用法：

    reader = EnterpriseDocumentReader(excel_rows_per_document=30)
    documents = reader.load_data(
        Path("upload/employee-policy.pdf"),
        extra_info={
            "category": "hr",
            "original_filename": "员工制度.pdf",
        },
    )

返回的每个 Document 都带有 document_id、file_name、category、content_hash，
PDF 还包含 page，Excel 还包含 sheet_name 和行号范围。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import docx2txt
import pandas as pd
import pymupdf
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from core.ingestion.ids import build_document_id, build_source_id, sha256_text


class EnterpriseDocumentReader(BaseReader):
    """将文件拆成可追溯的来源单元，再交给 Pipeline 做 token 切片。

    “来源单元”不是最终检索 chunk：

    - PDF：一页是一个来源单元；
    - DOCX：一组自然段是一个来源单元；
    - XLSX：一个 Sheet 中的一组行是一个来源单元；
    - Markdown：一个标题章节是一个来源单元；
    - TXT：整个文件先作为一个来源单元。

    来源单元有稳定 source_id。文件局部修改后，LlamaIndex 可以只更新发生变化的单元。
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".xlsx"}

    def __init__(self, excel_rows_per_document: int = 30) -> None:
        """初始化 Reader。

        Args:
            excel_rows_per_document: Excel 每个来源单元最多包含多少行。每组都会重复表头，
                所以即使只召回中间一组数据，也能知道每一列的含义。
        """
        self.excel_rows_per_document = excel_rows_per_document

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[Document]:
        """读取一个文件并返回带统一 metadata 的 LlamaIndex Document 列表。

        Args:
            file: 实际保存到磁盘的文件路径，文件名可能带随机上传前缀。
            extra_info: 上传接口提供的业务信息。常用字段包括 category、
                original_filename、document_id 和 ingested_at。

        Returns:
            按业务边界拆分后的 Document 列表。空白页或空白段落不会返回。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件扩展名不在支持列表中。

        注意：document_id 使用 original_filename，而不是磁盘上的随机文件名。
        因此再次上传同一业务文件时，系统能将其识别为更新而不是新文档。
        """
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {path.suffix.lower()}")

        # 复制一份，避免给调用方传入的字典追加字段，产生难以发现的副作用。
        metadata = dict(extra_info or {})
        original_filename = metadata.get("original_filename", path.name)
        category = metadata.get("category", "general")
        document_id = metadata.get("document_id") or build_document_id(category, original_filename)
        # ingested_at 表示文档首次进入知识库的时间；更新时由 manifest 复用，避免时间变化破坏去重 hash。
        ingested_at = metadata.get("ingested_at") or datetime.now(timezone.utc).isoformat()
        # common 会传播到该文件生成的每一个 Document，后续也会继承到每个 Node。
        common = {
            "document_id": document_id,
            "file_name": original_filename,
            "file_type": path.suffix.lower().lstrip("."),
            "source_path": f"upload/{original_filename}",
            "category": category,
            "ingested_at": ingested_at,
        }

        # 这里只做格式路由；每种格式的具体规则放在独立私有方法中，便于单测和替换。
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            units = self._read_pdf(path)
        elif suffix == ".docx":
            units = self._read_docx(path)
        elif suffix == ".xlsx":
            units = self._read_xlsx(path)
        else:
            units = self._read_text(path)

        documents: list[Document] = []
        for unit_key, text, unit_metadata in units:
            cleaned = text.strip()
            if not cleaned:
                continue
            # unit_key 描述页码/Sheet/段落组。内容变化不会改变 source_id，Pipeline 才能
            # 用“相同 ID、不同 hash”识别一次更新；如果 ID 也随内容变化，就只能识别成新增。
            source_id = build_source_id(document_id, unit_key)
            unit_hash = sha256_text(cleaned)
            doc_metadata = {**common, **unit_metadata, "content_hash": unit_hash}
            documents.append(
                Document(
                    text=cleaned,
                    id_=source_id,
                    metadata=doc_metadata,
                    # 这些字段只用于管理和追踪，不应参与向量语义，否则路径、时间等噪声会影响相似度。
                    excluded_embed_metadata_keys=["document_id", "source_path", "ingested_at", "content_hash"],
                    # LLM 回答时保留文件名、页码等引用信息，但不必看到内部路径和 hash。
                    excluded_llm_metadata_keys=["source_path", "ingested_at", "content_hash"],
                )
            )
        return documents

    @staticmethod
    def _read_pdf(path: Path) -> list[tuple[str, str, dict[str, Any]]]:
        """按页读取 PDF，返回 ``(单元键, 正文, 单元 metadata)``。

        页码使用从 1 开始的用户视角，而 PyMuPDF 内部页号从 0 开始。
        """
        # PDF 以页为来源单元，引用页码和单页变更都可以独立追踪。
        with pymupdf.open(path) as pdf:
            return [
                (f"page:{page.number + 1}", page.get_text("text"), {"page": page.number + 1})
                for page in pdf
            ]

    @staticmethod
    def _read_docx(path: Path) -> list[tuple[str, str, dict[str, Any]]]:
        """读取 DOCX，并将自然段按固定数量组合成来源单元。

        docx2txt 部署简单，但不会完整保留 Word 标题样式树。因此这里明确采用自然段分组，
        不假装拥有并不可靠的标题层级；如果以后需要标题树，可替换成本方法而不影响 Pipeline。
        """
        text = docx2txt.process(str(path)) or ""
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        # docx2txt 无法可靠保留 Word 样式，因此按自然段组装来源单元，避免把短段落切得过碎。
        groups = ["\n\n".join(paragraphs[index : index + 20]) for index in range(0, len(paragraphs), 20)]
        return [(f"paragraph-group:{index + 1}", value, {"section": index + 1}) for index, value in enumerate(groups)]

    def _read_xlsx(self, path: Path) -> list[tuple[str, str, dict[str, Any]]]:
        """将工作簿按 Sheet 和行组转换成 Markdown 表格。

        dtype=str 用来避免员工编号、订单号等业务标识被 Pandas 自动转成浮点数。
        row_start/row_end 会进入 metadata，回答表格问题时可以定位原始行范围。
        """
        # 每个行组都重复 Markdown 表头，使单独召回一个 chunk 时仍可理解每列含义。
        workbook = pd.read_excel(path, sheet_name=None, dtype=str)
        units: list[tuple[str, str, dict[str, Any]]] = []
        for sheet_name, frame in workbook.items():
            frame = frame.fillna("")
            if frame.empty and not len(frame.columns):
                continue
            for start in range(0, max(len(frame), 1), self.excel_rows_per_document):
                group = frame.iloc[start : start + self.excel_rows_per_document]
                markdown = self._dataframe_to_markdown(group)
                end = min(start + self.excel_rows_per_document, len(frame))
                units.append(
                    (
                        f"sheet:{sheet_name}:rows:{start + 1}-{end}",
                        f"## Sheet: {sheet_name}\n\n{markdown}",
                        {"sheet_name": str(sheet_name), "row_start": start + 1, "row_end": end},
                    )
                )
        return units

    @staticmethod
    def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
        """直接生成简单 Markdown 表格，避免为一个格式化动作额外引入 tabulate。

        单元格中的 ``|`` 必须转义、换行必须压平，否则会破坏 Markdown 表格结构。
        """
        columns = [str(column) for column in frame.columns]

        def clean(value: Any) -> str:
            return str(value).replace("|", "\\|").replace("\n", " ")

        header = "| " + " | ".join(map(clean, columns)) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        rows = [
            "| " + " | ".join(clean(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    @staticmethod
    def _read_text(path: Path) -> list[tuple[str, str, dict[str, Any]]]:
        """读取 UTF-8 文本；Markdown 优先按标题边界拆分。

        ``utf-8-sig`` 同时兼容普通 UTF-8 和带 BOM 的 Windows 文本文件。
        TXT 不在这里按字符数切片，最终 token 切片仍由 SentenceSplitter 统一完成。
        """
        text = path.read_text(encoding="utf-8-sig")
        # Markdown 标题是天然边界；普通 TXT 则按空行形成段落组。
        sections = re.split(r"(?m)(?=^#{1,6}\s+)", text) if path.suffix.lower() in {".md", ".markdown"} else [text]
        return [(f"section:{index + 1}", value, {"section": index + 1}) for index, value in enumerate(sections)]
