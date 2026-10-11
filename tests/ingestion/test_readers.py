from pathlib import Path

import pandas as pd
import pymupdf

from core.ingestion.readers import EnterpriseDocumentReader


def test_pdf_reader_preserves_one_based_page_number(tmp_path: Path):
    pdf_path = tmp_path / "handbook.pdf"
    pdf = pymupdf.open()
    for text in ("first page policy", "second page policy"):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(pdf_path)
    pdf.close()

    documents = EnterpriseDocumentReader().load_data(
        pdf_path, extra_info={"category": "hr", "original_filename": "员工手册.pdf"}
    )

    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert len({document.id_ for document in documents}) == 2
    assert all(document.metadata["document_id"] == documents[0].metadata["document_id"] for document in documents)


def test_excel_reader_repeats_headers_for_every_row_group(tmp_path: Path):
    excel_path = tmp_path / "benefits.xlsx"
    pd.DataFrame({"姓名": ["张三", "李四", "王五"], "额度": [100, 200, 300]}).to_excel(
        excel_path, sheet_name="福利", index=False
    )

    documents = EnterpriseDocumentReader(excel_rows_per_document=2).load_data(excel_path)

    assert len(documents) == 2
    assert all("| 姓名 | 额度 |" in document.text for document in documents)
    assert [document.metadata["row_start"] for document in documents] == [1, 3]
    assert all(document.metadata["sheet_name"] == "福利" for document in documents)


def test_same_logical_file_gets_stable_document_and_source_ids(tmp_path: Path):
    first_path = tmp_path / "random-a.txt"
    second_path = tmp_path / "random-b.txt"
    first_path.write_text("相同内容", encoding="utf-8")
    second_path.write_text("相同内容", encoding="utf-8")
    reader = EnterpriseDocumentReader()
    metadata = {"category": "policy", "original_filename": "制度.txt", "ingested_at": "fixed"}

    first = reader.load_data(first_path, extra_info=metadata)[0]
    second = reader.load_data(second_path, extra_info=metadata)[0]

    assert first.metadata["document_id"] == second.metadata["document_id"]
    assert first.id_ == second.id_
    assert first.metadata["content_hash"] == second.metadata["content_hash"]
