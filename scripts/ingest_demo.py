"""从一个或多个文件/目录摄取演示文档。

示例：
    python scripts/ingest_demo.py tests/sample_docs evaluation/sample_docs --category demo
"""

import argparse
import sys
from pathlib import Path

# 允许按文档中的 `python scripts/ingest_demo.py` 方式直接执行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ingestion import get_ingestion_service
from core.ingestion.readers import EnterpriseDocumentReader


def iter_supported_paths(inputs: list[Path]) -> list[Path]:
    """展开多个输入路径，去重后只保留 Reader 支持的普通文件。"""
    candidates: list[Path] = []
    for input_path in inputs:
        candidates.extend([input_path] if input_path.is_file() else sorted(input_path.iterdir()))
    # resolve 后去重，避免同一文件通过目录和显式路径被摄取两次。
    unique = {path.resolve(): path for path in candidates}
    return [
        unique[key]
        for key in sorted(unique, key=str)
        if unique[key].is_file()
        and unique[key].suffix.lower() in EnterpriseDocumentReader.SUPPORTED_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--category", default="demo")
    args = parser.parse_args()
    service = get_ingestion_service()
    for path in iter_supported_paths(args.paths):
        result = service.ingest_file(path, category=args.category, original_filename=path.name)
        print(f"{path.name}: nodes={result.node_count}, skipped={result.skipped}, id={result.document_id}")


if __name__ == "__main__":
    main()
