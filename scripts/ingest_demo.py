"""从目录摄取演示文档：python scripts/ingest_demo.py tests/sample_docs。"""

import argparse
import sys
from pathlib import Path

# 允许按文档中的 `python scripts/ingest_demo.py` 方式直接执行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ingestion import get_ingestion_service
from core.ingestion.readers import EnterpriseDocumentReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--category", default="demo")
    args = parser.parse_args()
    service = get_ingestion_service()
    paths = [args.path] if args.path.is_file() else sorted(args.path.iterdir())
    for path in paths:
        if path.suffix.lower() not in EnterpriseDocumentReader.SUPPORTED_EXTENSIONS:
            continue
        result = service.ingest_file(path, category=args.category, original_filename=path.name)
        print(f"{path.name}: nodes={result.node_count}, skipped={result.skipped}, id={result.document_id}")


if __name__ == "__main__":
    main()
