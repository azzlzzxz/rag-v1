"""查看 Chroma 中的 Node metadata：python scripts/list_nodes.py --limit 10。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    client = chromadb.PersistentClient(
        path=str(settings.VECTOR_DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=settings.CHROMA_ANONYMIZED_TELEMETRY),
    )
    collection = client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
    values = collection.get(limit=args.limit, include=["documents", "metadatas"])
    for node_id, content, metadata in zip(values["ids"], values["documents"], values["metadatas"]):
        print({"node_id": node_id, "text": content[:120], "metadata": metadata})


if __name__ == "__main__":
    main()
