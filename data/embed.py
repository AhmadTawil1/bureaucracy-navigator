"""Embed data/out/chunks.jsonl with e5 → data/out/embeddings.npy (run on Colab GPU).

Row i of embeddings.npy is line i of chunks.jsonl. Load into Qdrant locally with
`uv run python -c "from data.ingest_kolzchut import store; store()"`.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

OUT_DIR = Path(__file__).parent / "out"
CHUNKS_FILE = OUT_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = OUT_DIR / "embeddings.npy"
EMBED_MODEL = "intfloat/multilingual-e5-large"


def passage(chunk: dict) -> str:
    # e5 needs the "passage: " prefix. Section headings are already inside the text ("## ...").
    return f"passage: {chunk['title']}\n{chunk['text']}"


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()]
    model = SentenceTransformer(EMBED_MODEL)
    print(f"{len(chunks)} chunks on {model.device}")
    vectors = model.encode([passage(c) for c in chunks], normalize_embeddings=True,
                           batch_size=64, show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, vectors.astype(np.float32))
    print(f"saved {vectors.shape} → {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
