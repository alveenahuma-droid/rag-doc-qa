"""Minimal RAG pipeline: PDFs -> chunks -> embeddings (Chroma) -> retrieval -> cited answer (Claude).

Usage:
    python rag.py ingest --docs docs --chunk-size 800
    python rag.py ask "What is the late submission policy?" --chunk-size 800
"""
import argparse
import os
from pathlib import Path

import chromadb
from anthropic import Anthropic
from pypdf import PdfReader

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
DB_PATH = "chroma_db"

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
db = chromadb.PersistentClient(path=DB_PATH)  # uses Chroma's built-in embedding model

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided. "
    "Cite the source after each claim in the form [file.pdf p.N]. "
    "If the context does not contain the answer, reply exactly: "
    "\"I couldn't find that in the documents.\""
)


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping character chunks."""
    step = max(size - overlap, 1)
    chunks = [text[i:i + size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


def get_collection(chunk_size: int):
    """One collection per chunk size, so sizes can be compared fairly."""
    return db.get_or_create_collection(f"docs_{chunk_size}")


def ingest(docs_dir: str = "docs", size: int = 800, overlap: int = 100) -> int:
    """Read every PDF in docs_dir, chunk it page by page and store it."""
    collection = get_collection(size)
    total = 0
    for pdf_path in sorted(Path(docs_dir).glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            chunks = chunk_text(page.extract_text() or "", size, overlap)
            if not chunks:
                continue
            collection.upsert(
                ids=[f"{pdf_path.name}-p{page_number}-c{i}" for i in range(len(chunks))],
                documents=chunks,
                metadatas=[{"source": pdf_path.name, "page": page_number} for _ in chunks],
            )
            total += len(chunks)
    return total


def retrieve(question: str, chunk_size: int = 800, k: int = 4) -> list[dict]:
    """Return the k most similar chunks with their source and page."""
    result = get_collection(chunk_size).query(query_texts=[question], n_results=k)
    return [
        {"text": text, "source": meta["source"], "page": meta["page"]}
        for text, meta in zip(result["documents"][0], result["metadatas"][0])
    ]


def answer(question: str, chunk_size: int = 800, k: int = 4) -> tuple[str, list[dict]]:
    """Retrieve context, ask Claude, and return (answer, retrieved chunks)."""
    chunks = retrieve(question, chunk_size, k)
    context = "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text, chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Q&A with RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index the PDFs in a folder")
    p_ingest.add_argument("--docs", default="docs")
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--overlap", type=int, default=100)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--chunk-size", type=int, default=800)
    p_ask.add_argument("-k", type=int, default=4)

    args = parser.parse_args()
    if args.command == "ingest":
        count = ingest(args.docs, args.chunk_size, args.overlap)
        print(f"Indexed {count} chunks.")
    else:
        reply, sources = answer(args.question, args.chunk_size, args.k)
        print(reply)
        print("\nRetrieved from:", ", ".join(sorted({f"{c['source']} p.{c['page']}" for c in sources})))
