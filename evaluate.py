"""Evaluate the RAG pipeline and compare chunk sizes.

Run:  python evaluate.py
For each chunk size it reports:
  - answer accuracy: all expected keywords appear in the answer
  - retrieval hit rate: the correct source and page were among the retrieved chunks
"""
import json

from rag import answer, ingest

CHUNK_SIZES = [500, 1000]


def evaluate(size: int, cases: list[dict]) -> tuple[int, int]:
    ingest(size=size, overlap=size // 8)
    correct = 0
    hits = 0
    for case in cases:
        reply, chunks = answer(case["question"], chunk_size=size)
        if all(kw.lower() in reply.lower() for kw in case["expected_keywords"]):
            correct += 1
        if any(c["source"] == case["source"] and c["page"] == case["page"] for c in chunks):
            hits += 1
    return correct, hits


def main() -> None:
    with open("questions.json", encoding="utf-8") as f:
        cases = json.load(f)

    n = len(cases)
    print(f"{n} test questions\n")
    print(f"{'Chunk size':<12}{'Answer accuracy':<20}{'Retrieval hit rate'}")
    for size in CHUNK_SIZES:
        correct, hits = evaluate(size, cases)
        print(f"{size:<12}{f'{correct}/{n} ({correct / n:.0%})':<20}{hits}/{n} ({hits / n:.0%})")


if __name__ == "__main__":
    main()
