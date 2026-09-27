"""Evaluation: base vs fine-tuned vs teacher on the held-out test set (§7, Task 2.8).

Generate outputs from any OpenAI-compatible server (llama-server locally, vLLM on Colab):
    python -m eval.run generate --label finetuned --base-url http://localhost:8000/v1 --model bureaucracy
Speed on the laptop CPU (one request at a time, first N examples):
    uv run python -m eval.run generate --label finetuned-cpu --base-url http://localhost:8080/v1 --limit 20 --concurrency 1
Metrics table:
    uv run python -m eval.run report base finetuned teacher
"""
import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

from app.prompts import LETTER_HEADINGS, REFUSAL

ROOT = Path(__file__).parent.parent
TEST_FILE = ROOT / "data" / "out" / "test.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
CITATION = re.compile(r"\[(\d+)\]")


def ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- generation ----------

async def generate(label: str, base_url: str, model: str, limit: int | None, concurrency: int, max_tokens: int) -> None:
    examples = list(enumerate(read_jsonl(TEST_FILE)))
    if limit:  # fixed random sample, so every type is represented and all models get the same examples
        examples = sorted(random.Random(0).sample(examples, limit), key=lambda pair: pair[0])
    client = AsyncOpenAI(base_url=base_url, api_key="local", timeout=900)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(i: int, ex: dict) -> dict:
        async with semaphore:
            start = time.perf_counter()
            res = await client.chat.completions.create(
                model=model, messages=ex["messages"][:2],  # system + user, without the reference answer
                temperature=0, max_tokens=max_tokens,
            )
            seconds = time.perf_counter() - start
        output = re.sub(r"<think>.*?</think>", "", res.choices[0].message.content or "", flags=re.S).strip()
        return {"id": i, "type": ex["type"], "output": output, "seconds": round(seconds, 2)}

    rows = await asyncio.gather(*(one(i, ex) for i, ex in examples))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{label}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} outputs → {out}")


# ---------- metrics ----------

def citation_validity(rows: list[dict], tests: list[dict]) -> float:
    """QA answers with ≥ 1 citation and only valid chunk numbers (refusals count as invalid)."""
    qa = [r for r in rows if r["type"] == "qa"]
    ok = 0
    for r in qa:
        cited = [int(n) for n in CITATION.findall(r["output"])]
        n_chunks = len(tests[r["id"]]["context_ids"])
        ok += bool(cited) and all(1 <= n <= n_chunks for n in cited) and REFUSAL not in r["output"]
    return ratio(ok, len(qa))


def correct_refusals(rows: list[dict]) -> float:
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    return ratio(sum(REFUSAL in r["output"] for r in unanswerable), len(unanswerable))


def wrong_refusals(rows: list[dict]) -> float:
    qa = [r for r in rows if r["type"] == "qa"]
    return ratio(sum(REFUSAL in r["output"] for r in qa), len(qa))


def letter_headings(rows: list[dict]) -> float:
    letters = [r for r in rows if r["type"] == "letter"]
    return ratio(sum(all(h in r["output"] for h in LETTER_HEADINGS) for r in letters), len(letters))


def deadline_match(rows: list[dict], tests: list[dict]) -> float:
    from data.filter_split import date_variants
    letters = [r for r in rows if r["type"] == "letter" and tests[r["id"]]["deadline"]]
    return ratio(sum(any(v in r["output"] for v in date_variants(tests[r["id"]]["deadline"])) for r in letters), len(letters))


def plain_text(rows: list[dict]) -> float:
    """Answers without markdown (**bold**, # headings): the bot sends plain text to Telegram."""
    return ratio(sum("**" not in r["output"] and not re.search(r"^#+ ", r["output"], re.M) for r in rows), len(rows))


def words_per_sentence(rows: list[dict]) -> float:
    sentences = [s for r in rows for s in re.split(r"[.?!\n]+", r["output"]) if s.split()]
    return sum(len(s.split()) for s in sentences) / len(sentences)


def seconds_per_answer(rows: list[dict]) -> float:
    return sum(r["seconds"] for r in rows) / len(rows)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def report(labels: list[str]) -> None:
    tests = read_jsonl(TEST_FILE)
    n_deadlines = sum(1 for t in tests if t["type"] == "letter" and t["deadline"])
    table = {}
    for label in labels:
        rows = read_jsonl(RESULTS_DIR / f"{label}.jsonl")
        speed_file = RESULTS_DIR / f"{label}-cpu.jsonl"  # laptop CPU timing, if measured
        table[label] = {
            "Citation validity (QA)": pct(citation_validity(rows, tests)),
            "Correct refusals (unanswerable)": pct(correct_refusals(rows)),
            "Wrong refusals (QA)": pct(wrong_refusals(rows)),
            "Letter: all 4 headings": pct(letter_headings(rows)),
            f"Letter: deadline match (n={n_deadlines})": pct(deadline_match(rows, tests)),
            "Plain text (no markdown)": pct(plain_text(rows)),
            "Words per sentence": f"{words_per_sentence(rows):.1f}",
            "Seconds per answer (CPU)": f"{seconds_per_answer(read_jsonl(speed_file)):.1f}" if speed_file.exists() else "—",
        }
    metrics = list(next(iter(table.values())))
    width = max(map(len, metrics))
    print(f"{'Metric':{width}} | " + " | ".join(f"{label:>12}" for label in labels))
    print("-" * (width + 15 * len(labels)))
    for m in metrics:
        print(f"{m:{width}} | " + " | ".join(f"{table[label][m]:>12}" for label in labels))
    print(f"\ntest set: {len(tests)} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--label", required=True)
    g.add_argument("--base-url", required=True)
    g.add_argument("--model", default="local")
    g.add_argument("--limit", type=int)
    g.add_argument("--concurrency", type=int, default=32)
    g.add_argument("--max-tokens", type=int, default=700)
    r = sub.add_parser("report")
    r.add_argument("labels", nargs="+")
    args = parser.parse_args()
    if args.cmd == "generate":
        asyncio.run(generate(args.label, args.base_url, args.model, args.limit, args.concurrency, args.max_tokens))
    else:
        report(args.labels)
