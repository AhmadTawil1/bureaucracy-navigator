"""Blind manual scoring (Task 2.11).

    uv run python -m eval.manual build    # → eval/manual/blind.csv (model hidden) + key.json (don't open it)
    ... fill the "correct" and "grounded" columns in blind.csv ...
    uv run python -m eval.manual score    # reveal the models and count

correct:  c = correct, p = partial, w = wrong (a refusal of an answerable question is w)
grounded: y = every claim is in the context chunks, n = something is not
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEST_FILE = ROOT / "data" / "out" / "test.jsonl"
CHUNKS_FILE = ROOT / "data" / "out" / "chunks.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
MANUAL_DIR = Path(__file__).parent / "manual"
BLIND_FILE = MANUAL_DIR / "blind.csv"
KEY_FILE = MANUAL_DIR / "key.json"
MODELS = ["base", "finetuned"]
N_QUESTIONS = 30
SEED = 7


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build() -> None:
    tests = read_jsonl(TEST_FILE)
    chunks = {c["id"]: c for c in read_jsonl(CHUNKS_FILE)}
    outputs = {m: {r["id"]: r["output"] for r in read_jsonl(RESULTS_DIR / f"{m}.jsonl")} for m in MODELS}

    rng = random.Random(SEED)
    qa_ids = rng.sample([i for i, t in enumerate(tests) if t["type"] == "qa"], N_QUESTIONS)
    rows = []
    for i in qa_ids:
        question = tests[i]["messages"][1]["content"].rsplit("\n\nשאלה: ", 1)[1].split("\n")[0]
        context = "\n\n".join(f"[{n}] {chunks[cid]['title']}\n{chunks[cid]['text']}"
                              for n, cid in enumerate(tests[i]["context_ids"], 1))
        for m in MODELS:
            rows.append({"test_id": i, "model": m, "question": question, "answer": outputs[m][i], "context": context})
    rng.shuffle(rows)

    MANUAL_DIR.mkdir(exist_ok=True)
    with BLIND_FILE.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel shows Hebrew
        writer = csv.writer(f)
        writer.writerow(["row", "question", "answer", "correct (c/p/w)", "grounded (y/n)", "context"])
        for n, r in enumerate(rows, 1):
            writer.writerow([n, r["question"], r["answer"], "", "", r["context"]])
    KEY_FILE.write_text(json.dumps({n: {"test_id": r["test_id"], "model": r["model"]} for n, r in enumerate(rows, 1)},
                                   indent=1), encoding="utf-8")
    print(f"{len(rows)} rows ({N_QUESTIONS} questions × {len(MODELS)} models) → {BLIND_FILE}")


def score() -> None:
    key = json.loads(KEY_FILE.read_text(encoding="utf-8"))
    counts = {m: Counter() for m in MODELS}
    with BLIND_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            model = key[row["row"]]["model"]
            correct, grounded = row["correct (c/p/w)"].strip().lower(), row["grounded (y/n)"].strip().lower()
            if correct not in {"c", "p", "w"} or grounded not in {"y", "n"}:
                sys.exit(f"row {row['row']}: fill correct with c/p/w and grounded with y/n")
            counts[model][correct] += 1
            counts[model]["grounded" if grounded == "y" else "not grounded"] += 1
    for m in MODELS:
        c = counts[m]
        print(f"{m:10s} correct {c['c']}/30, partial {c['p']}/30, wrong {c['w']}/30, grounded {c['grounded']}/30")


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
