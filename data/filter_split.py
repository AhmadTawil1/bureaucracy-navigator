"""Quality filters (§6.4, Task 1.14) and split by article (Task 1.16).

    uv run python -m data.filter_split

Reads the generated examples from data/out/gen/ (downloaded from Drive).
"""
import json
import re
from collections import Counter
from pathlib import Path

from app.prompts import LETTER_HEADINGS, REFUSAL

OUT_DIR = Path(__file__).parent / "out"
GEN_DIR = OUT_DIR / "gen"
FILTERED_FILE = OUT_DIR / "filtered.jsonl"
GEN_FILES = ["qa.jsonl", "unanswerable.jsonl", "letters.jsonl", "rewrites.jsonl"]

MAX_WORDS = 250
MIN_HEBREW = 0.7
CITATION = re.compile(r"\[(\d+)\]")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def strip_markdown(text: str) -> str:
    """The teacher still writes **bold** and # headings; the bot sends plain text."""
    text = text.replace("**", "")
    text = re.sub(r"^(\s*)\*\s+", r"\1- ", text, flags=re.M)  # "*   item" bullets → "- item"
    return re.sub(r"^\s*#+\s*", "", text, flags=re.M).strip()


def clean(ex: dict) -> dict:
    user, answer = ex["messages"][1], ex["messages"][2]
    answer["content"] = strip_markdown(answer["content"])
    if ex["type"] == "letter":
        # The letter is the input: real letters (OCR text) have no markdown either.
        head, rest = user["content"].split("\n\nקטעי מידע רלוונטיים:", 1)
        user["content"] = strip_markdown(head) + "\n\nקטעי מידע רלוונטיים:" + rest
        # The teacher's "מועד אחרון" header date is often not in the letter (which uses "תוך 60 יום"):
        # keep it only when the letter really contains it.
        if ex["deadline"] and not any(v in head for v in date_variants(ex["deadline"])):
            ex["deadline"] = None
    if ex["type"] == "rewrite":  # no chunks, so any [n] is made up
        answer["content"] = re.sub(r"\s*\[\d+\]", "", answer["content"])
    return ex


def question_of(ex: dict) -> str:
    # The last "שאלה:" is the real question; chunks can contain Kol Zchut FAQ lines that start with it too.
    content = ex["messages"][1]["content"]
    return content.rsplit("\n\nשאלה: ", 1)[1].split("\n")[0] if "\n\nשאלה: " in content else content


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text)


HEBREW_MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
                 "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]


def date_variants(date: str) -> set[str]:
    """15/03/2026 → 15/3/2026, 15.03.2026, 15-03-2026, "15 במרץ 2026", ..."""
    d, m, y = (int(x) for x in date.split("/"))
    numeric = {f"{dd}{sep}{mm}{sep}{y}" for dd in (str(d), f"{d:02d}") for mm in (str(m), f"{m:02d}") for sep in "/.-"}
    return numeric | {f"{d} ב{HEBREW_MONTHS[m - 1]} {y}", f"{d} {HEBREW_MONTHS[m - 1]} {y}"}


def drop_reason(ex: dict) -> str | None:
    """First failed rule, or None if the example is kept."""
    answer = ex["messages"][2]["content"]
    cited = [int(n) for n in CITATION.findall(answer)]
    n_chunks = len(ex.get("context_ids", []))

    if any(n < 1 or n > n_chunks for n in cited):
        return "citation to a missing chunk"
    if ex["type"] == "qa" and not cited:
        return "QA without citation"
    if ex["type"] == "unanswerable" and REFUSAL not in answer:
        return "unanswerable without refusal"
    if ex["type"] == "letter" and not all(h in answer for h in LETTER_HEADINGS):
        return "letter missing a heading"
    if ex["type"] == "letter" and ex["deadline"] and not any(v in answer for v in date_variants(ex["deadline"])):
        return "letter deadline not in summary"
    if len(answer.split()) > MAX_WORDS:
        return f"over {MAX_WORDS} words"
    letters = re.findall(r"[A-Za-zא-ת]", answer)
    if not letters or len(re.findall(r"[א-ת]", answer)) / len(letters) < MIN_HEBREW:
        return "under 70% Hebrew"
    return None


def filter_examples() -> list[dict]:
    examples = [clean(ex) for name in GEN_FILES for ex in read_jsonl(GEN_DIR / name)]
    kept, reasons, seen = [], Counter(), set()
    for ex in examples:
        reason = drop_reason(ex)
        if reason is None:
            key = (ex["type"], normalize(question_of(ex)))
            if key in seen:
                reason = "near-duplicate question"
            seen.add(key)
        if reason:
            reasons[reason] += 1
        else:
            kept.append(ex)

    write_jsonl(FILTERED_FILE, kept)
    total, dropped = len(examples), len(examples) - len(kept)
    print(f"examples: {total}, kept: {len(kept)}, dropped: {dropped} ({dropped / total:.1%})")
    for reason, n in reasons.most_common():
        print(f"  {n:4d}  {reason}")
    before, after = Counter(ex["type"] for ex in examples), Counter(ex["type"] for ex in kept)
    print("by type (kept / generated):", {t: f"{after[t]}/{before[t]}" for t in before})
    return kept


if __name__ == "__main__":
    filter_examples()
