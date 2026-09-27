"""Task 3.11: the 5 synthetic letters (PDF + photo) through the bot's document pipeline, without Telegram.

    uv run python -m eval.letters_e2e     # needs llama-server on LLM_BASE_URL; stop the bot first (Qdrant lock)

Same code as the bot: documents.pdf_to_text / image_to_text → handlers.prepare_letter (mask + retrieve)
→ letter_user → llm.chat. Checks each output and prints a Markdown table for the README.
"""
import asyncio
import json
import re
from pathlib import Path

from app import documents, handlers, llm
from app.prompts import LETTER_HEADINGS, SYSTEM, letter_user
from data.filter_split import HEBREW_MONTHS, date_variants

LETTERS_DIR = Path(__file__).parent / "letters"
RESULTS_FILE = Path(__file__).parent / "results" / "letters.jsonl"
FAKE_ID = "000000018"

# Letter subject and the deadline(s) a correct summary must give (letter 3 has two dates).
LETTERS = {
    "letter1": ("ביטוח לאומי: השלמת מסמכים לדמי אבטלה", ["15/02/2026"]),
    "letter2": ("רשות המסים: מסמכים להחזר מס", ["30/04/2026"]),
    "letter3": ("מעסיק: זימון לשימוע", ["24/02/2026", "20/02/2026"]),
    "letter4": ("בעל דירה: סיום חוזה שכירות", ["31/03/2026"]),
    "letter5": ("עירייה: חוב ארנונה", ["12/02/2026"]),
}


def words(text: str) -> list[str]:
    return re.findall(r"[א-ת]{2,}", text)


def section(summary: str, heading: str) -> str:
    """Text under one heading, up to the next heading."""
    others = "|".join(re.escape(h) for h in LETTER_HEADINGS if h != heading)
    match = re.search(rf"{re.escape(heading)}:?(.*?)(?=(?:{others}):|\Z)", summary, re.S)
    return match.group(1) if match else ""


def deadline_ok(summary: str, deadlines: list[str]) -> bool:
    """The deadline must be under "עד מתי", not just somewhere in the summary."""
    when = section(summary, "עד מתי")
    variants = set()
    for d in deadlines:
        day, month, _ = (int(x) for x in d.split("/"))
        variants |= date_variants(d) | {f"{day} ב{HEBREW_MONTHS[month - 1]}"}  # "20 בפברואר" (no year) is fine
    return any(v in when for v in variants)


def word_recall(original: str, extracted: str) -> float:
    """Share of the letter's Hebrew words found in the extracted text (in reading order, so reversed text fails)."""
    found = set(words(extracted))
    return sum(w in found for w in words(original)) / len(words(original))


async def run_one(name: str, kind: str) -> dict:
    original = (LETTERS_DIR / f"{name}.md").read_text(encoding="utf-8")
    data = (LETTERS_DIR / f"{name}.{'pdf' if kind == 'PDF' else 'jpg'}").read_bytes()
    raw = documents.pdf_to_text(data) if kind == "PDF" else documents.image_to_text(data)
    letter, chunks = handlers.prepare_letter(raw)
    summary = await llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": letter_user(letter, chunks)}])

    subject, deadlines = LETTERS[name]
    return {
        "letter": name, "subject": subject, "kind": kind,
        "text_recall": round(word_recall(original, raw), 2),
        "reversed": documents.looks_reversed(raw),
        "headings": all(h in summary for h in LETTER_HEADINGS),
        "deadline": deadline_ok(summary, deadlines),
        "id_masked": FAKE_ID not in letter and FAKE_ID not in summary and "[ת״ז]" in letter,
        "summary": summary,
    }


async def main() -> None:
    rows = []
    for name in LETTERS:
        for kind in ("PDF", "Photo"):
            rows.append(await run_one(name, kind))
            print(f"done: {name} {kind}", flush=True)
    RESULTS_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    yes = lambda ok: "✅" if ok else "❌"
    print("\n| Letter | Input | Words recovered | Right order | 4 headings | Deadline | ID masked |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['subject']} | {r['kind']} | {r['text_recall']:.0%} | {yes(not r['reversed'])} | "
              f"{yes(r['headings'])} | {yes(r['deadline'])} | {yes(r['id_masked'])} |")
    n = len(rows)
    print(f"\nTotal: headings {sum(r['headings'] for r in rows)}/{n}, deadline {sum(r['deadline'] for r in rows)}/{n}, "
          f"ID masked {sum(r['id_masked'] for r in rows)}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
