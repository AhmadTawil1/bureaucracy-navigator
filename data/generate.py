"""Teacher data generation (runs on Colab, §6.3). One step per run:

    python -u -m data.generate questions     # Task 1.9

Outputs go to Drive (GEN_DIR) so they survive the Colab session.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from vllm import SamplingParams

from data.teacher import finish, load_teacher

CHUNKS_FILE = Path(__file__).parent / "out" / "chunks.jsonl"
GEN_DIR = Path("/content/drive/MyDrive/bureaucracy-navigator/gen")
SEED = 42

N_QUESTIONS = 1000   # 850 for grounded answers (1.10) + 150 unanswerable (1.11)
MIN_CHUNK_WORDS = 40  # skip tiny chunks (e.g. a list of two office names)

QUESTIONS_PROMPT = """לפניך קטע מתוך אתר "כל זכות":

{title}
{text}

כתוב 3 שאלות מציאותיות שאזרח ישראלי עשוי לשאול, שהקטע הזה עונה עליהן.
כתוב אותן בעברית פשוטה, כמו שאנשים באמת שואלים (למשל "פוטרתי אחרי שנתיים, מגיעים לי דמי אבטלה?"), בלי להזכיר את "הקטע" או את "כל זכות".
החזר JSON בלבד, בפורמט: {{"questions": ["...", "...", "..."]}}"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def parse_questions(text: str) -> list[str] | None:
    """The {"questions": [...]} object from the reply (tolerates ```json fences and extra text)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        questions = json.loads(match.group())["questions"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    questions = [q.strip() for q in questions if isinstance(q, str) and len(q.strip()) > 5]
    return questions or None


def step_questions(teacher) -> None:
    """Task 1.9: 3 questions per chunk → sample N_QUESTIONS spread across articles."""
    chunks = [c for c in read_jsonl(CHUNKS_FILE) if len(c["text"].split()) >= MIN_CHUNK_WORDS]
    conversations = [[{"role": "user", "content": QUESTIONS_PROMPT.format(title=c["title"], text=c["text"])}]
                     for c in chunks]
    outputs = teacher.chat(conversations, SamplingParams(temperature=0.5, max_tokens=400))

    all_questions, failures, seen = [], 0, set()
    for chunk, out in zip(chunks, outputs):
        questions = parse_questions(out.outputs[0].text)
        if questions is None:
            failures += 1
            continue
        for q in questions:
            key = re.sub(r"\W+", " ", q).strip()
            if key not in seen:
                seen.add(key)
                all_questions.append({"question": q, "article": chunk["title"], "source_chunk_id": chunk["id"]})
    write_jsonl(GEN_DIR / "questions_all.jsonl", all_questions)

    # Round-robin over articles so every article gets questions, not only the long ones.
    rng = random.Random(SEED)
    by_article = defaultdict(list)
    for q in all_questions:
        by_article[q["article"]].append(q)
    for qs in by_article.values():
        rng.shuffle(qs)
    articles = list(by_article)
    rng.shuffle(articles)
    sample = []
    while len(sample) < N_QUESTIONS and any(by_article.values()):
        for article in articles:
            if by_article[article] and len(sample) < N_QUESTIONS:
                sample.append(by_article[article].pop())
    write_jsonl(GEN_DIR / "questions.jsonl", sample)

    print(f"chunks: {len(chunks)}, parse failures: {failures} ({failures / len(chunks):.1%})")
    print(f"unique questions: {len(all_questions)} → sampled {len(sample)} "
          f"from {len({q['article'] for q in sample})} articles → {GEN_DIR / 'questions.jsonl'}")
    for q in rng.sample(sample, 5):
        print(f"  [{q['article']}] {q['question']}")


STEPS = {"questions": step_questions}

if __name__ == "__main__":
    step = sys.argv[1]
    teacher = load_teacher()
    STEPS[step](teacher)
    sys.stdout.flush()
    finish()
