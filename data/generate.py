"""Teacher data generation (runs on Colab, §6.3). One step per run:

    python -u -m data.generate questions     # Task 1.9
    python -u -m data.generate retrieve      # Task 1.10, part 1 (e5, no teacher)
    python -u -m data.generate answers       # Task 1.10, part 2
    python -u -m data.generate negatives     # Task 1.11, part 1 (e5, no teacher)
    python -u -m data.generate unanswerable  # Task 1.11, part 2 (teacher as judge)

Outputs go to Drive (GEN_DIR) so they survive the Colab session.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from vllm import SamplingParams

from app.prompts import REFUSAL_FULL, SYSTEM, _context, qa_user
from data.teacher import finish, load_teacher

CHUNKS_FILE = Path(__file__).parent / "out" / "chunks.jsonl"
DRIVE_DIR = Path("/content/drive/MyDrive/bureaucracy-navigator")
GEN_DIR = DRIVE_DIR / "gen"
EMBEDDINGS_FILE = DRIVE_DIR / "embeddings.npy"  # from data/embed.py, row i = chunk i
EMBED_MODEL = "intfloat/multilingual-e5-large"
SEED = 42

N_QUESTIONS = 1000   # 850 for grounded answers (1.10) + 150 unanswerable (1.11)
N_QA = 850
TOP_K = 4
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


def step_retrieve(_teacher=None) -> None:
    """Task 1.10a: split questions (850 QA / 150 held out for 1.11); top-4 e5 context per QA question."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks = read_jsonl(CHUNKS_FILE)
    vectors = np.load(EMBEDDINGS_FILE)
    assert len(vectors) == len(chunks), "embeddings.npy doesn't match chunks.jsonl"
    index_of = {c["id"]: i for i, c in enumerate(chunks)}

    rng = random.Random(SEED)
    questions = read_jsonl(GEN_DIR / "questions.jsonl")
    rng.shuffle(questions)
    qa, held_out = questions[:N_QA], questions[N_QA:]

    embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    q_vectors = embedder.encode([f"query: {q['question']}" for q in qa], normalize_embeddings=True, batch_size=64)
    top = np.argsort(-(q_vectors @ vectors.T), axis=1)[:, :TOP_K]

    source_found = 0
    for q, hits in zip(qa, top):
        ids = [int(i) for i in hits]
        source = index_of[q["source_chunk_id"]]
        if source in ids:
            source_found += 1
        else:
            ids[-1] = source  # the answer must be in the context
        rng.shuffle(ids)      # so the right chunk isn't always [1]
        q["context_ids"] = [chunks[i]["id"] for i in ids]
    write_jsonl(GEN_DIR / "qa_questions.jsonl", qa)
    write_jsonl(GEN_DIR / "heldout_questions.jsonl", held_out)
    print(f"QA questions: {len(qa)} (source chunk in e5 top-{TOP_K}: {source_found / len(qa):.0%}), "
          f"held out for 1.11: {len(held_out)}")


def step_answers(teacher) -> None:
    """Task 1.10b: the teacher answers each QA question with SYSTEM + qa_user (same prompt as the student)."""
    chunk_by_id = {c["id"]: c for c in read_jsonl(CHUNKS_FILE)}
    qa = read_jsonl(GEN_DIR / "qa_questions.jsonl")
    prompts = [[{"role": "system", "content": SYSTEM},
                {"role": "user", "content": qa_user(q["question"], [chunk_by_id[i] for i in q["context_ids"]])}]
               for q in qa]
    outputs = teacher.chat(prompts, SamplingParams(temperature=0.7, max_tokens=1200))

    examples = []
    for q, messages, out in zip(qa, prompts, outputs):
        answer = re.sub(r"<think>.*?</think>", "", out.outputs[0].text, flags=re.S).strip()
        examples.append({"type": "qa", "article": q["article"], "source_chunk_id": q["source_chunk_id"],
                         "context_ids": q["context_ids"],
                         "messages": messages + [{"role": "assistant", "content": answer}]})
    write_jsonl(GEN_DIR / "qa.jsonl", examples)
    print(f"QA examples: {len(examples)} → {GEN_DIR / 'qa.jsonl'}")

    for ex in random.Random(SEED).sample(examples, 5):
        source_pos = ex["context_ids"].index(ex["source_chunk_id"]) + 1
        print(f"\n--- Q: {ex['messages'][1]['content'].split('שאלה: ')[1].split(chr(10))[0]}")
        print(f"    context: {[chunk_by_id[i]['title'][:40] for i in ex['context_ids']]}  (source = [{source_pos}])")
        print(f"    A: {ex['messages'][2]['content'][:500]}")


N_UNANSWERABLE = 150
N_NEGATIVE_CANDIDATES = 1500  # only ~11% pass the judge: Kol Zchut articles overlap a lot

JUDGE_PROMPT = """קטעי מידע:
{context}

שאלה: {question}

האם אפשר לענות על השאלה, אפילו חלקית, רק לפי קטעי המידע? ענה במילה אחת: כן או לא."""


def step_negatives(_teacher=None) -> None:
    """Task 1.11a: candidate questions + top-4 chunks from OTHER articles (hard negatives)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks = read_jsonl(CHUNKS_FILE)
    vectors = np.load(EMBEDDINGS_FILE)

    # The 150 held-out questions first, then questions never sampled in 1.9 (no overlap with the QA set).
    used = {q["question"] for q in read_jsonl(GEN_DIR / "questions.jsonl")}
    unused = [q for q in read_jsonl(GEN_DIR / "questions_all.jsonl") if q["question"] not in used]
    random.Random(SEED).shuffle(unused)
    candidates = read_jsonl(GEN_DIR / "heldout_questions.jsonl")
    candidates += unused[:N_NEGATIVE_CANDIDATES - len(candidates)]

    embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    q_vectors = embedder.encode([f"query: {q['question']}" for q in candidates], normalize_embeddings=True,
                                batch_size=64)
    ranked = np.argsort(-(q_vectors @ vectors.T), axis=1)
    for q, order in zip(candidates, ranked):
        q["context_ids"] = [chunks[i]["id"] for i in order if chunks[i]["title"] != q["article"]][:TOP_K]
    write_jsonl(GEN_DIR / "negative_candidates.jsonl", candidates)
    print(f"negative candidates: {len(candidates)} → {GEN_DIR / 'negative_candidates.jsonl'}")


def step_unanswerable(teacher) -> None:
    """Task 1.11b: keep candidates the teacher judges unanswerable from their context → fixed refusal."""
    chunk_by_id = {c["id"]: c for c in read_jsonl(CHUNKS_FILE)}
    candidates = read_jsonl(GEN_DIR / "negative_candidates.jsonl")
    judge = [[{"role": "user", "content": JUDGE_PROMPT.format(
                  context=_context([chunk_by_id[i] for i in q["context_ids"]]),
                  question=q["question"])}]
             for q in candidates]
    outputs = teacher.chat(judge, SamplingParams(temperature=0, max_tokens=5))
    verdicts = [re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).strip() for o in outputs]
    kept = [q for q, v in zip(candidates, verdicts) if v.startswith("לא")]
    print(f"judge: {len(kept)} of {len(candidates)} unanswerable "
          f"(yes: {sum(v.startswith('כן') for v in verdicts)}, other: {sum(not v.startswith(('כן', 'לא')) for v in verdicts)})")

    rng = random.Random(SEED)
    kept = rng.sample(kept, min(N_UNANSWERABLE, len(kept)))
    examples = []
    for q in kept:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": qa_user(q["question"], [chunk_by_id[i] for i in q["context_ids"]])},
                    {"role": "assistant", "content": REFUSAL_FULL}]
        examples.append({"type": "unanswerable", "article": q["article"], "source_chunk_id": q["source_chunk_id"],
                         "context_ids": q["context_ids"], "messages": messages})
    write_jsonl(GEN_DIR / "unanswerable.jsonl", examples)
    print(f"unanswerable examples: {len(examples)} → {GEN_DIR / 'unanswerable.jsonl'}")
    for ex in rng.sample(examples, 3):
        print(f"\n--- Q ({ex['article']}): {ex['messages'][1]['content'].split('שאלה: ')[1].split(chr(10))[0]}")
        print(f"    context: {[i.split('#')[0][:40] for i in ex['context_ids']]}")


STEPS = {"questions": (step_questions, True), "retrieve": (step_retrieve, False), "answers": (step_answers, True),
         "negatives": (step_negatives, False), "unanswerable": (step_unanswerable, True)}

if __name__ == "__main__":
    step, needs_teacher = STEPS[sys.argv[1]]
    step(load_teacher() if needs_teacher else None)
    sys.stdout.flush()
    finish()
