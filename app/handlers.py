"""Message routing: text / PDF / photo / commands (§8.4)."""
import asyncio
import re
from contextlib import asynccontextmanager
from urllib.parse import quote

from app import documents, llm, privacy, rag, sessions, telegram
from app.config import settings
from app.prompts import REFUSAL, REFUSAL_FULL, SYSTEM, letter_user, qa_user

START = """שלום! אני נווט הבירוקרטיה 🧭

אפשר:
• לשאול שאלה על זכויות, למשל: "איך מגישים תביעה לדמי אבטלה?"
• לשלוח מכתב רשמי כקובץ PDF או כתמונה, ואסכם אותו בעברית פשוטה: מה כתוב, מה צריך לעשות, עד מתי ולמי לפנות.

המידע מבוסס על אתר "כל זכות". הוא מידע כללי ואינו מהווה ייעוץ משפטי.

פרטיות: קבצים מעובדים בזיכרון בלבד. מספרי ת״ז, טלפונים וכתובות אימייל מוסתרים, והמכתב נשמר 30 דקות כדי לענות על שאלות המשך. הפקודה /forget מוחקת אותו מיד. ההודעות עוברות דרך השרתים של טלגרם."""
HELP = START
FORGOT = "מחקתי את המסמך שלך."
NOTHING_TO_FORGET = "אין מסמך שמור."
NOT_PDF = "אני יודע לקרוא רק קבצי PDF או תמונות."
UNSUPPORTED = "אפשר לשלוח שאלה בטקסט, קובץ PDF או תמונה של מכתב."
TOO_BIG = "הקובץ גדול מדי. אפשר לשלוח קובץ עד 20MB ועד 10 עמודים."
UNREADABLE = "לא הצלחתי לקרוא טקסט מהמסמך. אפשר לנסות תמונה חדה יותר, באור טוב ובלי צל."
READING = "קיבלתי את המסמך, אני קורא אותו..."
FOLLOW_UP_HINT = "אפשר לשאול שאלות על המכתב ב-30 הדקות הקרובות. /forget מוחק אותו."
DISCLAIMER = "המידע כללי ואינו מהווה ייעוץ משפטי."

SEARCH_URL = "https://www.kolzchut.org.il/w/he/index.php?search="
MAX_FILE_BYTES = 20 * 1024 * 1024  # Telegram Bot API download limit
MAX_LETTER_CHARS = 6000  # keeps the prompt well inside the model's context
LETTER_QUERY_WORDS = 300
LETTER_TITLE = "המכתב שלך"

llm_lock = asyncio.Semaphore(1)  # one generation at a time: the laptop can't run two


@asynccontextmanager
async def keep_typing(chat_id: int):
    """Keep "typing…" visible while the model works (Telegram shows it for ~5 s per call)."""
    async def loop():
        while True:
            await telegram.typing(chat_id)
            await asyncio.sleep(4)
    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()


def refusal(question: str) -> str:
    return f"{REFUSAL_FULL}\n{SEARCH_URL}{quote(question)}"


def with_sources(answer: str, chunks: list[dict]) -> str:
    """Append one line per cited chunk: [n] title — url (the user's letter has no url)."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(chunks)})
    lines = [f"[{n}] {chunks[n - 1]['title']}" + (f" — {chunks[n - 1]['url']}" if chunks[n - 1].get("url") else "")
             for n in cited]
    sources = "\n\nמקורות:\n" + "\n".join(lines) if lines else ""
    return f"{answer}{sources}\n\n{DISCLAIMER}"


async def ask_model(chat_id: int, user_prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}]
    async with keep_typing(chat_id), llm_lock:
        return await llm.chat(messages)


# ---------- text ----------

async def answer_question(chat_id: int, question: str) -> None:
    letter = sessions.get(chat_id)
    hits = await asyncio.to_thread(rag.search, question, 3 if letter else 4)
    if not letter and (not hits or hits[0][0] < settings.min_retrieval_score):  # off-topic: no LLM call
        await telegram.send_text(chat_id, refusal(question))
        return
    chunks = [chunk for _, chunk in hits]
    if letter:  # follow-up about the stored letter: the letter itself is context [1]
        chunks = [{"title": LETTER_TITLE, "section": "", "text": letter}] + chunks
    answer = await ask_model(chat_id, qa_user(question, chunks))
    if REFUSAL in answer:
        await telegram.send_text(chat_id, refusal(question))
    else:
        await telegram.send_text(chat_id, with_sources(answer, chunks))


# ---------- documents ----------

def prepare_letter(raw_text: str) -> tuple[str, list[dict]]:
    """Masked letter text + the 3 Kol Zchut chunks retrieved for it (blocking: run in a thread)."""
    letter = privacy.mask(raw_text).strip()[:MAX_LETTER_CHARS]  # mask BEFORE storing or prompting
    if len(letter) < 30:
        return letter, []
    query = " ".join(letter.split()[:LETTER_QUERY_WORDS])
    return letter, [chunk for _, chunk in rag.search(query, 3)]


async def summarize_letter(chat_id: int, raw_text: str) -> None:
    letter, chunks = await asyncio.to_thread(prepare_letter, raw_text)
    if len(letter) < 30:
        await telegram.send_text(chat_id, UNREADABLE)
        return
    sessions.set(chat_id, letter)
    summary = await ask_model(chat_id, letter_user(letter, chunks))
    await telegram.send_text(chat_id, f"{with_sources(summary, chunks)}\n\n{FOLLOW_UP_HINT}")


async def handle_pdf(chat_id: int, document: dict) -> None:
    if document.get("file_size", 0) > MAX_FILE_BYTES:
        await telegram.send_text(chat_id, TOO_BIG)
        return
    await telegram.send_text(chat_id, READING)
    data = await telegram.download(document["file_id"])
    if await asyncio.to_thread(documents.pdf_pages, data) > documents.MAX_PAGES:
        await telegram.send_text(chat_id, TOO_BIG)
        return
    await summarize_letter(chat_id, await asyncio.to_thread(documents.pdf_to_text, data))


async def handle_photo(chat_id: int, photos: list[dict]) -> None:
    photo = photos[-1]  # Telegram sends several sizes; the last is the largest
    if photo.get("file_size", 0) > MAX_FILE_BYTES:
        await telegram.send_text(chat_id, TOO_BIG)
        return
    await telegram.send_text(chat_id, READING)
    data = await telegram.download(photo["file_id"])
    await summarize_letter(chat_id, await asyncio.to_thread(documents.image_to_text, data))


# ---------- routing ----------

async def handle_message(message: dict) -> None:
    chat_id = message["chat"]["id"]
    text = (message.get("text") or "").strip()

    if text.startswith("/start"):
        await telegram.send_text(chat_id, START)
    elif text.startswith("/help"):
        await telegram.send_text(chat_id, HELP)
    elif text.startswith("/forget"):
        await telegram.send_text(chat_id, FORGOT if sessions.forget(chat_id) else NOTHING_TO_FORGET)
    elif document := message.get("document"):
        if document.get("mime_type") == "application/pdf":
            await handle_pdf(chat_id, document)
        elif document.get("mime_type", "").startswith("image/"):  # photo sent "as a file"
            await handle_photo(chat_id, [document])
        else:
            await telegram.send_text(chat_id, NOT_PDF)
    elif photos := message.get("photo"):
        await handle_photo(chat_id, photos)
    elif text:
        await answer_question(chat_id, text)
    else:
        await telegram.send_text(chat_id, UNSUPPORTED)
