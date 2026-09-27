"""Telegram Bot API helpers (§8.4)."""
import logging

import httpx

from app.config import settings

# Every Bot API URL contains the token: never let httpx log request URLs.
logging.getLogger("httpx").setLevel(logging.WARNING)

BASE = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
FILE_BASE = f"https://api.telegram.org/file/bot{settings.telegram_bot_token}"
MAX_MESSAGE = 4000  # Telegram's limit is 4096 characters

http = httpx.AsyncClient(timeout=60)


async def send_text(chat_id: int, text: str) -> None:
    """Send text, split into pieces under Telegram's limit (at line breaks when possible)."""
    while text:
        piece = text[:MAX_MESSAGE]
        if len(text) > MAX_MESSAGE and "\n" in piece:
            piece = piece[:piece.rfind("\n")]
        await http.post(f"{BASE}/sendMessage", json={"chat_id": chat_id, "text": piece})
        text = text[len(piece):].lstrip("\n")


async def typing(chat_id: int) -> None:
    """Show "typing…" (lasts ~5 seconds or until the next message)."""
    await http.post(f"{BASE}/sendChatAction", json={"chat_id": chat_id, "action": "typing"})


async def download(file_id: str) -> bytes:
    r = await http.get(f"{BASE}/getFile", params={"file_id": file_id})
    r.raise_for_status()
    return (await http.get(f"{FILE_BASE}/{r.json()['result']['file_path']}")).content
