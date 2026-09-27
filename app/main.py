"""FastAPI webhook for Telegram (§8.4)."""
import asyncio
import hmac
from collections import deque
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from app import rag
from app.config import settings
from app.handlers import handle_message


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(rag.warm_up)
    yield


app = FastAPI(title="Bureaucracy Navigator", lifespan=lifespan)

# Telegram retries an update if it doesn't get a fast 200: remember recent update ids and handle each once.
seen_updates: deque[int] = deque(maxlen=1000)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    background: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(None),
):
    if not hmac.compare_digest(x_telegram_bot_api_secret_token or "", settings.telegram_webhook_secret):
        raise HTTPException(status_code=401)
    update = await request.json()
    if update["update_id"] in seen_updates:
        return {"ok": True}
    seen_updates.append(update["update_id"])
    if message := update.get("message"):
        background.add_task(handle_message, message)  # reply after the 200, so Telegram doesn't retry
    return {"ok": True}
