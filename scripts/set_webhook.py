"""Register the webhook with Telegram (§8.4): uv run python scripts/set_webhook.py"""
import httpx

from app.config import settings

API = f"https://api.telegram.org/bot{settings.telegram_bot_token}"

result = httpx.post(f"{API}/setWebhook", json={
    "url": f"{settings.public_url}/telegram/webhook",
    "secret_token": settings.telegram_webhook_secret,
    "allowed_updates": ["message"],
    "drop_pending_updates": True,
}).json()
print(result)

info = httpx.get(f"{API}/getWebhookInfo").json()["result"]
print({k: info.get(k) for k in ("url", "pending_update_count", "last_error_message")})
