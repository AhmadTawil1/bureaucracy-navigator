"""Per-chat document memory: the masked letter text, kept 30 minutes for follow-up questions (§8.3).

In memory only: a restart forgets everything, which is fine for privacy.
"""
import time

TTL_SECONDS = 30 * 60

_store: dict[int, tuple[str, float]] = {}  # chat_id → (masked text, expires_at)


def set(chat_id: int, text: str, now: float | None = None) -> None:  # noqa: A001 (plan's name)
    _store[chat_id] = (text, (now or time.time()) + TTL_SECONDS)


def get(chat_id: int, now: float | None = None) -> str | None:
    entry = _store.get(chat_id)
    if entry is None:
        return None
    text, expires_at = entry
    if (now or time.time()) >= expires_at:
        del _store[chat_id]
        return None
    return text


def forget(chat_id: int) -> bool:
    """Delete the chat's document; True if there was one."""
    return _store.pop(chat_id, None) is not None
