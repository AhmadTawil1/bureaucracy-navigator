from fastapi.testclient import TestClient

from app import main
from app.config import settings

client = TestClient(main.app)


def update(update_id: int) -> dict:
    return {"update_id": update_id, "message": {"message_id": 1, "chat": {"id": 42}, "text": "שלום"}}


def test_wrong_secret_is_rejected(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "handle_message", lambda m: calls.append(m))
    r = client.post("/telegram/webhook", json=update(1), headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"})
    assert r.status_code == 401
    assert calls == []


def test_missing_secret_is_rejected():
    assert client.post("/telegram/webhook", json=update(2)).status_code == 401


def test_correct_secret_handles_message(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "handle_message", lambda m: calls.append(m))
    r = client.post("/telegram/webhook", json=update(3),
                    headers={"X-Telegram-Bot-Api-Secret-Token": settings.telegram_webhook_secret})
    assert r.status_code == 200
    assert [m["text"] for m in calls] == ["שלום"]


def test_same_update_handled_once(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "handle_message", lambda m: calls.append(m))
    headers = {"X-Telegram-Bot-Api-Secret-Token": settings.telegram_webhook_secret}
    for _ in range(2):  # Telegram retry
        assert client.post("/telegram/webhook", json=update(4), headers=headers).status_code == 200
    assert len(calls) == 1
