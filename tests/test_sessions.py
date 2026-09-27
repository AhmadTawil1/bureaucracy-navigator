from app import sessions


def test_expires_after_30_minutes():
    sessions.set(1, "מכתב", now=1000.0)
    assert sessions.get(1, now=1000.0 + 29 * 60) == "מכתב"
    assert sessions.get(1, now=1000.0 + 30 * 60) is None
    assert sessions.get(1, now=1000.0) is None  # expired entries are deleted, not just hidden


def test_forget():
    sessions.set(2, "מכתב", now=1000.0)
    assert sessions.forget(2) is True
    assert sessions.get(2, now=1000.0) is None
    assert sessions.forget(2) is False


def test_chats_are_separate():
    sessions.set(3, "א", now=1000.0)
    sessions.set(4, "ב", now=1000.0)
    assert sessions.get(3, now=1000.0) == "א"
    assert sessions.get(4, now=1000.0) == "ב"
