import os

# Fake settings for tests (environment variables win over .env): tests never use the real bot token.
os.environ["TELEGRAM_BOT_TOKEN"] = "123:test-token"
os.environ["TELEGRAM_WEBHOOK_SECRET"] = "test-secret"
