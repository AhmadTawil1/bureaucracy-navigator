"""Client for the local llama-server (OpenAI-compatible API, §8.1)."""
import re

from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(base_url=settings.llm_base_url, api_key="local")


async def chat(messages: list[dict]) -> str:
    res = await client.chat.completions.create(
        model=settings.llm_model, messages=messages, temperature=0.3, max_tokens=900,
    )
    text = res.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
