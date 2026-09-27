"""Client for the local llama-server (OpenAI-compatible API, §8.1)."""
import re

from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(base_url=settings.llm_base_url, api_key="local")

# Training answers are ≤ 250 words (~450 tokens). A longer output means the model is copying the
# context; the cap bounds the wait (~8 tokens/s on the laptop).
MAX_TOKENS = 450


async def chat(messages: list[dict]) -> str:
    res = await client.chat.completions.create(
        model=settings.llm_model, messages=messages,
        temperature=0,  # same as the evaluation; at 0.3 the model copied long context passages more often
        max_tokens=MAX_TOKENS,
    )
    text = re.sub(r"<think>.*?</think>", "", res.choices[0].message.content or "", flags=re.S).strip()
    if res.choices[0].finish_reason == "length" and "\n" in text:
        text = text[:text.rfind("\n")].rstrip() + "\n…"  # cut at the last complete line
    return text
