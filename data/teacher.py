"""Teacher model (DictaLM 3.0 12B) on vLLM, for data generation on Colab (§6.3).

Tokenizer fix: the repo's tokenizer_config says `LlamaTokenizerFast`, and transformers 5 then builds a
Llama tokenizer that drops all Hebrew (a whole Hebrew question becomes the single token "?").
We copy the tokenizer files locally and set `tokenizer_class` to the generic `PreTrainedTokenizerFast`,
which uses tokenizer.json as-is (byte-level BPE) and keeps the chat template.

Smoke test (Task 1.8): `python -m data.teacher` from the repo root.
"""
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

TEACHER_MODEL = "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-W4A16"
TOKENIZER_DIR = Path("/content/teacher_tokenizer")


def fixed_tokenizer(repo: str = TEACHER_MODEL, dst: Path = TOKENIZER_DIR) -> str:
    snapshot_download(repo, local_dir=dst,
                      allow_patterns=["tokenizer*", "special_tokens_map.json", "chat_template.jinja"])
    cfg_path = dst / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return str(dst)


def load_teacher() -> LLM:
    return LLM(
        model=TEACHER_MODEL,
        tokenizer=fixed_tokenizer(),
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
    )


if __name__ == "__main__":
    from app.prompts import SYSTEM, qa_user

    chunks_file = Path(__file__).parent / "out" / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines()]
    ctx = [c for c in chunks if c["title"] == "הגשת תביעה לדמי אבטלה"][:3]

    teacher = load_teacher()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": qa_user("איך מגישים תביעה לדמי אבטלה?", ctx)}]
    out = teacher.chat([messages], SamplingParams(temperature=0.7, max_tokens=1200))[0]
    print(f"===== PROMPT TOKENS: {len(out.prompt_token_ids)} =====")
    print("===== ANSWER =====")
    print(out.outputs[0].text)
