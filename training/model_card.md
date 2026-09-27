---
base_model: dicta-il/DictaLM-3.0-1.7B-Instruct
library_name: peft
language:
- he
license: other
license_name: cc-by-nc-sa-2.5-il
license_link: https://creativecommons.org/licenses/by-nc-sa/2.5/il/
pipeline_tag: text-generation
tags:
- lora
- unsloth
- trl
- hebrew
- rag
---

# DictaLM 3.0 1.7B — Bureaucracy Navigator LoRA

LoRA adapters that turn [DictaLM 3.0 1.7B Instruct](https://huggingface.co/dicta-il/DictaLM-3.0-1.7B-Instruct) into a small assistant that explains Israeli bureaucracy in **simple Hebrew**, grounded in retrieved [Kol Zchut](https://www.kolzchut.org.il) articles. It powers a Telegram bot that runs locally on a laptop CPU.

Code, data pipeline and evaluation: [github.com/AhmadTawil1/bureaucracy-navigator](https://github.com/AhmadTawil1/bureaucracy-navigator)

> **General information, not legal advice.** The model is meant to be used with retrieved sources (RAG) and links to them in every answer.

## What it does

Given a system prompt and numbered context chunks, the model:

1. **Answers questions** in short, simple Hebrew, citing chunks as `[1]`, `[2]`… at the end of each sentence.
2. **Refuses** with a fixed sentence ("לא מצאתי מידע אמין על זה במקורות שלי.") when the chunks don't answer the question.
3. **Summarizes official letters** under four headings: מה כתוב במכתב / מה צריך לעשות / עד מתי / למי לפנות.
4. **Rewrites** bureaucratic Hebrew into short, plain sentences.

Always in plain text (no markdown). The exact prompt format is in [`app/prompts.py`](https://github.com/AhmadTawil1/bureaucracy-navigator/blob/master/app/prompts.py); use it as-is, since the model was trained on it.

## Training data

1,128 training examples (125 validation), generated from 287 Kol Zchut articles (unemployment, income support, income tax, dismissal) with `dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-W4A16` as the teacher:

| Task | Train examples |
|---|---|
| Grounded QA with citations | 509 |
| Unanswerable → refusal (teacher-verified hard negatives) | 92 |
| Letter summaries (synthetic letters, fake personal details) | 215 |
| Plain-Hebrew rewrites | 312 |

Filtered for valid citations, format, length, Hebrew ratio and duplicates, reviewed by hand, and **split by article**: 15% of the articles are held out for testing, and no test-article text appears in training inputs. Kol Zchut content is licensed [CC BY-NC-SA 2.5 IL](https://creativecommons.org/licenses/by-nc-sa/2.5/il/); these adapters are trained on it and shared under the same non-commercial terms.

## Training setup

LoRA (16-bit base, not QLoRA), r=16, alpha=32, dropout 0, on all attention and MLP projections; loss on assistant answers only; 2 epochs, effective batch 16, lr 2e-4 cosine, max length 4,096. Unsloth + TRL on a Colab L4, about 20 minutes. Train loss 0.383, best validation loss 0.391 (step 140 of 142).

## Results

232 held-out test examples, same retrieved chunks for every model:

| Metric | Base 1.7B | This model | Teacher 12B |
|---|---|---|---|
| Citation validity (QA) | 81% | 84% | 100% |
| Correct refusals (unanswerable) | 0% | 61% | 0% |
| Wrong refusals (answerable QA) | 0% | 12% | 0% |
| Letter: all 4 headings | 28% | 100% | 100% |
| Plain text (no markdown) | 49% | 100% | 84% |
| Correct (manual, 30 QA)* | 8/30 | 16/30 | — |
| Grounded (manual, 30 QA)* | 15/30 | 26/30 | — |
| Seconds per answer (laptop CPU, GGUF q8_0) | 51.1 | 32.6 | — |

\* Scored with an AI assistant, model hidden but recognizable by style; not a human expert evaluation.

## Limitations

- Over-refusal: refuses 12% of answerable test questions.
- Trained on teacher-generated data, which can carry the teacher's mistakes.
- Covers the topics of the ingested articles; answers are only as current as the retrieved sources.
- Occasionally mixes grammatical gender or puts the wrong date under "עד מתי" in letter summaries.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "dicta-il/DictaLM-3.0-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), "AhmadTawil1/dictalm3-bureaucracy-lora")
# messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": qa_user(question, chunks)}]
```
