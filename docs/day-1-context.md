# Day 1 context

## Setup (Before Day 1)

- Tools: uv 0.11.3, Python 3.13.12 (project `.venv`), Git 2.45.2.
- Colab Pro with an A100 runtime.
- Hugging Face write token stored as Colab secret `HF_TOKEN` (read with `userdata.get("HF_TOKEN")`); not stored anywhere in the repo.
- Google Drive folder `/content/drive/MyDrive/bureaucracy-navigator` (mounted in Colab with `drive.mount("/content/drive")`).
- Telegram on the user's phone.
- Tesseract 5.5.0 at `C:\Program Files\Tesseract-OCR` (added to user `PATH`), languages `eng`, `heb`, `osd`. `heb.traineddata` is from `tessdata_best` (more accurate, slower).
- llama.cpp build b11193 via `winget install ggml.llamacpp` (winget ships the **Vulkan** build). `llama-server --list-devices` shows `Vulkan0: Intel Iris Xe`. For CPU-only runs (the plan's speed metric), pass `-ngl 0`.

## Deviations from the plan

- **Native Windows, not WSL2.** The guide says to work inside WSL2 (Ubuntu). We stay on Windows: vLLM and Unsloth run on Colab anyway, and everything local (FastAPI, Qdrant local mode, sentence-transformers, PyMuPDF, Tesseract, `llama-server`) has Windows builds. Commands like `sudo apt install ...` are replaced by Windows installers.
