"""PDF + photo → text, with Hebrew order fix and OCR (§8.2)."""
import io
import re
import shutil
from pathlib import Path

import pymupdf  # the old name "fitz" is deprecated
import pytesseract
from bidi import get_display
from PIL import Image, ImageOps

# Windows installs Tesseract outside PATH by default.
WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if not shutil.which("tesseract") and WINDOWS_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT)

MAX_PAGES = 10
COMMON = {"של", "את", "על", "לא", "או", "כי", "אם", "עם", "זה"}


def looks_reversed(text: str) -> bool:
    """Some PDFs store Hebrew in visual order ("לש" instead of "של"): count common words both ways."""
    words = re.findall(r"[\u05d0-\u05ea]{2,}", text)
    return sum(w[::-1] in COMMON for w in words) > sum(w in COMMON for w in words)


def fix_rtl(text: str) -> str:
    # get_display restores reading order for Hebrew runs and keeps numbers and Latin text intact.
    if not looks_reversed(text):
        return text
    return "\n".join(get_display(line, base_dir="R") for line in text.splitlines())


def ocr(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("L")  # phone photos: apply rotation, grayscale
    return pytesseract.image_to_string(image, lang="heb+eng")


def image_to_text(data: bytes) -> str:
    return fix_rtl(ocr(Image.open(io.BytesIO(data))))


def pdf_pages(data: bytes) -> int:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def pdf_to_text(data: bytes) -> str:
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text()
            if len(text.strip()) < 50:  # scanned page: no text layer
                pix = page.get_pixmap(dpi=300)
                text = ocr(Image.open(io.BytesIO(pix.tobytes("png"))))
            pages.append(fix_rtl(text))
    return separate_numbers("\n".join(pages))


def separate_numbers(text: str) -> str:
    """Word's PDFs glue numbers to Hebrew ("ליום15/02/2026"): put a space between them."""
    text = re.sub(r"([א-ת])(\d)", r"\1 \2", text)
    return re.sub(r"(\d)([א-ת])", r"\1 \2", text)
