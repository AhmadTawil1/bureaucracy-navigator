"""Mask personal details in document text before it is stored or sent to the model (§8.3)."""
import re


def valid_il_id(digits: str) -> bool:
    """Israeli ID check digit (Luhn-like), so not every 9-digit number is treated as an ID."""
    total = 0
    for i, ch in enumerate(digits.zfill(9)):
        n = int(ch) * (1 if i % 2 == 0 else 2)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def mask(text: str) -> str:
    # "Not next to another digit" instead of \b: PDFs often glue numbers to Hebrew ("ת.ז000000018"),
    # and there is no word boundary between a Hebrew letter and a digit.
    text = re.sub(r"(?<!\d)\d{9}(?!\d)", lambda m: "[ת״ז]" if valid_il_id(m.group()) else m.group(), text)
    text = re.sub(r"(?<!\d)0\d{1,2}-?\d{7}(?!\d)", "[טלפון]", text)
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[אימייל]", text)
