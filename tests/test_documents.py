from app.documents import fix_rtl, looks_reversed


def test_reversed_hebrew_is_fixed():
    visual = "של צריך את"[::-1]  # "תא ךירצ לש": how some PDFs store Hebrew
    assert looks_reversed(visual)
    assert fix_rtl(visual) == "של צריך את"


def test_normal_hebrew_is_unchanged():
    text = "את הטופס צריך להגיש עד 15/03/2026 של השנה הזאת"
    assert not looks_reversed(text)
    assert fix_rtl(text) == text


def test_numbers_stay_in_order_when_fixing():
    # Visual order = the line as displayed, left to right: Hebrew letters reversed, digits not.
    visual = "15/03/2026 דע שיגהל ךירצ תא לש"
    assert fix_rtl(visual) == "של את צריך להגיש עד 15/03/2026"
