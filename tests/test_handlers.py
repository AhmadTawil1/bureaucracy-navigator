from app.handlers import DISCLAIMER, SEARCH_URL, refusal, with_sources

CHUNKS = [
    {"title": "דמי אבטלה", "url": "https://www.kolzchut.org.il/he/דמי_אבטלה"},
    {"title": "הגשת תביעה לדמי אבטלה", "url": "https://www.kolzchut.org.il/he/הגשת_תביעה_לדמי_אבטלה"},
]


def test_sources_list_only_cited_chunks():
    out = with_sources("נרשמים בשירות התעסוקה [2].", CHUNKS)
    assert "[2] הגשת תביעה לדמי אבטלה — https://www.kolzchut.org.il/he/הגשת_תביעה_לדמי_אבטלה" in out
    assert "[1] דמי אבטלה" not in out
    assert out.endswith(DISCLAIMER)


def test_no_valid_citation_lists_all_articles():
    out = with_sources("תשובה בלי ציטוט [7].", CHUNKS)
    assert "דמי אבטלה — https://www.kolzchut.org.il/he/דמי_אבטלה" in out
    assert "הגשת תביעה לדמי אבטלה — https://www.kolzchut.org.il/he/הגשת_תביעה_לדמי_אבטלה" in out
    assert out.endswith(DISCLAIMER)


def test_same_article_listed_once():
    out = with_sources("תשובה.", [CHUNKS[0], CHUNKS[0]])
    assert out.count("דמי_אבטלה") == 1


def test_refusal_has_search_link():
    assert SEARCH_URL in refusal("מה מזג האוויר מחר?")
