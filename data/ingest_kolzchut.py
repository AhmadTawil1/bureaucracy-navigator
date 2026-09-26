"""Kol Zchut ingestion: fetch → clean → chunk → embed → Qdrant + chunks.jsonl (§5)."""
import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

API = "https://www.kolzchut.org.il/w/he/api.php"
HEADERS = {"User-Agent": "BureaucracyNavigator/0.1 (student project; ahmadtawil.se@gmail.com)"}

CATEGORIES = ["אבטלה", "הבטחת הכנסה", "מס הכנסה", "פיטורים"]
# Practical articles only: skip court rulings, portals, glossary terms, laws, etc.
KEEP_TYPES = {"right", "proceeding", "service", "guide"}

RAW_DIR = Path(__file__).parent / "raw"
INDEX_FILE = RAW_DIR / "index.json"

# Page chrome, not article content.
REMOVE = [
    ".mw-editsection", ".toc-box", ".kz-preferred-source", "#chat-section", ".kolsherut-links-section",
    ".article-see-also", "#helpme-section", ".share-links", ".noprint", ".rs_skip",
    "button", "style", "script", "sup.reference", ".mw-references-wrap",
]
SKIP_SECTIONS = {"מקורות משפטיים ורשמיים"}  # only lists of law titles and links
INTRO = "תקציר"


def category_pages(client: httpx.Client, category: str) -> list[str]:
    """Titles of articles in a category whose ArticleType is in KEEP_TYPES."""
    titles, cont = [], {}
    while True:
        r = client.get(API, params={
            "action": "query", "generator": "categorymembers", "gcmtitle": f"קטגוריה:{category}",
            "gcmlimit": 500, "gcmnamespace": 0, "prop": "pageprops", "ppprop": "ArticleType",
            "format": "json", "formatversion": 2, **cont,
        })
        r.raise_for_status()
        time.sleep(1)  # max 1 request per second
        data = r.json()
        for page in data.get("query", {}).get("pages", []):
            if page.get("pageprops", {}).get("ArticleType") in KEEP_TYPES:
                titles.append(page["title"])
        if "continue" not in data:
            return titles
        cont = data["continue"]


def page_html(client: httpx.Client, title: str) -> str:
    r = client.get(API, params={
        "action": "parse", "page": title, "prop": "text",
        "format": "json", "formatversion": 2, "redirects": 1,
    })
    r.raise_for_status()
    time.sleep(1)  # max 1 request per second
    return r.json()["parse"]["text"]


def download() -> None:
    """Save each article to data/raw/<index>.html; index.json maps index → title."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {}
    saved = set(index.values())

    with httpx.Client(headers=HEADERS, timeout=60) as client:
        titles: list[str] = []
        for category in CATEGORIES:
            found = category_pages(client, category)
            print(f"{category}: {len(found)} articles")
            titles += [t for t in found if t not in titles]
        print(f"unique: {len(titles)}, already saved: {len(saved & set(titles))}")

        for title in titles:
            if title in saved:
                continue
            i = str(len(index))
            (RAW_DIR / f"{i}.html").write_text(page_html(client, title), encoding="utf-8")
            index[i] = title
            INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
            if len(index) % 25 == 0:
                print(f"  {len(index)} saved")

    print(f"done: {len(index)} articles in {RAW_DIR}")


def _flat(text: str) -> str:
    return re.sub(r" ([.,:;)])", r"\1", re.sub(r"\s+", " ", text)).strip()


def _text(el: Tag) -> str:
    """Inline text without nested lists/tables. No separator, so ל<a>תנאים</a> stays 'לתנאים'."""
    return _flat("".join(c.get_text() if isinstance(c, Tag) else str(c)
                         for c in el.children if not (isinstance(c, Tag) and c.name in {"ul", "ol", "table"})))


def _lines(el: Tag) -> list[str]:
    """Readable lines for one content block."""
    if el.name in {"ul", "ol"}:
        lines = []
        for li in el.find_all("li", recursive=False):
            lines.append(f"- {_text(li)}")
            for sub in li.find_all(["ul", "ol"], recursive=False):
                lines += ["  " + line for line in _lines(sub)]
        return lines
    if el.name == "table":
        return [" | ".join(_text(c) for c in tr.find_all(["th", "td"])) for tr in el.find_all("tr")]
    if el.name == "details":  # FAQ item
        summary = el.find("summary")
        question = _text(summary) if summary else ""
        if summary:
            summary.extract()
        return [f"שאלה: {question}", f"תשובה: {_flat(el.get_text())}"]
    if "wr-note" in el.get("class", []):  # "שימו לב" / "טיפ" / "לדוגמה" box, label set in clean()
        return [_flat(el.get_text())]
    if el.name == "div" and el.find(["p", "ul", "ol", "table"], recursive=False):
        return [line for child in el.find_all(True, recursive=False) for line in _lines(child)]
    return [_text(el)]


def clean(html: str) -> list[tuple[str, str]]:
    """Article HTML → [(section_title, text)] in page order."""
    root = BeautifulSoup(html, "html.parser").select_one(".mw-parser-output")
    for selector in REMOVE:
        for el in root.select(selector):
            el.decompose()
    for header in root.select(".wr-note .header-text"):
        header.string = f"{_text(header)}: "
    for el in root.find_all(["p", "div", "li", "td", "th", "summary", "br"]):
        el.append(" ")  # keep separate blocks from gluing words together

    # Intro box: keep the summary item, drop the navigation items ("ראו בהמשך", "לחצו כאן").
    intro_lines = []
    if intro := root.select_one(".article-intro"):
        intro_lines = [_text(it) for it in intro.select(".emphasis-type-info .emphasis-item-text")]
        intro.decompose()

    sections: list[tuple[str, list[str]]] = [(INTRO, intro_lines)]
    h2 = ""
    for el in root.find_all(True, recursive=False):
        if el.name in {"h2", "h3"}:
            title = _text(el)
            if el.name == "h2":
                h2 = title
            sections.append((title if el.name == "h2" else f"{h2} / {title}", []))
        elif h2 not in SKIP_SECTIONS:
            sections[-1][1].extend(line for line in _lines(el) if line.strip("- "))
    return [(title, "\n".join(lines)) for title, lines in sections if lines and title.split(" / ")[0] not in SKIP_SECTIONS]


if __name__ == "__main__":
    download()
