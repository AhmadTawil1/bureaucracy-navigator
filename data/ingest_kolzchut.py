"""Kol Zchut ingestion: fetch → clean → chunk → embed → Qdrant + chunks.jsonl (§5)."""
import json
import time
from pathlib import Path

import httpx

API = "https://www.kolzchut.org.il/w/he/api.php"
HEADERS = {"User-Agent": "BureaucracyNavigator/0.1 (student project; ahmadtawil.se@gmail.com)"}

CATEGORIES = ["אבטלה", "הבטחת הכנסה", "מס הכנסה", "פיטורים"]
# Practical articles only: skip court rulings, portals, glossary terms, laws, etc.
KEEP_TYPES = {"right", "proceeding", "service", "guide"}

RAW_DIR = Path(__file__).parent / "raw"
INDEX_FILE = RAW_DIR / "index.json"


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


if __name__ == "__main__":
    download()
