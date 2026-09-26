"""Kol Zchut ingestion: fetch → clean → chunk → embed → Qdrant + chunks.jsonl (§5)."""
import json
import re
import time
from pathlib import Path

import httpx
import numpy as np
from bs4 import BeautifulSoup, Tag
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

API = "https://www.kolzchut.org.il/w/he/api.php"
HEADERS = {"User-Agent": "BureaucracyNavigator/0.1 (student project; ahmadtawil.se@gmail.com)"}

CATEGORIES = ["אבטלה", "הבטחת הכנסה", "מס הכנסה", "פיטורים"]
# Practical articles only: skip court rulings, portals, glossary terms, laws, etc.
KEEP_TYPES = {"right", "proceeding", "service", "guide"}

RAW_DIR = Path(__file__).parent / "raw"
INDEX_FILE = RAW_DIR / "index.json"
CHUNKS_FILE = Path(__file__).parent / "out" / "chunks.jsonl"
EMBEDDINGS_FILE = Path(__file__).parent / "out" / "embeddings.npy"  # made by data/embed.py on Colab
QDRANT_PATH = Path(__file__).parent.parent / "qdrant_data"
COLLECTION = "kolzchut"
ARTICLE_URL = "https://www.kolzchut.org.il/he/"

EMBED_MODEL = "intfloat/multilingual-e5-large"
# Sizes in e5 tokens (limit 512); 384 leaves room for the "passage: title — section" prefix.
CHUNK_TOKENS = 384
CHUNK_OVERLAP = 64

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
    text = re.sub(r"[​‎‏­]", "", text)  # invisible zero-width / direction marks
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


def chunk() -> list[dict]:
    """Clean every saved article, pack its sections into token-sized chunks, write chunks.jsonl.

    Small neighbouring sections of one article are packed together (each keeps a "## heading"
    line); a section too big for one chunk is split with a 64-token overlap.
    """
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    n_tokens = lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"])
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_TOKENS - 32,  # room for the "## heading" line
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # prefer whole lines, then words
    )
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    chunks = []
    for i, title in sorted(index.items(), key=lambda kv: int(kv[0])):
        url = ARTICLE_URL + title.replace(" ", "_")
        # One block per section, or several if the section is too big.
        blocks = []
        for section, text in clean((RAW_DIR / f"{i}.html").read_text(encoding="utf-8")):
            pieces = [text] if n_tokens(f"## {section}\n{text}") <= CHUNK_TOKENS else splitter.split_text(text)
            blocks += [(section, f"## {section}\n{piece}") for piece in pieces]

        # Greedy packing: add blocks while the chunk stays within CHUNK_TOKENS.
        packed: list[list[tuple[str, str]]] = []
        size = 0
        for block in blocks:
            t = n_tokens(block[1])
            if packed and size + t <= CHUNK_TOKENS:
                packed[-1].append(block)
                size += t
            else:
                packed.append([block])
                size = t

        for n, group in enumerate(packed):
            sections = list(dict.fromkeys(s for s, _ in group))  # unique, in order
            chunks.append({"id": f"{title}#{n}", "title": title, "section": " | ".join(sections),
                           "url": url, "text": "\n\n".join(b for _, b in group)})

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks from {len(index)} articles → {CHUNKS_FILE}")
    return chunks


def store() -> None:
    """Load chunks.jsonl + embeddings.npy into a local Qdrant collection (row i = chunk i)."""
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()]
    vectors = np.load(EMBEDDINGS_FILE)
    assert len(vectors) == len(chunks), f"{len(vectors)} vectors for {len(chunks)} chunks: re-run data/embed.py"

    qdrant = QdrantClient(path=str(QDRANT_PATH))
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
    )
    qdrant.upsert(COLLECTION, points=[
        models.PointStruct(id=i, vector=v.tolist(), payload=c) for i, (v, c) in enumerate(zip(vectors, chunks))
    ])
    print(f"{qdrant.count(COLLECTION).count} points in '{COLLECTION}' at {QDRANT_PATH}")
    qdrant.close()


if __name__ == "__main__":
    # Embedding runs on Colab in between: chunk() → data/embed.py → store().
    download()
    chunk()
    if EMBEDDINGS_FILE.exists():
        store()
