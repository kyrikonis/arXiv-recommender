import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from email.utils import parsedate_to_datetime
import httpx
from recommender.db import Paper

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
RSS_URL = "https://rss.arxiv.org/rss/"
PAGE_SIZE = 2000 # the API's maximum per request
IDS_PER_REQUEST = 200 #keeps id_list URLs a sane length
DELAY = 3.0 # arXiv terms: at most one request every three seconds
# arXiv sends 429 without Retry-After, and in practice it lasted over a minute, far longer
# than a server hiccup, so rate limiting gets its own much longer backoff.
COOLDOWN = 60.0
MAX_ATTEMPTS = 4
RETRY_STATUS = {429, 500, 502, 503, 504}
KEEP_ANNOUNCE_TYPES = {"new", "cross"} # replacements are papers we've already seen

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

class Arxiv:
    """HTTP client that makes one request at a time, at least `delay` seconds apart."""

    def __init__(
        self,
        delay: float = DELAY,
        cooldown: float = COOLDOWN,
        transport: httpx.BaseTransport | None = None,
    ):
        contact = os.environ.get("ARXIV_CONTACT")
        agent = "arxiv-recommender/0.1" + (f" (mailto:{contact})" if contact else "")
        self.http = httpx.Client(
            headers={"User-Agent": agent},
            timeout=60,
            follow_redirects=True,
            transport=transport,
        )
        self.delay = delay
        self.cooldown = cooldown
        self.last_request = float("-inf")

    def get(self, url: str, params: dict | None = None) -> bytes:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            wait = self.last_request + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()

            backoff = self.delay * 2**attempt
            try:
                resp = self.http.get(url, params=params)
            except httpx.TransportError as e:
                error = repr(e)
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code not in RETRY_STATUS:
                    resp.raise_for_status()
                error = f"HTTP {resp.status_code}"
                if resp.status_code == 429:
                    backoff = self.cooldown * 2 ** (attempt - 1)

            if attempt < MAX_ATTEMPTS:
                log.warning("%s on attempt %d, retrying in %.0fs", error, attempt, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"giving up on {url} after {MAX_ATTEMPTS} attempts: {error}")

def arxiv_id(raw: str) -> str:
    """'http://arxiv.org/abs/2610.00844v2' or 'oai:arXiv.org:hep-th/9711200v3' -> unversioned ID."""
    for prefix in ("/abs/", "oai:arXiv.org:"):
        if prefix in raw:
            raw = raw.split(prefix, 1)[1]
    return re.sub(r"v\d+$", "", raw)

def clean(text: str | None) -> str:
    # Titles and abstracts contain hard line breaks and runs of spaces from the submission.
    return " ".join((text or "").split())

def parse_api_feed(xml: bytes) -> tuple[list[Paper], int]:
    """Parse an API response into papers plus the total number of results for the query."""
    root = ET.fromstring(xml)
    total = int(root.findtext("opensearch:totalResults", namespaces=NS))
    papers = []
    for entry in root.iterfind("atom:entry", NS):
        entry_id = entry.findtext("atom:id", namespaces=NS)
        if "/api/errors" in entry_id:
            raise ValueError(f"arXiv API error: {entry.findtext('atom:summary', namespaces=NS)}")
        papers.append(
            Paper(
                id=arxiv_id(entry_id),
                title=clean(entry.findtext("atom:title", namespaces=NS)),
                abstract=clean(entry.findtext("atom:summary", namespaces=NS)),
                authors=", ".join(
                    clean(name.text) for name in entry.iterfind("atom:author/atom:name", NS)
                ),
                categories=" ".join(c.get("term") for c in entry.iterfind("atom:category", NS)),
                primary_category=entry.find("arxiv:primary_category", NS).get("term"),
                published=entry.findtext("atom:published", namespaces=NS),
            )
        )
    return papers, total

def parse_rss(xml: bytes) -> tuple[str, list[str]]:
    """Return the listing date and the IDs of new and cross-listed papers in an RSS feed."""
    channel = ET.fromstring(xml).find("channel")
    day = parsedate_to_datetime(channel.findtext("pubDate")).date().isoformat()
    ids = [
        arxiv_id(item.findtext("guid"))
        for item in channel.iterfind("item")
        if item.findtext("arxiv:announce_type", namespaces=NS) in KEEP_ANNOUNCE_TYPES
    ]
    return day, ids

def search(client: Arxiv, query: str) -> list[Paper]:
    papers: list[Paper] = []
    start, total = 0, 1
    while start < total:
        for _ in range(MAX_ATTEMPTS):
            params = {
                "search_query": query,
                "start": start,
                "max_results": PAGE_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            page, total = parse_api_feed(client.get(API_URL, params))
            if page or start >= total:
                break
            # The API occasionally returns an empty page part-way through a result set
            # asking again usually works.
            log.warning("empty page at %d of %d for %r, retrying", start, total, query)
        else:
            raise RuntimeError(f"repeated empty pages at {start} of {total} for {query!r}")
        papers.extend(page)
        start += len(page)
    return papers

def fetch_range(client: Arxiv, category: str, start: date, end: date) -> list[Paper]:
    """Papers in `category` (primary or cross-listed) first submitted in [start, end), UTC."""
    return search(
        client, f"cat:{category} AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}0000]"
    )

def fetch_by_ids(client: Arxiv, ids: list[str]) -> list[Paper]:
    papers = []
    for i in range(0, len(ids), IDS_PER_REQUEST):
        chunk = ids[i : i + IDS_PER_REQUEST]
        params = {"id_list": ",".join(chunk), "max_results": len(chunk)}
        page, _ = parse_api_feed(client.get(API_URL, params))
        papers.extend(page)
    return papers

def fetch_announced(client: Arxiv, categories: tuple[str, ...]) -> tuple[str | None, list[str]]:
    """The latest listing date and the IDs announced in it across all categories."""
    day, seen = None, {}
    for category in categories:
        feed_day, ids = parse_rss(client.get(RSS_URL + category))
        day = day or feed_day
        if feed_day != day:
            log.warning("%s feed is dated %s, others %s", category, feed_day, day)
        seen.update(dict.fromkeys(ids))
    return day, list(seen)
