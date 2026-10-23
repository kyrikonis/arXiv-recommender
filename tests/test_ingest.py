import re
import time
from pathlib import Path

import httpx
import pytest

from recommender.ingest import (
    MAX_ATTEMPTS,
    Arxiv,
    arxiv_id,
    parse_api_feed,
    parse_rss,
    search,
)

# Real responses saved from the arXiv API and RSS feed, trimmed to a few entries.
FIXTURES = Path(__file__).parent / "fixtures"
API_RANGE = (FIXTURES / "api_range.xml").read_bytes()
API_IDS = (FIXTURES / "api_ids.xml").read_bytes()
API_ERROR = (FIXTURES / "api_error.xml").read_bytes()
RSS = (FIXTURES / "rss_gr-qc.xml").read_bytes()

# time.monotonic() on Windows only ticks roughly every 16 ms
CLOCK_TOLERANCE = 0.05

def with_total(xml: bytes, total: int, keep_entries: bool = True) -> bytes:
    xml = re.sub(rb"<opensearch:totalResults>\d+", b"<opensearch:totalResults>%d" % total, xml)
    if not keep_entries:
        xml = re.sub(rb"<entry>.*?</entry>", b"", xml, flags=re.S)
    return xml

def client_returning(responses: list) -> tuple[Arxiv, list[httpx.Request]]:
    """An Arxiv client whose transport replays `responses` in order and records requests."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status, body = responses[len(requests) - 1]
        return httpx.Response(status, content=body)

    return Arxiv(delay=0, cooldown=0, transport=httpx.MockTransport(handler)), requests

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://arxiv.org/abs/2610.00844v2", "2610.00844"),
        ("http://arxiv.org/abs/hep-th/9711200v3", "hep-th/9711200"),
        ("oai:arXiv.org:2610.30187v1", "2610.30187"),
        ("oai:arXiv.org:2502.08748v14", "2502.08748"),
        ("2610.00844", "2610.00844"),
    ],
)
def test_arxiv_id(raw, expected):
    assert arxiv_id(raw) == expected

def test_parse_api_feed():
    papers, total = parse_api_feed(API_RANGE)
    assert total == 27
    assert [p.id for p in papers] == ["2610.00844", "2610.00989", "2610.03798"]

    cross_listed = papers[0]
    assert cross_listed.primary_category == "hep-th"
    assert cross_listed.categories == "hep-th gr-qc quant-ph"
    assert cross_listed.published == "2026-10-01T00:04:04Z"
    assert papers[1].authors == "Zi-An Zhou, Cheng-Jun Fang, Zong-Kuan Guo"

def test_parse_api_feed_old_style_id_and_whitespace():
    papers, _ = parse_api_feed(API_IDS)
    assert [p.id for p in papers] == ["hep-th/9711200", "1706.03762"]
    # The 1997 abstract starts with two spaces in the raw XML.
    assert papers[0].abstract.startswith("We show that the large $N$ limit")
    assert papers[0].published == "1997-11-27T23:53:13Z"

def test_parse_api_feed_raises_on_api_error():
    with pytest.raises(ValueError, match="incorrect id format"):
        parse_api_feed(API_ERROR)

def test_parse_rss_keeps_new_and_cross_only():
    day, ids = parse_rss(RSS)
    assert day == "2026-10-23"
    # The fixture also contains a replace and a replace-cross item, which must be dropped.
    assert ids == ["2610.30187", "2610.30201", "2610.26652"]

def test_get_retries_server_errors():
    client, requests = client_returning([(503, b""), (502, b""), (200, b"ok")])
    assert client.get("https://export.arxiv.org/api/query") == b"ok"
    assert len(requests) == 3

def test_get_backs_off_longer_when_rate_limited():
    client, requests = client_returning([(429, b""), (200, b"ok")])
    client.cooldown = 0.3
    begin = time.monotonic()
    assert client.get("https://export.arxiv.org/api/query") == b"ok"
    assert time.monotonic() - begin >= 0.3 - CLOCK_TOLERANCE
    assert len(requests) == 2

def test_get_gives_up_after_max_attempts():
    client, requests = client_returning([(503, b"")] * MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match="giving up"):
        client.get("https://export.arxiv.org/api/query")
    assert len(requests) == MAX_ATTEMPTS

def test_get_does_not_retry_client_errors():
    client, requests = client_returning([(400, b"")])
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://export.arxiv.org/api/query")
    assert len(requests) == 1

def test_get_waits_between_requests():
    client, _ = client_returning([(200, b"")] * 3)
    client.delay = 0.2
    begin = time.monotonic()
    for _ in range(3):
        client.get("https://export.arxiv.org/api/query")
    # The first request goes immediately, the next two each wait out the delay.
    assert time.monotonic() - begin >= 0.4 - CLOCK_TOLERANCE

def test_search_paginates():
    client, requests = client_returning(
        [(200, with_total(API_RANGE, 5)), (200, with_total(API_IDS, 5))]
    )
    papers = search(client, "cat:gr-qc")
    assert len(papers) == 5
    assert [r.url.params["start"] for r in requests] == ["0", "3"]

def test_search_retries_empty_page():
    client, requests = client_returning(
        [(200, with_total(API_RANGE, 3, keep_entries=False)), (200, with_total(API_RANGE, 3))]
    )
    assert len(search(client, "cat:gr-qc")) == 3
    assert len(requests) == 2

def test_search_with_no_results_makes_one_request():
    client, requests = client_returning([(200, with_total(API_RANGE, 0, keep_entries=False))])
    assert search(client, "cat:gr-qc") == []
    assert len(requests) == 1