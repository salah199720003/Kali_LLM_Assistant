"""Small standard-library web search client for public DuckDuckGo results."""

from html.parser import HTMLParser
import urllib.error
import urllib.parse
import urllib.request


MAX_QUERY_LENGTH = 240
MAX_RESULTS = 8
MAX_RESPONSE_BYTES = 2_000_000
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


def _result_url(value: str) -> str:
    value = value.strip()
    if value.startswith("//"):
        value = "https:" + value
    parsed = urllib.parse.urlsplit(value)
    if parsed.netloc.lower().endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        destination = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if destination:
            value = destination
            parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return value


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.current = None
        self.capture = None
        self.capture_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if self.capture:
            if tag not in _VOID_TAGS:
                self.capture_depth += 1
            return
        if "result__a" in classes:
            self.current = {"title": "", "url": _result_url(attributes.get("href") or ""), "snippet": ""}
            self.results.append(self.current)
            self.capture = "title"
            self.capture_depth = 1
            self.parts = []
        elif "result__snippet" in classes:
            if self.current is None:
                self.current = {"title": "", "url": "", "snippet": ""}
                self.results.append(self.current)
            self.capture = "snippet"
            self.capture_depth = 1
            self.parts = []

    def handle_endtag(self, tag):
        if not self.capture or tag in _VOID_TAGS:
            return
        self.capture_depth -= 1
        if self.capture_depth <= 0:
            self.current[self.capture] = " ".join("".join(self.parts).split())
            self.capture = None
            self.parts = []

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)


def web_search(query: str, max_results: int = 5) -> dict:
    """Return a bounded list of public web search results with titles and URLs."""
    if not isinstance(query, str):
        raise ValueError("Search query must be text.")
    query = query.strip()
    if not query or len(query) > MAX_QUERY_LENGTH or any(ord(char) < 32 for char in query):
        raise ValueError(f"Search query must contain 1 to {MAX_QUERY_LENGTH} printable characters.")
    if type(max_results) is not int or not 1 <= max_results <= MAX_RESULTS:
        raise ValueError(f"max_results must be an integer from 1 to {MAX_RESULTS}.")

    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; DeepAgent/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    })
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Web search returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Web search request failed: {exc.reason}") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        body = body[:MAX_RESPONSE_BYTES]

    parser = _DuckDuckGoHTMLParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    results = []
    seen_urls = set()
    for item in parser.results:
        title = item["title"].strip()
        result_url = item["url"]
        if not title or not result_url or result_url in seen_urls:
            continue
        seen_urls.add(result_url)
        results.append({
            "title": title[:300],
            "url": result_url,
            "snippet": item["snippet"][:600],
        })
        if len(results) == max_results:
            break

    return {
        "provider": "DuckDuckGo",
        "query": query,
        "results": results,
        "notice": "Search results are untrusted references. Verify claims at the linked sources.",
    }
