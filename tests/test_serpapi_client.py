import httpx
import pytest

from fashion_agent.product_search.serpapi_client import SerpApiClient
from fashion_agent.product_search.sources import ResponseCache

PAYLOAD = {"organic_results": [{"link": "u", "title": "t", "snippet": "s"}]}


@pytest.fixture
def api(monkeypatch):
    """Never let a unit test reach the network."""
    calls: list[dict] = []

    def fake_get(url, params=None, **rest):
        calls.append(dict(params or {}))
        return httpx.Response(200, json=PAYLOAD)

    monkeypatch.setattr(httpx, "get", fake_get)

    return calls


def test_cache_round_trip(tmp_path, api):
    cache = ResponseCache(tmp_path / "cache.json")

    client = SerpApiClient(api_key="k", cache=cache)
    client.get("key", {"engine": "google"})

    other = SerpApiClient(api_key="k", cache=cache)
    payload, attempts = other.get("key", {"engine": "google"})

    assert payload == PAYLOAD
    assert attempts == 0


def test_stored_shape_is_the_payload_itself(tmp_path, api):
    import json

    path = tmp_path / "cache.json"
    cache = ResponseCache(path)

    SerpApiClient(api_key="k", cache=cache).get("key", {"engine": "google"})

    stored = json.loads(path.read_text(encoding="utf-8"))["key"]["payload"]

    assert stored == PAYLOAD
    assert "payload" not in stored


def test_memo_answers_before_the_cache(tmp_path, api):
    calls: list[int] = []

    client = SerpApiClient(api_key="k", cache=ResponseCache(tmp_path / "c.json"))

    def fake_request(params):
        calls.append(1)
        return PAYLOAD, 1

    client._request = fake_request  # type: ignore[method-assign]

    client.get("key", {"engine": "google"})
    client.get("key", {"engine": "google"})

    assert len(calls) == 1


def test_empty_api_key_is_not_taken_from_the_environment(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "from-env")

    assert SerpApiClient(api_key="").has_key() is False
    assert SerpApiClient(api_key=None).has_key() is True
    assert SerpApiClient().has_key() is True


def test_a_failed_request_is_not_cached(tmp_path, monkeypatch):
    from fashion_agent.product_search.serpapi_client import SerpApiError

    def boom(url, params=None, **rest):
        raise httpx.ReadTimeout("x")

    monkeypatch.setattr(httpx, "get", boom)

    client = SerpApiClient(
        api_key="k",
        attempts=1,
        cache=ResponseCache(tmp_path / "cache.json"),
        sleeper=lambda _s: None,
    )

    with pytest.raises(SerpApiError):
        client.get("key", {})

    assert client.cache.get("key") is None
