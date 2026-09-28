"""Shared SerpApi plumbing: retries, caching and a circuit breaker.

The shopping engine costs nothing extra to ask for but on restricted plans it
can hang instead of answering. A circuit breaker keeps that cost to one short
attempt instead of one per outfit category.
"""

import os
import time

import httpx

from fashion_agent.product_search.sources import ResponseCache

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

MEMO_LIMIT = 64


class SerpApiError(Exception):
    """SerpApi answered with an error we should not retry."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class SerpApiClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache: ResponseCache | None = None,
        timeout: float = 60.0,
        attempts: int = 3,
        backoff: float = 2.0,
        sleeper=time.sleep,
    ):
        # An explicit empty string means "no key", not "look in the environment".
        self.api_key = os.getenv("SERPAPI_API_KEY") if api_key is None else api_key
        self.cache = cache
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self.backoff = backoff
        self.sleeper = sleeper
        self._memo: dict[str, tuple[str, dict]] = {}

    def has_key(self) -> bool:
        return bool(self.api_key)

    def get(
        self,
        cache_key: str,
        params: dict,
    ) -> tuple[dict, int]:
        """Return the API payload, reusing an in-process or on-disk copy."""
        memoized = self._memo.get(cache_key)

        if memoized is not None:
            return memoized[1], 0

        cached = self.cache.get(cache_key) if self.cache else None

        if cached is not None:
            self._remember(cache_key, ("", cached))

            return cached, 0

        payload, attempts = self._request(params)

        if self.cache is not None:
            self.cache.set(cache_key, payload)

        self._remember(cache_key, ("", payload))

        return payload, attempts

    def _request(self, params: dict) -> tuple[dict, int]:
        if not self.api_key:
            raise SerpApiError("SERPAPI_API_KEY is not configured")

        payload = {**params, "api_key": self.api_key}
        last_error: Exception | None = None

        for attempt in range(1, self.attempts + 1):
            try:
                response = httpx.get(
                    SERPAPI_ENDPOINT,
                    params=payload,
                    timeout=self.timeout,
                    follow_redirects=True,
                )
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = error

                if attempt < self.attempts:
                    self.sleeper(self.backoff * attempt)
                    continue

                raise SerpApiError(
                    f"transport error: {type(error).__name__}",
                    retryable=True,
                ) from error

            if response.status_code in RETRYABLE_STATUS:
                last_error = SerpApiError(
                    f"HTTP {response.status_code}",
                    retryable=True,
                )

                if attempt < self.attempts:
                    self.sleeper(self.backoff * attempt)
                    continue

                raise last_error

            try:
                data = response.json()
            except ValueError as error:
                raise SerpApiError("response was not JSON") from error

            error_text = data.get("error")

            if error_text:
                # Quota errors are not worth retrying, they need time, not tries.
                raise SerpApiError(
                    str(error_text),
                    retryable="quota" in str(error_text).lower()
                    or "limit" in str(error_text).lower(),
                )

            return data, attempt

        raise SerpApiError(f"exhausted retries: {last_error}", retryable=True)

    def _remember(self, key: str, value: tuple[str, dict]) -> None:
        if len(self._memo) >= MEMO_LIMIT:
            self._memo.pop(next(iter(self._memo)))

        self._memo[key] = value
