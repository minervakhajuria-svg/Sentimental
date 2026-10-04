"""Minimal JSON-over-HTTP helper shared by API collectors.

Retries rate limits and server errors with exponential backoff; anything
else (bad key, 404) raises straight away, since retrying won't help.
Secrets go in headers, never in URLs, so they can't leak into logs.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

log = logging.getLogger(__name__)

USER_AGENT = "sentimental/0.1 (personal research)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


def _urlopen_json(url: str, headers: dict[str, str], timeout: float) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, url.split("?")[0]) from None


def get_json(
    url: str,
    params: dict | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 5,
    timeout: float = 30,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[[str, dict, float], object] = _urlopen_json,
) -> object:
    full = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    for attempt in range(max_retries + 1):
        try:
            return opener(full, headers or {}, timeout)
        except HttpError as e:
            if e.status not in RETRY_STATUS or attempt == max_retries:
                raise
            wait = backoff_seconds * 2**attempt
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == max_retries:
                raise
            wait = backoff_seconds * 2**attempt
            log.debug("network error %s", type(e).__name__)
        log.warning("%s: retrying in %ss (attempt %d)", url, wait, attempt + 1)
        sleep(wait)
