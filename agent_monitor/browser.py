"""
agent_monitor.browser — Playwright-controlled Chromium for agent tools.

Why this exists:
    Real agent platforms (Claude Code, Codex, Open Interpreter) drive a
    browser to read pages, fill forms, scrape content. Playwright is the
    industry-standard wrapper around Chromium / Firefox / WebKit.

Design:
    A single BrowserSession owns one Chromium process and one page.
    Synchronous API (Playwright's sync_api flavor) -> easier to call from
    REST endpoints without async ceremony. The whole server can run on a
    threadpool; Playwright handles its own internal asyncio.

Lifecycle:
    session = BrowserSession.start(headless=True)
    session.goto("https://example.com")
    title = session.title()
    png   = session.screenshot()       # bytes
    text  = session.text_content()     # readable inner text
    session.close()

Screenshots are returned as PNG bytes; encode to base64 in the API layer.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from playwright.sync_api import sync_playwright


@dataclass
class BrowserSession:
    headless: bool = True
    _pw: object = field(default=None, repr=False)
    _browser: object = field(default=None, repr=False)
    _page: object = field(default=None, repr=False)
    last_url: Optional[str] = None
    last_title: Optional[str] = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    def start(cls, *, headless: bool = True) -> "BrowserSession":
        s = cls(headless=headless)
        s._pw = sync_playwright().start()
        s._browser = s._pw.chromium.launch(headless=headless)
        s._page = s._browser.new_page()
        return s

    # -- navigation -----------------------------------------------------

    def goto(self, url: str, *, timeout_ms: int = 15000) -> dict:
        with self._lock:
            self._page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            self.last_url = self._page.url
            self.last_title = self._page.title()
            return {"url": self.last_url, "title": self.last_title}

    def title(self) -> str:
        with self._lock:
            return self._page.title()

    def url(self) -> str:
        with self._lock:
            return self._page.url

    # -- read content ---------------------------------------------------

    def text_content(self, selector: str = "body", *, max_len: int = 8000) -> str:
        with self._lock:
            txt = self._page.locator(selector).first.inner_text()
        return (txt or "")[:max_len]

    def html(self, *, max_len: int = 20000) -> str:
        with self._lock:
            h = self._page.content()
        return (h or "")[:max_len]

    # -- write actions --------------------------------------------------

    def click(self, selector: str, *, timeout_ms: int = 5000) -> None:
        with self._lock:
            self._page.locator(selector).first.click(timeout=timeout_ms)

    def fill(self, selector: str, value: str, *, timeout_ms: int = 5000) -> None:
        with self._lock:
            self._page.locator(selector).first.fill(value, timeout=timeout_ms)

    # -- visuals --------------------------------------------------------

    def screenshot(self, *, full_page: bool = False) -> bytes:
        with self._lock:
            return self._page.screenshot(full_page=full_page)

    def screenshot_b64(self, *, full_page: bool = False) -> str:
        return base64.b64encode(self.screenshot(full_page=full_page)).decode("ascii")

    # -- shutdown -------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._browser.close()
            finally:
                self._pw.stop()
                self._browser = None
                self._page = None
                self._pw = None

    @property
    def is_open(self) -> bool:
        return self._page is not None


# ---------------------------------------------------------------------------
# Process-wide singleton -- the API layer reuses one browser session.
# ---------------------------------------------------------------------------

_SESSION: Optional[BrowserSession] = None
_LOCK = Lock()


def get_or_start(*, headless: bool = True) -> BrowserSession:
    global _SESSION
    with _LOCK:
        if _SESSION is None or not _SESSION.is_open:
            _SESSION = BrowserSession.start(headless=headless)
        return _SESSION


def shutdown() -> None:
    global _SESSION
    with _LOCK:
        if _SESSION is not None and _SESSION.is_open:
            _SESSION.close()
        _SESSION = None
