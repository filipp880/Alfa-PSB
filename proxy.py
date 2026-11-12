"""Client for the single OpenAI-compatible gateway specified by the organizers."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from http.client import IncompleteRead
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from telemetry import span

LOG = logging.getLogger(__name__)


def env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def bounded(name: str, default: int, low: int, high: int) -> int:
    try:
        return min(high, max(low, int(env(name))))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    base_url: str = ""
    key: str = field(default="", repr=False)
    model: str = ""
    embed_model: str = ""
    rerank_model: str = ""
    embed_key: str = field(default="", repr=False)
    rerank_key: str = field(default="", repr=False)
    max_tokens: int = 1500
    temperature: float = 0.0
    workers: int = 6
    top_k: int = 18
    rerank_top_n: int = 10
    json_mode: bool = True
    thinking: str = "auto"
    retry_search: bool = True
    embed_cache: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        try:
            temperature = float(env("TEMPERATURE", default="0"))
            temperature = min(2.0, max(0.0, temperature)) if math.isfinite(temperature) else 0.0
        except ValueError:
            temperature = 0.0
        return cls(
            base_url=env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/"),
            key=env("LLM_API_KEY", "OPENAI_API_KEY"),
            model=env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL"),
            embed_model=env("EMB_MODEL", default="baai/bge-m3"),
            rerank_model=env("RERANKER_MODEL"),
            embed_key=env("EMBEDDER_API_KEY"),
            rerank_key=env("RERANKER_API_KEY"),
            max_tokens=bounded("MAX_TOKENS", 1500, 128, 8192),
            temperature=temperature,
            workers=bounded("MAX_WORKERS", 6, 1, 8),
            top_k=bounded("TOP_K", 18, 4, 32),
            rerank_top_n=bounded("RERANKER_TOP_N", 10, 2, 24),
            json_mode=env("LLM_JSON_MODE", default="1") != "0",
            thinking=env("LLM_THINKING", default="auto").casefold(),
            retry_search=env("RAG_RETRY_SEARCH", default="1") != "0",
            embed_cache=env("PSB_EMBED_CACHE_DIR"),
        )

    @property
    def ready(self) -> bool:
        if not (self.base_url and self.key and self.model):
            return False
        try:
            url = urlsplit(self.base_url)
            return bool(url.scheme in {"http", "https"} and url.hostname and not
                        (url.username or url.password or url.query or url.fragment))
        except ValueError:
            return False


class GatewayError(Exception):
    def __init__(self, endpoint: str, status: int | None = None, reason: str = "invalid_response"):
        super().__init__(f"{endpoint}: {status if status is not None else reason}")
        self.status = status
        self.reason = reason


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Gateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = Lock()
        self._requests: Counter[str] = Counter()
        self._unsupported: set[str] = set()

    def request_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._requests)

    def _post(self, endpoint: str, key: str, payload: dict, deadline: float,
              timeout: float, *, retry: bool = False) -> dict:
        request = Request(
            self.settings.base_url + endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            method="POST",
        )
        opener = build_opener(NoRedirect())
        for attempt in range(2 if retry else 1):
            remaining = deadline - time.monotonic()
            if remaining < 0.2:
                raise GatewayError(endpoint, reason="deadline")
            delay = 0.6 + random.uniform(0.0, 0.4)
            try:
                with self._lock:
                    self._requests[endpoint] += 1
                with opener.open(request, timeout=min(remaining, timeout)) as response:
                    body = response.read(12_000_001)
                    # A bounded read can return early without raising on a short Content-Length body.
                    if response.length and len(body) <= 12_000_000:
                        raise IncompleteRead(body, response.length)
                if len(body) > 12_000_000:
                    raise ValueError("Превышен размер ответа")
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Ожидался JSON-объект")
                return data
            except HTTPError as exc:
                if exc.code == 429 and (hint := exc.headers.get("Retry-After", "")).isdigit():
                    delay = min(4.0, max(delay, float(hint)))
                exc.close()
                error = GatewayError(endpoint, exc.code)
                transient = exc.code in {408, 429} or 500 <= exc.code <= 599
            except (URLError, TimeoutError, OSError, IncompleteRead):
                error = GatewayError(endpoint, reason="network_error")
                transient = True
            except (ValueError, RecursionError):
                error = GatewayError(endpoint)
                transient = False
            if attempt or not retry or not transient or deadline - time.monotonic() < delay + 1.0:
                raise error
            time.sleep(delay)
        raise GatewayError(endpoint)

    def embeddings(self, texts: list[str], deadline: float) -> list[list[float]]:
        endpoint = "/embeddings"
        data = self._post(endpoint, self.settings.embed_key or self.settings.key, {
            "model": self.settings.embed_model,
            "input": texts,
            "encoding_format": "float",
        }, deadline, 8.0)
        try:
            indexed = {int(item["index"]): item["embedding"] for item in data["data"]}
            if len(data["data"]) != len(texts) or set(indexed) != set(range(len(texts))):
                raise ValueError("Некорректные индексы")
            vectors = []
            dimensions = len(indexed[0])
            for index in range(len(texts)):
                vec = [float(number) for number in indexed[index]]
                norm = math.hypot(*vec)
                if not math.isfinite(norm) or not norm or len(vec) != dimensions:
                    raise ValueError("Некорректный вектор")
                vectors.append([number / norm for number in vec])
            return vectors
        except (IndexError, KeyError, ValueError, TypeError, OverflowError) as exc:
            raise GatewayError(endpoint) from exc

    def rerank(self, query: str, documents: list[str], deadline: float) -> list[int]:
        endpoint = "/rerank"
        data = self._post(endpoint, self.settings.rerank_key or self.settings.key, {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(self.settings.rerank_top_n, len(documents)),
        }, deadline, 7.0)
        try:
            indices = [int(item["index"]) for item in data.get("results", data.get("data", []))]
            if any(index not in range(len(documents)) for index in indices):
                raise ValueError("Некорректные индексы")
            return list(dict.fromkeys(indices))
        except (KeyError, ValueError, TypeError) as exc:
            raise GatewayError(endpoint) from exc

    def chat(self, system: str, user: dict, deadline: float, *, verify: bool = False) -> dict:
        endpoint = "/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system}, {
                "role": "user", "content": json.dumps(user, ensure_ascii=False),
            }],
            "temperature": 0.0 if verify else self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        with self._lock:
            unsupported = self._unsupported.copy()
        if self.settings.json_mode and "response_format" not in unsupported:
            payload["response_format"] = {"type": "json_object"}
        thinking = self.settings.thinking
        if thinking in {"on", "off"} and "thinking" not in unsupported:
            payload["thinking"] = {"type": "enabled" if thinking == "on" else "disabled"}

        with span("proxy.verify" if verify else "proxy.generate", kind="LLM") as trace:
            try:
                data = self._chat_post(payload, deadline)
                return self._content(data)
            except GatewayError as exc:
                if exc.reason not in {"empty", "truncated", "invalid_json", "interrupted"} or deadline - time.monotonic() < 5:
                    trace.set("psb.gateway.failure", exc.reason)
                    raise
                trace.set("psb.gateway.repair", exc.reason)
                if exc.reason == "truncated":
                    payload["max_tokens"] = min(8192, 2 * self.settings.max_tokens)
            # No model draft or hidden reasoning is carried into the repair request.
            if thinking == "auto" and "thinking" not in unsupported:
                payload["thinking"] = {"type": "disabled"}
            payload["messages"][0]["content"] += "\nВерни короткий законченный JSON. Не добавляй комментарии вне JSON."
            try:
                return self._content(self._chat_post(payload, deadline, retry=False))
            except GatewayError as exc:
                trace.set("psb.gateway.failure", exc.reason)
                raise

    def _chat_post(self, payload: dict, deadline: float, *, retry: bool = True) -> dict:
        endpoint = "/chat/completions"
        with self._lock:
            payload = {name: value for name, value in payload.items() if name not in self._unsupported}
        try:
            return self._post(endpoint, self.settings.key, payload, deadline, 38.0, retry=retry)
        except GatewayError as exc:
            optional = set(payload) & {"thinking", "response_format"}
            if not optional or exc.status not in {400, 422}:
                raise
            compatible = {key: value for key, value in payload.items() if key not in optional}
            data = self._post(endpoint, self.settings.key, compatible, deadline, 38.0)
            with self._lock:
                self._unsupported.update(optional)
            return data

    @staticmethod
    def _content(data: dict) -> dict:
        endpoint = "/chat/completions"
        try:
            choice = data["choices"][0]
            finish = choice.get("finish_reason")
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise GatewayError(endpoint) from exc
        if finish == "length":
            raise GatewayError(endpoint, reason="truncated")
        if finish in ("insufficient_system_resource", "aborted"):
            raise GatewayError(endpoint, reason="interrupted")
        if finish == "content_filter":
            raise GatewayError(endpoint, reason="content_filter")
        try:
            content = choice["message"].get("content")
        except (AttributeError, KeyError, TypeError) as exc:
            raise GatewayError(endpoint) from exc
        if not isinstance(content, str) or not content.strip():
            raise GatewayError(endpoint, reason="empty")
        content = content.lstrip("\ufeff \r\n\t")
        if match := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```\s*", content, re.I | re.S):
            content = match[1]
        try:
            result = json.loads(content)
        except (ValueError, RecursionError) as exc:
            raise GatewayError(endpoint, reason="invalid_json") from exc
        if not isinstance(result, dict):
            raise GatewayError(endpoint, reason="invalid_json")
        return result
