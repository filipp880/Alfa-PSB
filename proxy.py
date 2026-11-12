"""Client for the single OpenAI-compatible gateway specified by the organizers."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

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
    json_mode: bool = False

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
            json_mode=env("LLM_JSON_MODE") == "1",
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
    def __init__(self, endpoint: str, status: int | None = None):
        super().__init__(f"{endpoint}: {status if status is not None else 'некорректный ответ или тайм-аут'}")
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Gateway:
    def __init__(self, settings: Settings):
        self.settings = settings

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
                raise GatewayError(endpoint)
            try:
                with opener.open(request, timeout=min(remaining, timeout)) as response:
                    data = json.loads(response.read(12_000_001))
                if not isinstance(data, dict):
                    raise ValueError("Ожидался JSON-объект")
                return data
            except HTTPError as exc:
                exc.close()
                error = GatewayError(endpoint, exc.code)
                transient = exc.code in {408, 429} or 500 <= exc.code <= 599
            except (URLError, TimeoutError, OSError, ValueError):
                error = GatewayError(endpoint)
                transient = True
            if attempt or not retry or not transient or deadline - time.monotonic() < 1.5:
                raise error
            time.sleep(0.65)
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
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
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
        if self.settings.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(endpoint, self.settings.key, payload, deadline, 42.0, retry=True)
        except GatewayError as exc:
            if "response_format" not in payload or exc.status not in {400, 422}:
                raise
            del payload["response_format"]
            data = self._post(endpoint, self.settings.key, payload, deadline, 42.0)
        try:
            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```") and content.endswith("```"):
                _, _, content = content.partition("\n")
                content = content[:-3].strip()
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Ожидался JSON-объект")
            return result
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise GatewayError(endpoint) from exc
