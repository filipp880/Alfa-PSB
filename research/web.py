"""Serve the local legal assistant and its document browser."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from assistant import Answer, LegalAssistant, load_assistant
from corpus import Corpus, Document


STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 128 * 1024
MAX_DIALOG_CHARACTERS = 24_000
LOGGER = logging.getLogger("psb.web")
APPLICABILITY = {"applicable", "inapplicable", "unknown", "comparison"}
VERDICTS = {"supported", "rejected", "unclear"}
SERVICE_ERROR = "Сервис не смог обработать запрос. Повторите попытку."
CONDITIONAL_LEAD = re.compile(r"^(?:если|при условии(?:, что)?)\s+.{3,150}?(?=,\s|\s+[—–]\s)", re.I)


def format_dialog(payload: object) -> tuple[str, list[dict[str, str]]]:
    if not isinstance(payload, dict):
        raise ValueError("Ожидается объект с полем messages.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 15:
        raise ValueError("В диалоге должно быть от 1 до 15 реплик.")
    if len(messages) % 2 == 0:
        raise ValueError("Последняя реплика должна быть вопросом пользователя.")

    turns = []
    total_characters = 0
    for index, message in enumerate(messages):
        is_user = index % 2 == 0
        role = "user" if is_user else "assistant"
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError("Нарушен порядок реплик диалога.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Реплика должна содержать текст.")
        content = content.strip()
        total_characters += len(content)
        if total_characters > MAX_DIALOG_CHARACTERS:
            raise ValueError("Диалог превышает допустимый объём текста.")
        turns.append({"role": role, "text": content})
    return turns[-1]["text"], turns[:-1]


def text_span(text: str, excerpt: str, start: int = 0, end: int | None = None) -> tuple[int, int] | None:
    """Locate a passage in its original file despite normalized line breaks."""
    words = excerpt.split()
    if not words:
        return None
    pattern = re.compile(r"\s+".join(map(re.escape, words)))
    match = pattern.search(text, start, len(text) if end is None else end)
    return match.span() if match else None


def excerpt_spans(text: str, excerpt: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    if span := text_span(text, excerpt, start, end):
        return [span]
    spans = []
    for line in filter(str.strip, excerpt.splitlines()):
        found = text_span(text, line, start, end)
        if found is None:
            return []
        spans.append(found)
        start = found[1]
    return spans


def passage_bounds(corpus: Corpus) -> dict[str, tuple[int, int]]:
    """Match passages in document order to distinguish repeated provisions."""
    bounds = {}
    cursors: dict[str, int] = {}
    for passage in corpus.passages.values():
        document = corpus.documents[passage.document_id]
        spans = excerpt_spans(document.text, passage.text, cursors.get(document.id, 0))
        if spans:
            cursors[document.id] = spans[-1][1]
            bounds[passage.id] = spans[0][0], spans[-1][1]
    return bounds


def quote_offsets(text: str, quote: str, bounds: tuple[int, int] | None = None) -> list[list[int]]:
    """Return browser string offsets for each verbatim portion of a quote."""
    spans = excerpt_spans(text, quote, *(bounds or (0, len(text))))
    # JavaScript slices UTF-16 code units, while Python indexes code points.
    return [[len(text[:offset].encode("utf-16-le")) // 2 for offset in span] for span in spans]


def public_trace(trace: dict, documents: dict[str, Document]) -> dict:
    """Serialize only diagnostics intended for the chat interface."""
    sources = []
    for entry in trace.get("documents", [])[:24]:
        document = documents.get(entry.get("document_id"))
        if document is not None:
            applicability = entry.get("applicability")
            sources.append({"title": document.title,
                            "applicability": applicability if applicability in APPLICABILITY else "unknown"})
    claims = [
        {"text": entry["text"][:900], "verdict": entry["verdict"], "reason": entry.get("reason", "")[:160]}
        for entry in trace.get("claims", [])[:18]
        if entry.get("verdict") in VERDICTS and isinstance(entry.get("text"), str)
    ]
    return {
        "documents": sources,
        "claims": claims,
        "retry_used": trace.get("retry_used") is True,
        "elapsed_ms": max(0, int(trace.get("elapsed_ms", 0))),
    }


def confirmed_condition(text: str) -> str:
    """Use only a condition already stated in the verified wording."""
    match = CONDITIONAL_LEAD.match(" ".join(text.split()))
    return match[0] if match else ""


def answer_payload(answer: Answer, corpus: Corpus, bounds: dict[str, tuple[int, int]], diagnostics: bool) -> dict:
    citations = list(dict.fromkeys(answer.citations))
    refs = {citation: f"source-{index}" for index, citation in enumerate(citations, 1)}
    sources = []
    for citation in citations:
        document = corpus.documents.get(citation.document_id)
        highlights = quote_offsets(document.text, citation.quote, bounds.get(citation.id)) if document else []
        sources.append({
            "id": citation.id,
            "ref": refs[citation],
            "document_id": citation.document_id,
            "title": citation.title,
            "locator": citation.locator,
            "quote": citation.quote,
            "highlights": highlights,
            "effective_from": citation.effective_from,
            "effective_to": citation.effective_to,
            "snapshot_date": citation.snapshot_date,
            "applicability": citation.applicability,
            "edition": citation.edition,
        })
    result = {
        "status": answer.status,
        "text": SERVICE_ERROR if answer.status == "error" else answer.csv_text(),
        "event_ranges": answer.event_ranges,
        "citations": sources,
        "claims": [
            {"text": claim.text, "sources": [refs[citation] for citation in claim.citations],
             "condition": confirmed_condition(claim.text) if claim.citations else ""}
            for claim in answer.claims
        ],
        "gap": answer.gap,
        "intro": answer.intro,
        "conflict": answer.conflict,
    }
    if diagnostics and answer.trace is not None:
        result["diagnostics"] = public_trace(answer.trace, corpus.documents)
    return result


class AssistantServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], assistant: LegalAssistant, offline: bool, *, diagnostics: bool = False):
        self.assistant = assistant
        self.offline = offline
        self.diagnostics = diagnostics
        self.passage_bounds = passage_bounds(assistant.corpus)
        super().__init__(address, AssistantHandler)


class AssistantHandler(BaseHTTPRequestHandler):
    server: AssistantServer

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionError:
            # A browser may abort when starting a new dialogue.
            return

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            self._send(HTTPStatus.OK, (STATIC_ROOT / filename).read_bytes(), content_type)
            return
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return

        documents = self.server.assistant.corpus.documents
        if path == "/api/documents":
            index = sorted(documents.values(), key=lambda item: (item.title.casefold(), item.id))
            self._json(HTTPStatus.OK, {"offline": self.server.offline,
                "model_configured": self.server.assistant.settings.ready, "documents": [
                {"id": item.id, "title": item.title, "filename": item.filename} for item in index
            ]})
            return

        if path.startswith("/api/documents/"):
            document = documents.get(unquote(path.removeprefix("/api/documents/")))
            if document is not None:
                self._json(HTTPStatus.OK, {
                    "id": document.id,
                    "title": document.title,
                    "filename": document.filename,
                    "text": document.text,
                })
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "Документ не найден в базе."})
            return

        self._json(HTTPStatus.NOT_FOUND, {"error": "Страница не найдена."})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/ask":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Страница не найдена."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Ожидается запрос в формате JSON."})
            return

        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Превышен допустимый размер запроса или запрос пуст.")
            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValueError("Не удалось прочитать JSON-запрос.") from exc
            question, history = format_dialog(payload)
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            options = {"diagnostics": True} if self.server.diagnostics else {}
            answer = self.server.assistant.ask_dialogue(question, history, **options)
            result = answer_payload(answer, self.server.assistant.corpus, self.server.passage_bounds,
                                    self.server.diagnostics)
        except Exception as exc:
            LOGGER.error("Assistant request failed: %s", type(exc).__name__)
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": SERVICE_ERROR})
            return

        self._json(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    data_dir = Path(os.getenv("DATA_DIR") or "data")
    default_kb = next(
        (data_dir / name for name in ("kb.zip", "knowledge-base.zip", "knowledge_base.zip")
         if (data_dir / name).is_file()),
        data_dir / "kb.zip",
    )
    parser.add_argument("--kb", type=Path, default=default_kb)
    parser.add_argument(
        "--offline", action="store_true",
        default=os.getenv("OFFLINE", "").strip().casefold() in {"1", "true", "yes", "on"},
        help="Работа без обращения к API моделей",
    )
    args = parser.parse_args()

    assistant = load_assistant(args.kb, offline=args.offline)
    diagnostics = os.getenv("PSB_WEB_DIAGNOSTICS", "").strip().casefold() in {"1", "true", "yes", "on"}
    with AssistantServer((args.host, args.port), assistant, args.offline, diagnostics=diagnostics) as server:
        print(f"ПСБ: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
