"""Serve the local legal assistant and its document browser."""

from __future__ import annotations

import argparse
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from assistant import LegalAssistant, load_assistant


STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 128 * 1024
MAX_DIALOG_CHARACTERS = 24_000
LOGGER = logging.getLogger("psb.web")


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


class AssistantServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], assistant: LegalAssistant, offline: bool):
        self.assistant = assistant
        self.offline = offline
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
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            self._send(HTTPStatus.OK, (STATIC_ROOT / filename).read_bytes(), content_type)
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
            length = int(self.headers.get("Content-Length", "0"))
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
            answer = self.server.assistant.ask_dialogue(question, history)
        except Exception as exc:
            LOGGER.error("Assistant request failed: %s", type(exc).__name__)
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Сервис не смог обработать запрос. Повторите попытку."})
            return

        self._json(HTTPStatus.OK, {
            "status": answer.status,
            "text": answer.text,
            "event_ranges": answer.event_ranges,
            "citations": [
                {
                    "id": citation.id,
                    "document_id": citation.document_id,
                    "title": citation.title,
                    "locator": citation.locator,
                    "quote": citation.quote,
                    "effective_from": citation.effective_from,
                    "effective_to": citation.effective_to,
                    "snapshot_date": citation.snapshot_date,
                    "applicability": citation.applicability,
                }
                for citation in answer.citations
            ],
        })


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
    with AssistantServer((args.host, args.port), assistant, args.offline) as server:
        print(f"ПСБ: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
