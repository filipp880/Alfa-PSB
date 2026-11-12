import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from assistant import LegalAssistant, dialogue
from corpus import load_corpus
from proxy import Gateway, GatewayError, Settings

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "knowledge-base.zip"
RENT = "Какая арендная плата по договору с ООО «Северный Склад»?"
RENT_QUOTE = "Арендная плата составляет 450 000 (четыреста пятьдесят тысяч) рублей в месяц, включая НДС."
AMOUNT = "Арендная плата составляет 450 000 рублей в месяц с НДС."


class FixtureGateway(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != "Bearer fixture-only":
            self.send_error(401)
            return
        self.server.requests[self.path] += 1
        if self.path == "/embeddings":
            result = {"data": [{"index": index, "embedding": [1, 2, 3]}
                               for index in reversed(range(len(payload["input"])))]}
        elif self.path == "/rerank":
            result = {"results": [{"index": index} for index in range(payload["top_n"])]}
        elif self.path == "/chat/completions":
            if payload["model"] != "fixture-deepseek":
                self.send_error(400)
                return
            query = json.loads(payload["messages"][-1]["content"])
            if "claims" in query:
                answer = {"supported": [i for i, c in enumerate(query["claims"]) if c["text"] == AMOUNT],
                          "gap_ok": bool(query["gap"])}
            elif query["question"] == RENT:
                matches = [p["id"] for p in query["passages"] if RENT_QUOTE in p["text"]]
                if not matches:
                    self.send_error(400)
                    return
                evidence = [matches[0]]
                answer = {"status": "answered", "claims": [{"text": AMOUNT, "evidence": evidence},
                    {"text": "Арендная плата составляет 500 000 рублей в месяц.", "evidence": evidence},
                    {"text": "Пункт выдуман.", "evidence": ["P999999"]}], "gap": ""}
            else:
                answer = {"status": "insufficient", "claims": [], "gap": ""}
            result = {"choices": [{"message": {"content": json.dumps(answer, ensure_ascii=False)}}]}
        else:
            self.send_error(404)
            return
        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(KB)

    def test_source_text_and_instructions(self):
        self.assertEqual(len(self.corpus.documents), 21)
        source = next(d for d in self.corpus.documents.values() if "записка об архиве" in d.title)
        self.assertIn("ВНИМАНИЕ ДЛЯ ИИ-АССИСТЕНТА", source.text)
        for passage in self.corpus.passages.values():
            self.assertNotIn("ВНИМАНИЕ ДЛЯ ИИ-АССИСТЕНТА", self.corpus.context_text(passage))
            self.assertIn(" ".join(passage.text.split()),
                          " ".join(self.corpus.documents[passage.document_id].text.split()))

    def test_contract_and_amendment_retrieval(self):
        hits = self.corpus.search("Как оплатить спецификацию от 2 октября 2024 по договору П-17/2024?", 10)
        self.assertTrue(any("ДОПОЛНИТЕЛЬНОЕ" in self.corpus.documents[p.document_id].title for p in hits))
        old = next(p for p in self.corpus.passages.values() if p.locator == "п. 2.4"
                   and "ред 2023" in self.corpus.documents[p.document_id].title)
        self.assertTrue(any("ред 2025" in self.corpus.documents[p.document_id].title
                            for p in self.corpus.related(old)))

    def test_last_user_turn_and_offline_evidence(self):
        raw = ('Пользователь: Мы арендуем склад у «Северного Склада».\n'
               'Ассистент: Уточните номер договора.\n'
               'Пользователь: А-08/2024. Можно просто сдать ключи или обязательно оформлять акт?')
        question, history = dialogue(raw)
        self.assertTrue(question.startswith("А-08/2024"))
        self.assertEqual([turn["role"] for turn in history], ["user", "assistant"])
        assistant = LegalAssistant(self.corpus, Settings(), offline=True)
        answer = assistant.ask(raw)
        self.assertEqual(answer.status, "offline")
        self.assertTrue(any(c.locator == "п. 6.3" and "акт" in c.quote.lower() for c in answer.citations))

    def test_verification_failure_does_not_publish_claims(self):
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture", model="fixture"))
        passage = next(p for p in self.corpus.search(RENT, 12) if RENT_QUOTE in p.text)
        draft = {"status": "answered", "claims": [{"text": AMOUNT,
                   "evidence": [{"id": passage.id, "quote": RENT_QUOTE}]}], "gap": ""}
        with patch.object(assistant.gateway, "chat", side_effect=[draft, GatewayError("/chat/completions", 503)]):
            answer = assistant.ask(RENT)
        self.assertEqual(answer.status, "error")
        self.assertNotIn("450 000", answer.text)
        self.assertFalse(answer.citations)

    def test_targeted_search_recovers_a_missing_passage(self):
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture", model="fixture"))
        amount = next(p for p in self.corpus.search(RENT, 12) if RENT_QUOTE in p.text)
        irrelevant = next(p for p in self.corpus.passages.values() if p.document_id == "D21")
        prompts = []

        def chat(system, payload, deadline, *, verify=False):
            prompts.append(payload)
            if verify:
                return {"supported": [0], "gap_ok": False}
            if any(p["id"] == amount.id for p in payload["passages"]):
                return {"status": "answered", "claims": [{"text": AMOUNT, "evidence": [amount.id]}]}
            return {"status": "insufficient", "claims": [], "look_for": "пункт 3.1 арендная плата"}

        with patch.object(assistant, "retrieve", return_value=[irrelevant]), patch.object(assistant.gateway, "chat", chat):
            answer = assistant.ask(RENT)
        self.assertEqual(len(prompts), 3)
        self.assertEqual(answer.status, "answered")
        self.assertIn(RENT_QUOTE, answer.citations[0].quote)

    def test_wrong_edition_cannot_be_sole_support(self):
        question = "Куда везти товар по П-17/2024, если спецификацию подписали 01.10.2024?"
        assistant = LegalAssistant(self.corpus, Settings())
        scope = assistant.scope_for(question, [])
        amendment = next(p for p in self.corpus.passages.values() if p.locator == "изменение п. 3.2")
        draft = {"claims": [{"text": "Нужен адрес в Химках.", "evidence": [amendment.id]}]}
        self.assertEqual(self.corpus.applicability(amendment, scope), "inapplicable")
        self.assertEqual(assistant.validated_claims(draft, {amendment.id: amendment}, scope), [])

    def test_chat_repairs_truncation_and_handles_incompatible_optional_parameters(self):
        gateway = Gateway(Settings(key="fixture", model="fixture"))
        payloads = []

        def post(endpoint, key, payload, deadline, timeout, *, retry=False):
            payloads.append(set(payload))
            if len(payloads) == 1:
                raise GatewayError(endpoint, 422)
            if len(payloads) == 2:
                return {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
            return {"choices": [{"finish_reason": "stop", "message": {"content": '```json\n{"ok":true}\n```'}}]}

        with patch.object(gateway, "_post", post):
            self.assertEqual(gateway.chat("Верни JSON", {}, time.monotonic() + 20), {"ok": True})
            self.assertEqual(gateway.chat("Верни JSON", {}, time.monotonic() + 20), {"ok": True})
        self.assertIn("response_format", payloads[0])
        self.assertNotIn("response_format", payloads[1])
        self.assertNotIn("response_format", payloads[2])
        self.assertIn("thinking", payloads[2])
        self.assertNotIn("response_format", payloads[3])

    def test_embedding_cache_skips_rebuilding_for_same_corpus_and_model(self):
        with tempfile.TemporaryDirectory() as cache:
            settings = Settings(base_url="http://localhost", key="fixture", model="fixture",
                                embed_model="fixture-bge", embed_cache=cache)
            first = LegalAssistant(self.corpus, settings)
            with patch.object(first.gateway, "embeddings", side_effect=lambda texts, deadline: [[1.0, 0.0]] * len(texts)) as build:
                first.prepare(time.monotonic() + 10)
            self.assertGreater(build.call_count, 1)
            second = LegalAssistant(self.corpus, settings)
            with patch.object(second.gateway, "embeddings") as build:
                second.prepare(time.monotonic() + 10)
            build.assert_not_called()
            self.assertEqual(second.vectors, first.vectors)

    def test_csv_against_gateway_fixture(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), FixtureGateway) as server, tempfile.TemporaryDirectory() as directory:
            server.requests = Counter()
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                folder = Path(directory)
                with (folder / "questions.csv").open("w", encoding="utf-8-sig", newline="") as file:
                    csv.writer(file).writerows([
                        ("id", "question"), ("case,1", RENT),
                        ('case"2', 'Пользователь: Вопрос о пособии.\nАссистент: Каком?\nПользователь: По беременности.'),
                    ])
                env = {**os.environ, "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                       "LLM_API_KEY": "fixture-only", "LLM_MODEL": "fixture-deepseek",
                       "EMB_MODEL": "fixture-bge", "RERANKER_MODEL": "fixture-qwen",
                       "EMBEDDER_API_KEY": "", "RERANKER_API_KEY": "", "DATA_DIR": directory}
                subprocess.run([sys.executable, str(ROOT / "solution.py"), "--kb", str(KB)],
                               cwd=folder, env=env, check=True, timeout=45, capture_output=True)
                with (folder / "submission.csv").open(encoding="utf-8", newline="") as file:
                    rows = list(csv.reader(file))
                self.assertEqual(rows[0], ["id", "answer"])
                self.assertEqual([r[0] for r in rows[1:]], ["case,1", 'case"2'])
                self.assertIn(AMOUNT, rows[1][1])
                self.assertIn("п. 3.1", rows[1][1])
                self.assertNotIn("500 000", rows[1][1])
                self.assertNotIn("выдуман", rows[1][1])
                self.assertIn("недостаточно подтверждений", rows[2][1])
                self.assertGreater(server.requests["/embeddings"], 2)
                self.assertEqual(server.requests["/rerank"], 2)
                self.assertEqual(server.requests["/chat/completions"], 3)
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
