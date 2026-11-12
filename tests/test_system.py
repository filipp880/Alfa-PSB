import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from assistant import LegalAssistant, dialogue
from corpus import load_corpus
from proxy import GatewayError, Settings

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
                evidence = [{"id": matches[0], "quote": RENT_QUOTE}]
                answer = {"status": "answered", "claims": [{"text": AMOUNT, "evidence": evidence},
                    {"text": "Арендная плата составляет 500 000 рублей в месяц.", "evidence": evidence},
                    {"text": "Пункт выдуман.", "evidence": [{"id": "P999999", "quote": RENT_QUOTE}]}], "gap": ""}
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
