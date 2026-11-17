from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant import Answer, Citation, Claim, LegalAssistant
from corpus import Document, Passage, load_corpus
from proxy import Settings
from web import AssistantServer, passage_bounds, quote_offsets


KB = Path(__file__).resolve().parents[1] / "data" / "knowledge-base.zip"


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(KB)
        cls.rent = next(p for p in cls.corpus.passages.values() if p.text.startswith("3.1. Арендная плата составляет"))

    def launch(self, handler, *, diagnostics=False):
        assistant = SimpleNamespace(corpus=self.corpus, ask_dialogue=handler)
        server = AssistantServer(("127.0.0.1", 0), assistant, False, diagnostics=diagnostics)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def request(path, payload=None):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                if payload is None:
                    connection.request("GET", path)
                else:
                    connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, json.load(response)
            finally:
                connection.close()

        return request

    def test_structured_citations_and_opt_in_diagnostics(self):
        def handler(question, history, *, diagnostics=False):
            self.assertEqual(history, [])
            self.assertTrue(diagnostics)
            passage = self.rent
            title = self.corpus.documents[passage.document_id].title
            first = Citation(passage.id, passage.document_id, title, passage.locator, "Арендная плата составляет 450 000")
            second = Citation(passage.id, passage.document_id, title, passage.locator, "включая НДС")
            claims = (Claim("Аренда стоит 450 000 рублей в месяц.", (first,)),
                      Claim("НДС включён.", (second,)))
            return Answer(
                "partial", "", [first, second], claims=claims, gap="Размер депозита не установлен.",
                trace={
                    "documents": [{"document_id": passage.document_id, "title": "external secret", "applicability": "unknown"}],
                    "claims": [{"text": claims[0].text, "verdict": "supported", "reason": "Совпадает с источником",
                                "prompt": "external secret"}],
                    "retry_used": True, "elapsed_ms": 300, "api_key": "external secret",
                },
            )

        request = self.launch(handler, diagnostics=True)
        self.assertEqual(request("/healthz"), (200, {"status": "ok"}))
        status, answer = request("/api/ask", {"messages": [{"role": "user", "content": "Какая аренда?"}]})
        self.assertEqual(status, 200)
        self.assertEqual(answer["status"], "partial")
        self.assertIn("Размер депозита", answer["gap"])
        sources = {citation["ref"]: citation for citation in answer["citations"]}
        first_ref, = answer["claims"][0]["sources"]
        second_ref, = answer["claims"][1]["sources"]
        self.assertNotEqual(first_ref, second_ref)
        self.assertEqual(sources[first_ref]["id"], sources[second_ref]["id"])
        status, document = request("/api/documents/" + self.rent.document_id)
        self.assertEqual(status, 200)
        for source in sources.values():
            start, end = source["highlights"][0]
            highlighted = document["text"].encode("utf-16-le")[start * 2:end * 2].decode("utf-16-le")
            self.assertEqual(highlighted, source["quote"])
        self.assertTrue(answer["diagnostics"]["retry_used"])
        self.assertNotIn("external secret", json.dumps(answer))

    def test_parallel_clients_cannot_inherit_another_dialogue(self):
        def handler(question, history):
            anchor = history[0]["text"] if history else "Без истории"
            return Answer("insufficient", f"{anchor} — {question}", [], trace={"elapsed_ms": 0})

        request = self.launch(handler)
        follow_up = {"role": "user", "content": "А по нему какой срок?"}
        first = {"messages": [{"role": "user", "content": "Договор А-08/2024"},
                              {"role": "assistant", "content": "Ответ по договору."}, follow_up]}
        second = {"messages": [follow_up]}
        with ThreadPoolExecutor(max_workers=2) as pool:
            with_history = pool.submit(request, "/api/ask", first)
            without_history = pool.submit(request, "/api/ask", second)
            first_status, first_answer = with_history.result()
            second_status, second_answer = without_history.result()
        self.assertEqual((first_status, second_status), (200, 200))
        self.assertIn("А-08/2024", first_answer["text"])
        self.assertIn("Без истории", second_answer["text"])
        self.assertNotIn("А-08/2024", second_answer["text"])
        self.assertNotIn("diagnostics", first_answer)

    def test_conditional_variants_are_cited_and_keep_the_missing_information(self):
        older, newer = (
            next(p for p in self.corpus.passages.values() if p.locator == "п. 2.3" and edition in
                 self.corpus.documents[p.document_id].title)
            for edition in ("ред 2023", "ред 2025")
        )
        claims = [
            {"text": "Если применима прежняя редакция, порог составляет свыше 1 млн рублей.",
             "evidence": [older.id]},
            {"text": "Если применима действующая редакция, порог составляет свыше 3 млн рублей.",
             "evidence": [newer.id]},
            {"text": "Для применения порога важна сумма договора.", "evidence": [newer.id],
             "condition": "Выдуманное исключение"},
        ]
        gap = "Не указана дата заключения договора."
        assistant = LegalAssistant(self.corpus, Settings(
            base_url="http://fixture.local", key="fixture-only", model="fixture", retry_search=False,
        ))

        def handler(question, history):
            if question == "Какая редакция была в 2020 году?":
                return Answer("insufficient", gap, [], gap=gap)
            return assistant.ask_dialogue(question, history)

        request = self.launch(handler)
        with patch.object(assistant.gateway, "chat", side_effect=[
            {"status": "partial", "claims": claims, "gap": gap},
            {"supported": [0, 1, 2], "gap_ok": True},
        ]) as gateway:
            status, answer = request("/api/ask", {"messages": [{
                "role": "user",
                "content": "Как менялся порог согласования с финансовым директором? Дата договора неизвестна.",
            }]})
        self.assertEqual(gateway.call_count, 2)
        self.assertEqual([item["text"] for item in gateway.call_args.args[1]["claims"]],
                         [claim["text"] for claim in claims])
        self.assertEqual(status, 200)
        self.assertEqual(answer["status"], "partial")
        self.assertEqual(answer["gap"], gap)
        self.assertEqual([claim["condition"] for claim in answer["claims"]],
                         ["Если применима прежняя редакция", "Если применима действующая редакция", ""])
        for claim, source in zip(answer["claims"][:2], answer["citations"], strict=True):
            self.assertEqual(claim["sources"], [source["ref"]])
        status, unanswered = request("/api/ask", {
            "messages": [{"role": "user", "content": "Какая редакция была в 2020 году?"}]
        })
        self.assertEqual(status, 200)
        self.assertEqual(unanswered["gap"], gap)
        self.assertEqual(unanswered["claims"], [])

    def test_service_failure_does_not_return_exception_or_model_details(self):
        def handler(question, history):
            if question == "exception":
                raise RuntimeError("private-proxy-url and private-key")
            return Answer("error", "private-proxy-url and private-key", [])

        request = self.launch(handler)
        for question, expected in (("exception", 503), ("error answer", 200)):
            status, result = request("/api/ask", {"messages": [{"role": "user", "content": question}]})
            self.assertEqual(status, expected)
            self.assertNotIn("private-", json.dumps(result))

    def test_highlights_normalize_whitespace_and_use_browser_offsets(self):
        text = "📁 Глава\n\nПункт 1.  Условие\r\n\t и срок.\nПункт 2."
        quote = "Условие и срок."
        spans = quote_offsets(text, quote)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        original = text.encode("utf-16-le")[start * 2:end * 2].decode("utf-16-le")
        self.assertEqual(original, "Условие\r\n\t и срок.")

    def test_every_corpus_passage_can_be_highlighted(self):
        bounds = passage_bounds(self.corpus)
        for passage in self.corpus.passages.values():
            with self.subTest(passage=passage.id):
                document = self.corpus.documents[passage.document_id]
                self.assertTrue(quote_offsets(document.text, passage.text, bounds.get(passage.id)))

    def test_repeated_excerpt_highlights_the_cited_provision(self):
        sentence = "Правило применяется."
        text = f"Раздел 1\n{sentence}\nРаздел 2\n{sentence}"
        corpus = SimpleNamespace(
            documents={"D1": Document("D1", "Документ", text, "fixture.txt")},
            passages={f"P{i}": Passage(f"P{i}", "D1", f"раздел {i}", sentence) for i in (1, 2)},
        )
        self.assertEqual(quote_offsets(text, sentence, passage_bounds(corpus)["P2"]),
                         [[text.rindex(sentence), len(text)]])


if __name__ == "__main__":
    unittest.main()
