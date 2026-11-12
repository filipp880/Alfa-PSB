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

    def test_follow_up_inherits_a_contract_with_an_amendment(self):
        assistant = LegalAssistant(self.corpus, Settings(), offline=True)
        history = [{"role": "user", "text": "Мы подписали спецификацию по П-17/2024."},
                   {"role": "assistant", "text": "Сроки по ПТ-05/2023 могут отличаться."}]
        original = self.corpus.scope(history[0]["text"]).exact_document_ids
        self.assertEqual(len(original), 2)
        self.assertEqual(assistant.scope_for("А куда доставлять по этому договору?", history).exact_document_ids,
                         original)
        switched = history + [{"role": "user", "text": "Теперь речь о договоре П-99/2024."}]
        self.assertFalse(assistant.scope_for("А куда доставлять по этому договору?", switched).exact_document_ids)
        explicit = "А как доставлять по ПТ-05/2023?"
        self.assertEqual(assistant.scope_for(explicit, history).exact_document_ids,
                         self.corpus.scope(explicit).exact_document_ids)

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

    def test_insufficient_draft_can_still_answer_a_verified_part(self):
        assistant = LegalAssistant(
            self.corpus, Settings(base_url="http://localhost", key="fixture", model="fixture", retry_search=False)
        )
        passage = next(p for p in self.corpus.search(RENT, 12) if RENT_QUOTE in p.text)
        draft = {
            "status": "insufficient",
            "claims": [
                {"text": AMOUNT, "evidence": [passage.id]},
                {"text": "Обеспечительный платёж составляет 500 000 рублей.", "evidence": [passage.id]},
            ],
            "gap": "История перечислений арендатора в базе отсутствует.",
        }
        with patch.object(assistant.gateway, "chat", side_effect=[draft, {"supported": [0], "gap_ok": True}]) as chat:
            answer = assistant.ask(RENT + " В каком месяце арендатор последний раз перечислил деньги?")
        self.assertEqual(answer.status, "partial")
        self.assertEqual([citation.id for citation in answer.citations], [passage.id])
        self.assertIn(AMOUNT, answer.text)
        self.assertIn(draft["gap"], answer.text)
        self.assertNotIn("500 000", answer.text)
        self.assertTrue(chat.call_args.kwargs["verify"])

    def test_mistyped_quote_is_replaced_by_verbatim_verified_passage(self):
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture", model="fixture"))
        passage = next(p for p in self.corpus.search(RENT, 12) if RENT_QUOTE in p.text)
        draft = {"status": "answered", "claims": [{"text": AMOUNT,
                 "evidence": [{"id": passage.id, "quote": "Арендная плата составляет 450.000 рублей в месяц."}]}]}
        review = {"supported": [0], "gap_ok": False}
        with patch.object(assistant.gateway, "chat", side_effect=[draft, review]) as chat:
            answer = assistant.ask(RENT)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(answer.status, "answered")
        self.assertEqual([(c.id, c.quote) for c in answer.citations], [(passage.id, passage.text)])

    def test_empty_answered_draft_is_technical_failure(self):
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture", model="fixture"))
        with patch.object(assistant.gateway, "chat", return_value={"status": "answered", "claims": []}):
            answer = assistant.ask(RENT)
        self.assertEqual(answer.status, "error")
        self.assertFalse(answer.citations)

    def test_verifier_sees_exception_and_keeps_confirmed_part(self):
        question = ("Можно ли в 2026 году нанять человека на заведомо временные работы на 45 дней "
                    "и установить ему испытание на неделю?")
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture", model="fixture"))
        term = next(p for p in self.corpus.passages.values()
                    if "на время выполнения временных (до двух месяцев) работ" in p.text)
        general = next(p for p in self.corpus.passages.values()
                       if "от двух до шести месяцев испытание" in p.text)
        exception = next(p for p in self.corpus.passages.values()
                         if "лиц, заключающих трудовой договор на срок до двух месяцев;" in p.text)
        draft = {"status": "answered", "claims": [
            {"text": "Допускается срочный договор на 45 дней для временных работ.", "evidence": [term.id]},
            {"text": "Можно установить испытание на неделю.", "evidence": [general.id]},
        ]}
        draft_ids = set()

        def chat(system, payload, deadline, *, verify=False):
            ids = {p["id"] for p in payload["passages"]}
            if not verify:
                draft_ids.update(ids)
                self.assertIn(exception.id, draft_ids)
                return draft
            self.assertEqual(ids, draft_ids)
            return {"supported": [0], "gap_ok": False}

        with patch.object(assistant.gateway, "chat", chat):
            answer = assistant.ask(question)
        self.assertEqual(answer.status, "partial")
        self.assertEqual([citation.id for citation in answer.citations], [term.id])
        self.assertNotIn("испытание на неделю", answer.text)

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

    def test_wrong_edition_is_removed_from_mixed_support(self):
        assistant = LegalAssistant(self.corpus, Settings())
        scope = assistant.scope_for("Нужно ли одобрение гендиректора, если договор заключили 28.02.2025?", [])
        old, new = [next(p for p in self.corpus.passages.values() if p.locator == "п. 2.4"
                         and edition in self.corpus.documents[p.document_id].title)
                    for edition in ("ред 2023", "ред 2025")]
        sources = {p.id: p for p in (old, new)}
        draft = {"claims": [{"text": "Применяется порог 10 000 000 рублей.", "evidence": [old.id, new.id]}]}
        claim, = assistant.validated_claims(draft, sources, scope)
        self.assertEqual([citation.id for citation in claim.citations], [old.id])
        comparison = assistant.scope_for(
            "Сравните пороги одобрения для договоров, подписанных 28.02.2025 и 01.03.2025.", [])
        claim, = assistant.validated_claims(draft, sources, comparison)
        self.assertEqual([citation.id for citation in claim.citations], [old.id, new.id])

    def test_transition_rule_can_explain_why_amendment_does_not_apply(self):
        assistant = LegalAssistant(self.corpus, Settings())
        scope = assistant.scope_for("Куда везти товар по П-17/2024, если спецификацию подписали 01.10.2024?", [])
        base, amendment = [next(p for p in self.corpus.passages.values() if p.locator == locator
                                and self.corpus.documents[p.document_id].reference == "п-17/2024")
                           for locator in ("п. 3.2", "изменение п. 3.2")]
        draft = {"claims": [{"text": "Применяется адрес в Москве: спецификация подписана 1 октября.",
                             "evidence": [base.id, amendment.id]}]}
        claim, = assistant.validated_claims(draft, {p.id: p for p in (base, amendment)}, scope)
        self.assertEqual([citation.id for citation in claim.citations], [base.id, amendment.id])

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
