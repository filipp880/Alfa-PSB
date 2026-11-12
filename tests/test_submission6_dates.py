from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

from assistant import LegalAssistant
from corpus import load_corpus
from proxy import Settings


class Submission6DateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(Path(__file__).resolve().parents[1] / "data" / "knowledge-base.zip")

    def test_requested_signature_date_is_not_a_document_requisite(self):
        old = next(p for p in self.corpus.passages.values()
                   if p.locator == "п. 3.1" and "ред 2023" in self.corpus.documents[p.document_id].title)
        new = next(p for p in self.corpus.passages.values()
                   if p.locator == "п. 3.1" and "ред 2025" in self.corpus.documents[p.document_id].title)
        for wording in ("подписания", "заключения"):
            for delivery in ("2 марта", "2 марта 2025 года"):
                question = (f"Какой порядок согласования действовал на дату {wording} договора от "
                            f"28 февраля 2025 года, если товар доставили {delivery}?")
                with self.subTest(question=question):
                    scope = self.corpus.scope(question)
                    self.assertEqual(scope.event_role, "signature")
                    self.assertEqual(scope.date_ranges, ((date(2025, 2, 28), date(2025, 2, 28)),))
                    self.assertFalse(scope.comparison)
                    self.assertEqual(self.corpus.applicability(old, scope), "applicable")
                    self.assertEqual(self.corpus.applicability(new, scope), "inapplicable")

    def test_document_requisite_alone_does_not_establish_an_event(self):
        passage = next(p for p in self.corpus.passages.values()
                       if p.locator == "п. 3.1" and "ред 2023" in self.corpus.documents[p.document_id].title)
        for question in (
            "Как согласовать договор поставки от 28 февраля 2025 года?",
            "Что сказано в регламенте согласования договоров № 42 от 01.03.2025?",
            "Спецификация к договору поставки от 28 февраля 2025 года: какой адрес?",
        ):
            with self.subTest(question=question):
                scope = self.corpus.scope(question)
                self.assertEqual(scope.event_role, "document")
                self.assertEqual(self.corpus.applicability(passage, scope), "unknown")

    def test_record_date_and_later_signature_stay_distinct(self):
        question = ("Договор поставки от 28 февраля 2025 года подписали 2 марта. "
                    "Какой порядок согласования действовал на дату подписания договора?")
        scope = self.corpus.scope(question)
        self.assertEqual(scope.event_role, "signature")
        self.assertEqual(scope.date_ranges, ((date(2025, 3, 2), date(2025, 3, 2)),))

    def test_specification_is_the_nearest_dated_object(self):
        for question in (
            "По договору поставки спецификация от 2 октября 2024 года. Куда везти по П-17/2024?",
            "Договор поставки № П-17/2024 от 12.02.2024, спецификация от 2 октября 2024 года. Куда везти?",
            "По договору поставки П-17/2024 спецификация от 2 октября 2024 года. Товар доставили 14 октября.",
        ):
            with self.subTest(question=question):
                scope = self.corpus.scope(question)
                self.assertEqual(scope.event_role, "specification")
                self.assertEqual(scope.date_ranges, ((date(2024, 10, 2), date(2024, 10, 2)),))
                self.assertFalse(scope.comparison)

    def test_explicit_delivery_focus_overrides_signature(self):
        scope = self.corpus.scope(
            "Договор подписали 28 февраля 2025 года, товар доставили 2 марта. "
            "Какой порядок действовал на дату доставки?"
        )
        self.assertEqual(scope.event_role, "delivery")
        self.assertEqual(scope.date_ranges, ((date(2025, 3, 2), date(2025, 3, 2)),))

    def test_explicit_comparison_preserves_both_signature_periods(self):
        scope = self.corpus.scope(
            "Сравните порядок согласования на дату подписания договора от 28 февраля 2025 года "
            "и на дату подписания договора от 2 марта."
        )
        self.assertTrue(scope.comparison)
        self.assertEqual(scope.event_role, "signature")
        self.assertEqual(scope.date_ranges, (
            (date(2025, 2, 28), date(2025, 2, 28)),
            (date(2025, 3, 2), date(2025, 3, 2)),
        ))

    def test_paper_approval_survives_generation_verification_and_csv(self):
        question = ("Какой порядок согласования по п. 3.1 регламента действовал на дату подписания договора от "
                    "28 февраля 2025 года, если товар доставили 2 марта?")
        passage = next(p for p in self.corpus.passages.values()
                       if p.locator == "п. 3.1" and "ред 2023" in self.corpus.documents[p.document_id].title)
        claim = "Согласование оформляется в листе согласования на бумажном носителе."
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture",
                                                        model="fixture", retry_search=False))
        draft = {"status": "answered", "claims": [{"text": claim, "evidence": [passage.id]}]}
        with patch.object(assistant.gateway, "chat", side_effect=[draft, {"supported": [0]}]) as chat:
            answer = assistant.ask(question)
        generation = {p["id"]: p for p in chat.call_args_list[0].args[1]["passages"]}
        self.assertEqual(generation[passage.id]["applicability"], "applicable")
        self.assertEqual(chat.call_args_list[1].args[1]["claims"][0]["evidence"], [passage.id])
        self.assertEqual(answer.status, "answered")
        self.assertIn(claim, answer.csv_text())
        self.assertEqual(answer.citations[0].quote, passage.text)

    def test_complete_addresses_follow_specification_boundary(self):
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture",
                                                        model="fixture", retry_search=False))
        for day, locator, address in (
            (1, "п. 3.2", "Москва, ул. Складская, д. 4, стр. 2"),
            (2, "изменение п. 3.2", "Московская область, г. Химки, ул. Заводская, д. 15, склад № 3"),
        ):
            with self.subTest(day=day):
                passage = next(p for p in self.corpus.passages.values()
                               if p.locator == locator and self.corpus.documents[p.document_id].reference == "п-17/2024")
                question = (f"По договору поставки спецификация от {day} октября 2024 года. "
                            "Укажите полный адрес доставки по П-17/2024.")
                draft = {"status": "answered", "claims": [{"text": address, "evidence": [passage.id]}]}
                with patch.object(assistant.gateway, "chat", side_effect=[draft, {"supported": [0]}]):
                    answer = assistant.ask(question)
                self.assertEqual(answer.status, "answered")
                self.assertIn(address, answer.csv_text())
                self.assertEqual(answer.citations[0].quote, passage.text)

    def test_compliance_keeps_signature_and_request_clock_separate(self):
        letter = [p for p in self.corpus.passages.values()
                  if self.corpus.documents[p.document_id].reference == "к-12"]
        for day, expected in ((20, "inapplicable"), (21, "applicable")):
            with self.subTest(day=day):
                question = (f"Запрос отправили 18 апреля 2025 года, договор подписывают {day} апреля. "
                            "Какие требования письма К-12 применимы?")
                scope = self.corpus.scope(question)
                self.assertEqual(scope.event_role, "signature")
                self.assertEqual(scope.date_ranges, ((date(2025, 4, day), date(2025, 4, day)),))
                self.assertEqual({self.corpus.applicability(p, scope) for p in letter}, {expected})
        clock = next(p for p in letter if "3 (трёх) рабочих дней" in p.text)
        self.assertIn("получения запроса", clock.text)


if __name__ == "__main__":
    unittest.main()
