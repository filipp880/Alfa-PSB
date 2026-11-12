from datetime import date
from pathlib import Path
import unittest

from corpus import load_corpus


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(Path(__file__).resolve().parents[1] / "data" / "knowledge-base.zip")

    def test_similar_organization_names_stay_distinct(self):
        corpus = self.corpus
        exact = corpus.scope("Какой гарантийный срок у ООО «Альфа-Трейд»?")
        typo = corpus.scope("Какой гарантийный срок у ООО Альфа Трйдинг?")
        ambiguous = corpus.scope("Какая гарантия у «Альфа-Трей»?")
        self.assertEqual(exact.candidate_entities, ("Альфа-Трейд",))
        self.assertEqual(typo.candidate_entities, ("Альфа-Трейдинг",))
        self.assertFalse(typo.exact_document_ids)
        self.assertFalse(typo.ambiguous_entities)
        self.assertEqual(set(ambiguous.ambiguous_entities), {"Альфа-Трейд", "Альфа-Трейдинг"})

    def test_typed_article_and_provision(self):
        corpus = self.corpus
        scope = corpus.scope("Сопоставьте ч. 2 ст. 13.11 КоАП РФ и ст. 9 Федерального закона 152-ФЗ.")
        self.assertEqual(len(scope.exact_document_ids), 2)
        self.assertFalse(scope.unresolved_references)
        result = corpus.search("Что предусмотрено ч. 2 ст. 13.11 КоАП РФ?", 1)[0]
        self.assertEqual(result.locator, "ст. 13.11, ч. 2")
        self.assertTrue(result.text.startswith("2. Обработка персональных данных"))
        self.assertEqual(corpus.scope("ст. 70 ГК РФ").unresolved_references, ("ст. 70 (ГК РФ)",))

    def test_unicode_dashes_preserve_contract_references(self):
        corpus = self.corpus
        for dash in "\u2010\u2011\u2012":
            with self.subTest(dash=dash):
                query = f"Назовите адрес поставки по П{dash}17/2024 для спецификации от 2 октября 2024."
                scope = corpus.scope(query)
                self.assertEqual(len(scope.exact_document_ids), 2)
                self.assertTrue(all(corpus.documents[doc_id].reference == "п-17/2024"
                                    for doc_id in scope.exact_document_ids))
                ranked = corpus.search(query, 4, scope=scope)
                self.assertIn("изменение п. 3.2", {p.locator for p in ranked})
                unknown = corpus.scope(f"Кто ремонтирует крышу по договору АР{dash}441/2022?")
                self.assertEqual(unknown.unresolved_references, ("ар-441/2022",))

    def test_scope_preserves_comparison_dates_and_sources(self):
        query = ("Сравните оплату у «Альфа-Трейд» на 28 февраля 2025 и у «Альфа-Трейдинг» на 1 марта 2025.")
        scope = self.corpus.scope(query)
        self.assertTrue(scope.comparison)
        self.assertEqual(len(scope.date_ranges), 2)
        self.assertEqual(len(scope.exact_document_ids), 2)
        self.assertFalse(scope.ambiguous_entities)
        hits = self.corpus.search(query, 6, scope=scope)
        self.assertTrue(set(scope.exact_document_ids) <= {p.document_id for p in hits})

    def test_amendment_boundary_is_strictly_after(self):
        corpus = self.corpus
        base, amendment = [next(p for p in corpus.passages.values() if p.locator == locator and
                                corpus.documents[p.document_id].reference == "п-17/2024")
                           for locator in ("п. 3.2", "изменение п. 3.2")]
        for day, base_status, amendment_status in (
            (1, "applicable", "inapplicable"),
            (2, "inapplicable", "applicable"),
        ):
            with self.subTest(day=day):
                scope = corpus.scope(f"Какой адрес по договору П-17/2024 для спецификации от {day} октября 2024?")
                self.assertEqual(scope.event_role, "specification")
                self.assertEqual(corpus.applicability(base, scope), base_status)
                self.assertEqual(corpus.applicability(amendment, scope), amendment_status)
        open_interval = corpus.scope(
            "Какой адрес по договору П-17/2024 для спецификаций, подписанных после 1 октября 2024?"
        )
        self.assertEqual(corpus.applicability(amendment, open_interval), "applicable")

    def test_event_date_ignores_document_date(self):
        scope = self.corpus.scope(
            "В договоре от 10 января 2025 согласовали условие. Допсоглашение подписали 2 марта 2025. "
            "Какая редакция действовала на дату подписания допсоглашения?"
        )
        self.assertEqual(scope.event_role, "amendment_signature")
        self.assertEqual(scope.date_ranges, ((date(2025, 3, 2), date(2025, 3, 2)),))

    def test_explicit_cross_reference_and_consecutive_list(self):
        corpus = self.corpus
        sanctions = next(p for p in corpus.passages.values() if p.locator == "ст. 114, п. 4")
        self.assertIn("ст. 112, п. 2", {p.locator for p in corpus.related(sanctions)})
        lead = next(p for p in corpus.passages.values() if p.text == "Испытание при приеме на работу не устанавливается для:")
        items = corpus.list_context(lead)
        target = next(p for p in items if "на срок до двух месяцев" in p.text)
        self.assertEqual(corpus.neighbors(target)[0], lead)
        self.assertTrue(all(p.document_id == lead.document_id for p in items))


if __name__ == "__main__":
    unittest.main()
