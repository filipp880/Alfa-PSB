from datetime import date
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from corpus import INDEX_FILENAME, build_prebuilt_index, date_mentions, legal_references, load_corpus


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

    def test_spaced_contract_references_preserve_the_year(self):
        corpus = self.corpus
        known = corpus.scope("Какой обеспечительный платёж по договору № А - 08 / 2024?")
        self.assertEqual([corpus.documents[key].reference for key in known.exact_document_ids], ["а-08/2024"])
        missing = corpus.scope("Есть ли договор А - 08 / 2022 и письмо № К - 900?")
        self.assertFalse(missing.exact_document_ids)
        self.assertEqual(missing.unresolved_references, ("а-08/2022", "к-900"))

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

    def test_specification_date_controls_the_amendment_despite_document_date(self):
        corpus = self.corpus
        base, amendment = [next(p for p in corpus.passages.values() if p.locator == locator and
                                corpus.documents[p.document_id].reference == "п-17/2024")
                           for locator in ("п. 3.2", "изменение п. 3.2")]
        for day in (1, 2):
            with self.subTest(day=day):
                scope = corpus.scope(
                    "Договор поставки № П - 17 / 2024 от 12.02.2024. "
                    f"Куда везти партию по спецификации от {day} октября 2024?"
                )
                self.assertEqual(scope.event_role, "specification")
                self.assertEqual(scope.date_ranges, ((date(2024, 10, day), date(2024, 10, day)),))
                self.assertFalse(scope.comparison)
                expected = ("applicable", "inapplicable") if day == 1 else ("inapplicable", "applicable")
                self.assertEqual(tuple(corpus.applicability(p, scope) for p in (base, amendment)), expected)

        document_only = corpus.scope("Что сказано в регламенте согласования договоров № 42 от 01.03.2025?")
        self.assertEqual(document_only.event_role, "document")

    def test_compliance_uses_contract_signature_not_request_date(self):
        corpus = self.corpus
        letter = [p for p in corpus.passages.values()
                  if corpus.documents[p.document_id].reference == "к-12"]
        self.assertTrue(letter)
        for query, day, expected in (
            ("Запрос отправили 18 апреля 2025, подписать договор планируют 21 апреля.", 21, "applicable"),
            ("Запрос отправили 18 апреля, договор подпишут 20 апреля 2025.", 20, "inapplicable"),
            ("18 апреля 2025 направили запрос. 21 апреля подпишут договор.", 21, "applicable"),
            ("Запрос отправили 18 апреля 2025. Дата подписания неизвестна.", 18, "unknown"),
        ):
            with self.subTest(query=query):
                scope = corpus.scope(query + " Что требует письмо К-12?")
                self.assertFalse(scope.comparison)
                self.assertEqual(scope.date_ranges, ((date(2025, 4, day), date(2025, 4, day)),))
                self.assertEqual({corpus.applicability(p, scope) for p in letter}, {expected})

    def test_comparison_with_omitted_years_and_delivery_date(self):
        corpus = self.corpus
        for query in (
            "Сравните адрес по П-17/2024 для спецификаций от 1 и 2 октября 2024.",
            "Обе партии мебели по П-17/2024 приехали 14 октября 2024 года. "
            "Спецификацию первой подписали 1 октября, второй — 2 октября. Куда везти каждую?",
        ):
            with self.subTest(query=query):
                scope = corpus.scope(query)
                self.assertTrue(scope.comparison)
                self.assertEqual(scope.event_role, "specification")
                self.assertEqual(scope.date_ranges, (
                    (date(2024, 10, 1), date(2024, 10, 1)),
                    (date(2024, 10, 2), date(2024, 10, 2)),
                ))

    def test_explicit_cross_reference_and_consecutive_list(self):
        corpus = self.corpus
        sanctions = next(p for p in corpus.passages.values() if p.locator == "ст. 114, п. 4")
        self.assertIn("ст. 112, п. 2", {p.locator for p in corpus.related(sanctions)})
        lead = next(p for p in corpus.passages.values() if p.text == "Испытание при приеме на работу не устанавливается для:")
        items = corpus.list_context(lead)
        target = next(p for p in items if "на срок до двух месяцев" in p.text)
        self.assertEqual(corpus.neighbors(target)[0], lead)
        self.assertTrue(all(p.document_id == lead.document_id for p in items))

    def test_number_first_contract_reference_and_missing_clause(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "knowledge-base.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("Договор поставки Ромашка.txt", "\n".join((
                    "ДОГОВОР ПОСТАВКИ № 12-Д", "ООО «Ромашка» — Поставщик.", "1. Сроки поставки",
                    "1.1. Срок — 3 рабочих дня.", "1.2. Поставка после предоплаты.",
                )))
                archive.writestr("Договор поставки Ромашка Плюс.txt", "\n".join((
                    "ДОГОВОР ПОСТАВКИ № 13-Д", "ООО «Ромашка Плюс» — Поставщик.", "1. Сроки поставки",
                    "1.1. Срок — 5 календарных дней.", "1.2. Поставка после подписания.",
                )))
            corpus = load_corpus(path)
        query = "Что сказано о сроке по п. 1.1 договора № 12-Д?"
        top = corpus.search(query, 1)[0]
        self.assertEqual(top.locator, "п. 1.1")
        self.assertIn("3 рабочих дня", top.text)
        self.assertEqual(corpus.scope(query).unresolved_references, ())
        self.assertEqual(corpus.scope("Что требует п. 9.9 договора № 12-Д?").unresolved_references, ("п. 9.9",))
        self.assertEqual(corpus.scope("Какой срок у ООО Ромашка Плюс?").candidate_entities, ("Ромашка Плюс",))
        self.assertEqual(corpus.scope("Сравните сроки Ромашки и Ромашки Плюс").candidate_entities,
                         ("Ромашка", "Ромашка Плюс"))

    def test_typed_references_are_pinned_for_context(self):
        corpus = self.corpus
        query = "Что содержит приложение № 1 письма К-12?"
        scope = corpus.scope(query)
        self.assertEqual(corpus.exact_passages(scope, corpus.search(query))[0].locator, "Приложение 1")
        self.assertIn("ред 2025", corpus.documents[corpus.scope("По приказу № 14-П").exact_document_ids[0]].title)
        self.assertEqual(legal_references("Примените подп. 2.8-1 и раздел IV приказа № 14-П"),
                         (("subpoint", "2.8-1"), ("section", "iv"), ("order", "14-п")))
        self.assertEqual([value for _, _, value in date_mentions("п. 5.3.12, редакция от 01.03.24")],
                         [date(2024, 3, 1)])

    def test_dialogue_disambiguates_ordinal_and_keeps_uncertain_candidates(self):
        history = [{"role": "user", "text": "Сравните договоры № П-17/2024 и № ПТ-05/2023."},
                   {"role": "assistant", "text": "Продолжим по договору А-08/2024."}]
        scope, enriched = self.corpus.dialogue_scope("А для второго договора?", history)
        self.assertEqual({self.corpus.documents[doc_id].reference for doc_id in scope.exact_document_ids}, {"пт-05/2023"})
        self.assertNotIn("П-17/2024", enriched)
        scope, _ = self.corpus.dialogue_scope("А по нему какой срок?", history)
        self.assertEqual({self.corpus.documents[doc_id].reference for doc_id in scope.document_ids},
                         {"п-17/2024", "пт-05/2023"})
        self.assertTrue(scope.ambiguous_documents)
        scope, enriched = self.corpus.dialogue_scope("А по нему какой срок?", [])
        self.assertFalse(scope.document_ids)
        self.assertEqual(enriched, "А по нему какой срок?")

    def test_dialogue_infers_date_without_reusing_the_later_value(self):
        history = [{"role": "user", "text": "Куда везти по П-17/2024 спецификацию от 2 октября 2024?"}]
        for question, start in (("А если дата была раньше?", date.min), ("А если 1 октября?", date(2024, 10, 1))):
            with self.subTest(question=question):
                scope, _ = self.corpus.dialogue_scope(question, history)
                self.assertEqual(scope.date_ranges, ((start, date(2024, 10, 1)),))
                self.assertEqual(scope.event_role, "specification")

    def test_current_and_historical_context_keep_distinct_text(self):
        old, current = [next(p for p in self.corpus.passages.values() if p.locator == "п. 3.1" and
                             f"ред {year}" in self.corpus.documents[p.document_id].title) for year in (2023, 2025)]
        self.assertNotEqual(self.corpus.context_key(old), self.corpus.context_key(current))
        scope = self.corpus.scope("Как оформляют согласование по п. 3.1?")
        self.assertFalse(self.corpus.include_in_context(old, scope))
        self.assertTrue(self.corpus.include_in_context(current, scope))
        scope = self.corpus.scope("Как оформляли согласование в прежней редакции п. 3.1?")
        self.assertTrue(self.corpus.include_in_context(old, scope))

    def test_addendum_changes_only_the_numbered_specification(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "knowledge-base.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("Договор поставки.txt", "Договор поставки № 12-Д\n1. Сроки\n1.1. Срок поставки — 5 дней.")
                archive.writestr("Доп соглашение № 1.txt", "Дополнительное соглашение № 1\nк договору № 12-Д\n"
                                  "Изложить пункт 1.1 договора в следующей редакции: срок — 3 рабочих дня.\n"
                                  "Изменение применяется только к спецификации № 7.")
                archive.writestr("Спецификация № 8.txt", "Спецификация № 8\nк договору № 12-Д\n1. Товар\n1.1. Мебель.")
            corpus = load_corpus(path)
        base, amendment = [next(p for p in corpus.passages.values() if p.locator == locator)
                           for locator in ("п. 1.1", "изменение п. 1.1")]
        for suffix, expected in (("по спецификации № 7", ("inapplicable", "applicable")),
                                 ("по спецификации № 8", ("applicable", "inapplicable")),
                                 ("", ("unknown", "unknown"))):
            with self.subTest(suffix=suffix):
                scope = corpus.scope(f"Какой срок по договору № 12-Д {suffix}?")
                self.assertEqual(tuple(corpus.applicability(p, scope) for p in (base, amendment)), expected)

    def test_prebuilt_index_replays_retrieval_and_rebuilds_after_changes(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "knowledge-base.zip"
            copyfile(Path(__file__).resolve().parents[1] / "data" / "knowledge-base.zip", path)
            index = path.with_name(INDEX_FILENAME)
            index.write_bytes(build_prebuilt_index(path))
            prebuilt = load_corpus(path, embedding_model="BAAI/bge-m3")
            self.assertTrue(prebuilt.prebuilt)
            self.assertEqual(prebuilt.documents, self.corpus.documents)
            self.assertEqual(prebuilt.passages, self.corpus.passages)
            query = "Что требуется по приложению № 1 письма К-12?"
            self.assertEqual(prebuilt.search(query), self.corpus.search(query))
            self.assertFalse(load_corpus(path, embedding_model="other-model").prebuilt)
            with ZipFile(path, "a") as archive:
                archive.writestr("приказ.txt", "ПРИКАЗ № 1\nДата\nУтверждено\nДействует с 01.01.2024.")
            self.assertFalse(load_corpus(path).prebuilt)
            index.write_bytes(build_prebuilt_index(path).replace("Утверждено".encode(), "Испорчено".encode()))
            self.assertFalse(load_corpus(path).prebuilt)


if __name__ == "__main__":
    unittest.main()
