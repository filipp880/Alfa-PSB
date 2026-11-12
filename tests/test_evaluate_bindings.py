import re
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from assistant import Answer, LegalAssistant
from corpus import load_corpus
from evaluate import evaluate_live_case, gold_passages, grade_answer, mutated_kb, passed, read_cases
from proxy import Settings


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "knowledge-base.zip"


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(KB)
        cls.assistant = LegalAssistant(cls.corpus, Settings(), offline=True)

    def fixture(self, case_id, filename="cases.json"):
        case, = read_cases(ROOT / "eval" / filename, [case_id])
        slots = gold_passages(self.corpus, case["evidence"])
        ids = [min(slot) for slot in slots]
        citations = [self.citation(pid, case["question"]) for pid in dict.fromkeys(ids)]
        return case, slots, ids, citations

    def citation(self, pid, question):
        p = self.corpus.passages[pid]
        return self.assistant.citation(p, p.text, self.corpus.scope(question))

    def test_contracts_and_sources_bind_inside_a_single_paragraph(self):
        case, slots, (first_id, second_id), citations = self.fixture("supplier-payment-comparison")
        first = "П-17/2024: 30 календарных дней с даты подписания УПД."
        second = "ПТ-05/2023: 45 календарных дней с даты поставки."
        sources = f"[{first_id}] [{second_id}]"
        good = (
            f"{first} {second} {sources}",
            f"{first} [{first_id}] {second} [{second_id}]",
            f"30 календарных дней с даты подписания УПД по П-17/2024. "
            f"45 календарных дней с даты поставки по ПТ-05/2023. {sources}",
        )
        bad = (
            f"П-17/2024: 45 календарных дней с даты поставки. "
            f"ПТ-05/2023: 30 календарных дней с даты подписания УПД. {sources}",
            f"П-17/2024: 45 календарных дней с даты поставки а "
            f"ПТ-05/2023: 30 календарных дней с даты подписания УПД. {sources}",
            f"П-17/2024: 30 календарных дней с даты поставки; "
            f"ПТ-05/2023: 45 календарных дней с даты подписания УПД. {sources}",
            f"{first} [{second_id}] {second} [{first_id}]",
        )
        for expected, bodies in ((True, good), (False, bad)):
            for body in bodies:
                with self.subTest(body=body):
                    grade = grade_answer(Answer("answered", body, citations), case, slots, self.corpus)
                    self.assertFalse(grade["missing_patterns"])
                    self.assertFalse(grade["missing_citations"])
                    self.assertEqual(passed(grade), expected)
        wrong_unit = Answer("answered", f"{first.replace('дней', 'недель')} {second} {sources}", citations)
        self.assertEqual(grade_answer(wrong_unit, case, slots, self.corpus)["missing_assertions"], [0])

    def test_address_requires_the_house_and_correct_csv_citation(self):
        case, slots, ids, citations = self.fixture("delivery-amendment-after")
        marker = f"[{ids[0]}]"
        complete = Answer("answered", f"Химки, ул. Заводская, д. 15, склад № 3. {marker}", citations)
        missing = replace(complete, text=complete.text.replace("д. 15, ", ""))
        wrong_locator = replace(complete, citations=[replace(citations[0], locator="п. 0.0")])
        self.assertTrue(passed(grade_answer(complete, case, slots, self.corpus)))
        self.assertTrue(grade_answer(missing, case, slots, self.corpus)["missing_patterns"])
        self.assertEqual(grade_answer(wrong_locator, case, slots, self.corpus)["invalid_citations"], 1)
        self.assertIn("д. 15", complete.csv_text())

    def test_workflow_requires_the_clock_for_each_action(self):
        case, slots, (transfer, issue, scan), citations = self.fixture("holdout-archive-three-clocks", "holdout.json")
        correct = (
            f"Оригинал передают в юротдел в течение 3 рабочих дней со дня подписания. [{transfer}] "
            f"Скан размещают в СЭД в течение 1 рабочего дня после поступления оригинала. [{scan}] "
            f"Оригинал выдают по заявке в СЭД на срок не более 3 рабочих дней. [{issue}]"
        )
        swapped = correct.replace("3 рабочих дней со дня подписания", "1 рабочего дня со дня подписания")
        swapped = swapped.replace("1 рабочего дня после поступления", "3 рабочих дней после поступления")
        self.assertTrue(passed(grade_answer(Answer("answered", correct, citations), case, slots, self.corpus)))
        grade = grade_answer(Answer("answered", swapped, citations), case, slots, self.corpus)
        self.assertFalse(grade["missing_patterns"])
        self.assertEqual(grade["missing_assertions"], [0, 1])

    def test_correct_calculation_with_the_wrong_clause_is_rejected(self):
        case, slots, (penalty, limit, price), citations = self.fixture("holdout-liability-dual-limit", "holdout.json")
        formula = (
            "Неустойка: 400 000 × 0,1% × 120 = 48 000 руб.; с пределом 10% неоплаченной суммы — 40 000 руб."
        )
        cap = f"Общий предел — 50% от цены договора, 4 200 000 руб. [{limit}] [{price}]"
        right = Answer("answered", f"{formula} [{penalty}]\n\n{cap}", citations)
        wrong_pid = next(p.id for p in self.corpus.passages.values()
                         if p.locator == "п. 7.1" and p.document_id == self.corpus.passages[penalty].document_id)
        wrong = Answer("answered", f"{formula} [{wrong_pid}]\n\n{cap}",
                       [self.citation(wrong_pid, case["question"]), *citations[1:]])
        self.assertTrue(passed(grade_answer(right, case, slots, self.corpus)))
        grade = grade_answer(wrong, case, slots, self.corpus)
        self.assertFalse(grade["missing_patterns"])
        self.assertEqual(grade["missing_citations"], [0])
        self.assertEqual(grade["missing_assertions"], [0, 1])

    def test_archive_variant_preserves_original_and_changes_rate_and_cap_operands(self):
        case, _, (penalty, _, _), _ = self.fixture("holdout-liability-dual-limit", "holdout.json")
        original = self.corpus.passages[penalty]
        mutation = {
            "member": self.corpus.documents[original.document_id].filename,
            "old": original.text,
            "new": original.text.replace("0,1%", "0,2%").replace("10%", "8%"),
        }
        before = KB.read_bytes()
        with mutated_kb(KB, mutation) as archive:
            changed = load_corpus(archive)
            evidence = [{**case["evidence"][0], "quote": mutation["new"]}]
            slot, = gold_passages(changed, evidence)
            operand_source = changed.passages[min(slot)].text
            operands = re.findall(r"(\d+(?:,\d+)?)% от неоплаченной суммы", operand_source)
            self.assertEqual(operands, ["0,2", "8"])
            rate, limit = (Decimal(value.replace(",", ".")) / 100 for value in operands)
            for base, days, expected in [(300_000, 30, 18_000), (350_000, 50, 28_000)]:
                with self.subTest(base=base, days=days):
                    self.assertEqual(min(base * days * rate, base * limit), Decimal(expected))
        self.assertFalse(archive.exists())
        self.assertEqual(KB.read_bytes(), before)

    def test_reference_audit_observes_passages_with_retry_disabled(self):
        case, slots, _, _ = self.fixture("delivery-amendment-boundary")
        assistant = LegalAssistant(self.corpus, Settings(base_url="http://127.0.0.1", key="fixture",
                                                        model="controlled", retry_search=True))

        def draft(*args, **kwargs):
            self.assertFalse(assistant.settings.retry_search)
            return {"status": "insufficient", "claims": [], "look_for": "пункт о новом адресе"}

        with patch.object(assistant.gateway, "chat", side_effect=draft):
            result = evaluate_live_case(assistant, case, slots, [5, 10], 30, audit=True, reference_context=True)
        self.assertTrue(assistant.settings.retry_search)
        self.assertFalse(result["additional_search"])
        self.assertEqual(result["status"], "insufficient")
        stage, = result["audit"]["stages"]
        self.assertTrue(stage["anchors"]["event_ranges"])
        self.assertEqual({entry["id"] for entry in stage["passages"]}, {min(slot) for slot in slots})
        self.assertTrue(all(isinstance(entry["text_visible"], bool) for entry in stage["passages"]))
        self.assertTrue(result["audit"]["ranked_ids"][0])


if __name__ == "__main__":
    unittest.main()
