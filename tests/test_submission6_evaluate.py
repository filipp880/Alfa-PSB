"""Regression checks for visible evidence and selective-answer reporting."""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant import Answer, LegalAssistant
from corpus import load_corpus
from evaluate import gold_passages, grade_answer, passed, read_cases
from proxy import Settings


ROOT = Path(__file__).resolve().parents[1]


class Submission6EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(ROOT / "data/knowledge-base.zip")
        cls.assistant = LegalAssistant(cls.corpus, Settings(), offline=True)

    def fixture(self, case_id, filename="cases.json"):
        case, = read_cases(ROOT / "eval" / filename, [case_id])
        slots = gold_passages(self.corpus, case["evidence"])
        ids = [min(slot) for slot in slots]
        citations = [self.assistant.citation(self.corpus.passages[pid], self.corpus.passages[pid].text,
                                            self.corpus.scope(case["question"]))
                     for pid in dict.fromkeys(ids)]
        return case, slots, ids, citations

    def test_metadata_without_visible_source_does_not_support_csv(self):
        case, slots, ids, citations = self.fixture("delivery-amendment-after")
        text = "Московская область, г. Химки, ул. Заводская, д. 15, склад № 3."
        missing = Answer("answered", text, citations)
        self.assertNotIn(citations[0].title, missing.csv_text())
        grade = grade_answer(missing, case, slots, self.corpus)
        self.assertFalse(passed(grade))
        self.assertEqual(grade["missing_citations"], list(range(len(slots))))
        visible = replace(missing, text=f"{text} [{ids[0]}]")
        self.assertTrue(passed(grade_answer(visible, case, slots, self.corpus)))
        self.assertIn(citations[0].title, visible.csv_text())
        self.assertIn(citations[0].locator, visible.csv_text())

    def test_dangling_source_marker_is_rejected_even_with_metadata(self):
        case, slots, ids, citations = self.fixture("delivery-amendment-after")
        answer = Answer("answered", "Московская область, г. Химки, ул. Заводская, д. 15, склад № 3. [P999999]", citations)
        grade = grade_answer(answer, case, slots, self.corpus)
        self.assertFalse(passed(grade))
        self.assertEqual(grade["invalid_citations"], 1)
        self.assertTrue(grade["missing_citations"])
        self.assertNotIn(citations[0].title, answer.csv_text())

    def test_full_address_includes_region_and_all_building_components(self):
        case, slots, ids, citations = self.fixture("delivery-amendment-after")
        text = f"Московская область, г. Химки, ул. Заводская, д. 15, склад № 3. [{ids[0]}]"
        self.assertTrue(passed(grade_answer(Answer("answered", text, citations), case, slots, self.corpus)))
        for removed in ("Московская область, ", "г. Химки, ", "д. 15, ", "склад № 3"):
            with self.subTest(removed=removed):
                grade = grade_answer(Answer("answered", text.replace(removed, ""), citations), case, slots, self.corpus)
                self.assertFalse(passed(grade))

    def test_wrong_arithmetic_does_not_pass_with_correct_operands_and_source(self):
        case, slots, (penalty, limit, price), citations = self.fixture("holdout-liability-dual-limit", "holdout.json")
        prefix = "Ставка — 0,1% от неоплаченной суммы за каждый день, предел 10%. "
        suffix = f"Общий предел — 50% от цены договора, 4 200 000 руб. [{limit}] [{price}]"
        correct = Answer("answered", f"{prefix} Неустойка — 40 000 руб. [{penalty}]\n\n{suffix}", citations)
        self.assertTrue(passed(grade_answer(correct, case, slots, self.corpus)))
        wrong = replace(correct, text=correct.text.replace("40 000", "48 000"))
        grade = grade_answer(wrong, case, slots, self.corpus)
        self.assertFalse(grade["missing_citations"])
        self.assertFalse(passed(grade))
        self.assertIn(0, grade["missing_assertions"])

    def test_selective_metrics_distinguish_coverage_fixture_failures_and_refusals(self):
        from evaluate import selective_metrics
        base = dict(status_ok=True, missing_patterns=[], forbidden_patterns=[], missing_assertions=[],
                    missing_citations=[], invalid_citations=0, unwanted_no_answer=False, expected_substantive=True)
        grades = [dict(base, status="answered"), dict(base, status="partial", missing_patterns=["required"]),
                  dict(base, status="insufficient", status_ok=False, unwanted_no_answer=True),
                  dict(base, status="clarification", expected_substantive=False),
                  dict(base, status="error", status_ok=False)]
        result = selective_metrics(grades)
        self.assertEqual(result["live_substantive_answers"], 2)
        self.assertEqual(result["live_answer_coverage"], 0.4)
        self.assertEqual(result["live_substantive_fixture_failures"], 1)
        self.assertEqual(result["live_substantive_fixture_failure_rate"], 0.5)
        self.assertEqual(result["live_answerable_cases"], 4)
        self.assertEqual(result["live_answerable_unwanted_no_answer"], 1)
        self.assertEqual(result["live_answerable_unwanted_no_answer_rate"], 0.25)
        self.assertIsNone(selective_metrics([])["live_answer_coverage"])
        self.assertIsNone(selective_metrics([])["live_substantive_fixture_failure_rate"])

    def test_malformed_fixture_schema_is_rejected_before_grading(self):
        case, _, _, _ = self.fixture("delivery-amendment-after")
        variants = [
            dict(case, statuses="answered"), dict(case, statuses=[]), dict(case, statuses=["answer"]),
            dict(case, cite=[True]), dict(case, must="химки"),
            dict(case, assertions=[{"must": ["химки"], "cite": [False]}]),
            dict(case, assertions=[{"must": ["химки"], "cite": [0], "same_clause": "yes"}]),
            dict(case, evidence=[{"file": "", "quote": "text"}]),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            for value in [variants[0], *[[item] for item in variants]]:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_cases(path, [])

    def test_zero_live_cases_are_unknown_not_zero_error_rate(self):
        from evaluate import metrics, print_result
        row = dict(context=[True], ranked={"5": [True], "10": [True]}, index=[True],
                   context_size=1, context_chars=50, search_ms=1)
        run = dict(cases=[row], live_preparation=None, live_requests=None)
        run["metrics"] = metrics(run, [5, 10])
        self.assertEqual(run["metrics"]["live_cases"], 0)
        for key in ("live_answer_coverage", "live_substantive_fixture_failure_rate",
                    "live_answerable_unwanted_no_answer_rate"):
            self.assertIsNone(run["metrics"][key])
        output = io.StringIO()
        with redirect_stdout(output):
            print_result("Offline", run, [], SimpleNamespace(k=[5, 10], live=True, details=False))
        self.assertIn("Прогон API пропущен", output.getvalue())
        self.assertNotIn("Доля содержательных ответов", output.getvalue())
        self.assertNotIn("0.0%", output.getvalue().replace("100.0%", ""))

    def test_retry_audit_keeps_filtered_claims_review_and_final_selection(self):
        from evaluate import evaluate_live_case
        rag = LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture-secret",
                                                   model="controlled", retry_search=True))
        first = "Оригинал передают в течение 3 рабочих дней с даты подписания."
        second = "Скан размещают не позднее 1 рабочего дня после поступления оригинала в архив."
        old_gap, new_gap = "Не найдены срок скана и порядок выдачи.", "Не найден порядок выдачи."
        initial = [self.corpus.passages["P94"]]
        extra = self.corpus.passages["P104"]
        case = dict(question="Договор подписан 1 июня 2025 года. Когда передают оригинал, размещают скан и выдают оригинал?",
                    statuses=["partial"], must=["3 рабочих", "1 рабочего"], evidence=[], cite=[])
        outputs = [
            dict(status="insufficient", claims=[dict(text=first, evidence=["P94"])], gap=old_gap, look_for="скан в СЭД"),
            dict(supported=[0], gap_ok=True),
            dict(status="partial", claims=[dict(text=first, evidence=["P94"]), dict(text=second, evidence=["P104"])],
                 gap=new_gap, look_for="порядок выдачи"),
            dict(supported=[0, 1], gap_ok=True, previous_supported=[0], previous_covered=[0],
                 adds_information=True, consistent=True),
        ]
        with patch.object(rag, "retrieve", return_value=initial), \
                patch.object(rag, "context", side_effect=[initial, [extra]]), \
                patch.object(self.corpus, "search", return_value=[extra]), \
                patch.object(rag.gateway, "chat", side_effect=outputs):
            result = evaluate_live_case(rag, case, [], [5, 10], 80, audit=True)
        audit = result["audit"]
        verify = audit["stages"][-1]
        self.assertEqual(verify["stage"], "verify")
        self.assertEqual(verify["input"]["claims"][1]["evidence"], ["P104"])
        self.assertEqual(verify["input"]["previous_claims"][0]["evidence"], ["P94"])
        self.assertEqual(verify["input"]["previous_gap"], old_gap)
        self.assertEqual(verify["anchors"], audit["stages"][0]["anchors"])
        self.assertEqual(verify["output"]["previous_covered"], [0])
        self.assertTrue(verify["output"]["adds_information"])
        self.assertTrue(verify["output"]["consistent"])
        self.assertEqual(audit["retry_decision"]["selected"], "accepted_candidate")
        self.assertEqual(audit["attempts"][-1]["citation_ids"], ["P94", "P104"])
        self.assertNotIn("fixture-secret", json.dumps(audit))


if __name__ == "__main__":
    unittest.main()
