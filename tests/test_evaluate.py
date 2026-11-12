import unittest
from pathlib import Path

from assistant import Answer, LegalAssistant
from corpus import load_corpus
from evaluate import gold_passages, grade_answer, passed, read_cases
from proxy import Settings


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_payment_terms_must_match_their_own_citations(self):
        case, = read_cases(ROOT / "eval" / "cases.json", ["supplier-payment-comparison"])
        corpus = load_corpus(ROOT / "data" / "knowledge-base.zip")
        assistant = LegalAssistant(corpus, Settings(), offline=True)
        slots = gold_passages(corpus, case["evidence"])
        scope = corpus.scope(case["question"])
        ids = [min(slot) for slot in slots]
        citations = [assistant.citation(corpus.passages[pid], corpus.passages[pid].text, scope) for pid in ids]
        first = "П-17/2024: 30 календарных дней с даты подписания УПД."
        second = "ПТ-05/2023: 45 календарных дней с даты поставки."
        correct = Answer("answered", f"{first} [{ids[0]}]\n\n{second} [{ids[1]}]", citations)
        swapped = Answer("answered", f"{first} [{ids[1]}]\n\n{second} [{ids[0]}]", citations)
        self.assertTrue(passed(grade_answer(correct, case, slots, corpus)))
        rejected = grade_answer(swapped, case, slots, corpus)
        self.assertFalse(rejected["missing_patterns"])
        self.assertFalse(rejected["missing_citations"])
        self.assertEqual(rejected["missing_assertions"], [0, 1])
        self.assertFalse(passed(rejected))
        bundled = Answer("answered", f"{first[:-1]}, а по {second.lower()} [{ids[0]}] [{ids[1]}]", citations)
        self.assertTrue(passed(grade_answer(bundled, case, slots, corpus)))
        crossed = Answer("answered",
                         f"П-17/2024: 45 календарных дней с даты поставки, "
                         f"а ПТ-05/2023: 30 календарных дней с даты подписания УПД. [{ids[0]}] [{ids[1]}]",
                         citations)
        self.assertEqual(grade_answer(crossed, case, slots, corpus)["missing_assertions"], [0, 1])
