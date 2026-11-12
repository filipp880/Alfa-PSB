"""Regressions for evidence roles and lossless, verified retry replacement.

The gateway is scripted: these tests prove orchestration, not model quality.
"""
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from assistant import LegalAssistant
from corpus import load_corpus
from proxy import GatewayError, Settings


ROOT = Path(__file__).resolve().parents[1]
FIRST = "Оригинал передают в юридический отдел в течение 3 рабочих дней с даты подписания."
SECOND = "Скан размещают в СЭД не позднее 1 рабочего дня после поступления оригинала в архив."
FIRST_GAP = "Нужно найти срок размещения скана и порядок выдачи оригинала."
LAST_GAP = "Не хватает подтверждения порядка выдачи оригинала сотруднику другого подразделения."
QUESTION = "Договор подписан 1 июня 2025 года. Когда передают оригинал, размещают скан и выдают оригинал другому подразделению?"
COMPARISON = "Как изменился порог по п. 2.3 на 2 марта 2025 года? Назовите прежний и новый порог."


def claim(text, *evidence):
    return {"text": text, "evidence": list(evidence)}


def draft(status, claims, gap="", look_for=""):
    return {"status": status, "claims": claims, "gap": gap, "look_for": look_for}


class Submission6ClaimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(ROOT / "data" / "knowledge-base.zip")

    def assistant(self, retry=True):
        return LegalAssistant(self.corpus, Settings(base_url="http://localhost", key="fixture",
                                                    model="fixture-no-model", retry_search=retry))

    def test_historical_composite_keeps_both_editions_for_verification(self):
        rag = self.assistant(False)
        text = "Порог вырос с суммы свыше 1 млн руб. в прежней редакции до суммы свыше 3 млн руб. в новой."
        seen = []

        def chat(system, payload, deadline, *, verify=False):
            if not verify:
                return draft("answered", [claim(text, "P78", "P90")])
            seen.append(payload)
            evidence = set(payload["claims"][0]["evidence"])
            return {"supported": [0] if evidence == {"P78", "P90"} else [], "gap_ok": False}

        with patch.object(rag.gateway, "chat", chat):
            result = rag.ask(COMPARISON)
        self.assertEqual(result.status, "answered")
        self.assertEqual({c.id for c in result.citations}, {"P78", "P90"})
        self.assertEqual(next(p["applicability"] for p in seen[0]["passages"] if p["id"] == "P78"),
                         "inapplicable")
        self.assertIn("1 млн", result.csv_text())
        self.assertIn("3 млн", result.csv_text())

    def test_separate_historical_claim_may_use_the_old_edition(self):
        rag = self.assistant(False)
        old = "В прежней редакции требовалось согласование финансового директора при сумме свыше 1 млн рублей."
        new = "На 2 марта 2025 года порог — свыше 3 млн рублей."
        with patch.object(rag.gateway, "chat", side_effect=[
            draft("answered", [claim(old, "P78"), claim(new, "P90")]),
            {"supported": [0, 1], "gap_ok": False},
        ]) as gateway:
            result = rag.ask(COMPARISON)
        self.assertEqual(result.status, "answered")
        self.assertEqual([c["evidence"] for c in gateway.call_args.args[1]["claims"]], [["P78"], ["P90"]])
        self.assertEqual({c.id for c in result.citations}, {"P78", "P90"})

    def test_historical_comparison_paraphrases_preserve_evidence_roles(self):
        for text in (
            "Порог повысился с суммы свыше 1 млн до суммы свыше 3 млн рублей.",
            "Порог был свыше 1 млн рублей, стал свыше 3 млн рублей.",
            "До 1 марта 2025 года порог составлял свыше 1 млн рублей, затем — свыше 3 млн рублей.",
        ):
            with self.subTest(text=text):
                rag = self.assistant(False)
                with patch.object(rag.gateway, "chat", side_effect=[
                    draft("answered", [claim(text, "P78", "P90")]),
                    {"supported": [0], "gap_ok": False},
                ]) as gateway:
                    result = rag.ask(COMPARISON)
                self.assertEqual(set(gateway.call_args.args[1]["claims"][0]["evidence"]), {"P78", "P90"})
                self.assertEqual({c.id for c in result.citations}, {"P78", "P90"})

    def test_old_edition_cannot_establish_a_current_requirement(self):
        for question, text in (
            (COMPARISON, "На 2 марта 2025 года согласование обязательно при сумме свыше 1 млн рублей."),
            ("Какой порог действует на 2 марта 2025 года?",
             "Раньше порог был 1 млн рублей, поэтому сейчас при сумме 2 млн согласование обязательно."),
            (COMPARISON, "Сейчас договор до 3 млн рублей требует согласования при сумме свыше 1 млн рублей."),
        ):
            with self.subTest(question=question):
                rag = self.assistant(False)
                with patch.object(rag.gateway, "chat", return_value=draft("answered", [claim(text, "P78")])) as gateway:
                    result = rag.ask(question)
                self.assertEqual(result.status, "insufficient")
                self.assertFalse(result.citations)
                self.assertEqual(gateway.call_count, 1)

    def test_mixed_historical_and_wrong_current_assertion_is_not_preapproved(self):
        rag = self.assistant(False)
        wrong = "Раньше порог был свыше 1 млн рублей и на 2 марта 2025 года он остался прежним."
        with patch.object(rag.gateway, "chat", side_effect=[
            draft("answered", [claim(wrong, "P78", "P90")]),
            {"supported": [], "gap_ok": False},
        ]) as gateway:
            result = rag.ask(COMPARISON)
        self.assertEqual(set(gateway.call_args.args[1]["claims"][0]["evidence"]), {"P78", "P90"})
        self.assertEqual(result.status, "insufficient")
        self.assertNotIn(wrong, result.text)

    def test_identical_claim_with_alternative_evidence_reaches_verification(self):
        rag = self.assistant(False)
        question = "Какой порог согласования с финансовым директором действует на 2 марта 2025 года?"
        text = "Согласование с финансовым директором требуется при сумме свыше 3 млн рублей."
        sources = [self.corpus.passages[pid] for pid in ("P91", "P90")]
        verified = []

        def chat(system, payload, deadline, *, verify=False):
            if not verify:
                # The first valid ID belongs to a different approval rule. It must not
                # suppress the same statement backed by the actual financial threshold.
                return draft("answered", [claim(text, "P91"), claim(text, "P90")])
            verified.extend(payload["claims"])
            supported = []
            for item in payload["claims"]:
                evidence = " ".join(self.corpus.passages[pid].text for pid in item["evidence"])
                if "свыше 3 000 000 (трёх миллионов) рублей дополнительно согласуются с финансовым директором" in evidence:
                    supported.append(item["index"])
            return {"supported": supported, "gap_ok": False}

        with patch.object(rag.gateway, "chat", chat):
            result, _ = rag._answer_once(question, [], sources, self.corpus.scope(question), time.monotonic() + 30)
        self.assertEqual([item["evidence"] for item in verified], [["P91"], ["P90"]])
        self.assertIn(text, result.text)
        self.assertEqual([citation.id for citation in result.citations], ["P90"])
        self.assertNotEqual(result.status, "insufficient")

    def replay_retry(self, second_draft, *, supported=None, previous_supported=None,
                     previous_covered=None, adds_information=True, consistent=True,
                     complete_review=True, second_error=None, review_error=None):
        rag = self.assistant()
        # A prior context clause that is not cited must also survive into retry review.
        initial = [self.corpus.passages[pid] for pid in ("P94", "P95")]
        extra = self.corpus.passages["P104"]
        prompts = []
        generation = 0
        checks = 0

        def chat(system, payload, deadline, *, verify=False):
            nonlocal generation, checks
            prompts.append((verify, payload))
            if not verify:
                generation += 1
                if generation == 1:
                    return draft("insufficient", [claim(FIRST, "P94")], FIRST_GAP, "размещение скана в СЭД")
                if second_error:
                    raise second_error
                return second_draft
            checks += 1
            if checks == 1:
                return {"supported": [0], "gap_ok": True}
            if review_error:
                raise review_error
            result = {"supported": supported if supported is not None else list(range(len(payload["claims"]))),
                      "gap_ok": True}
            if complete_review:
                result.update(previous_supported=[0] if previous_supported is None else previous_supported,
                              previous_covered=[0] if previous_covered is None else previous_covered,
                              adds_information=adds_information, consistent=consistent)
            return result

        with patch.object(rag, "retrieve", return_value=initial), \
                patch.object(rag, "context", side_effect=[initial, [extra]]), \
                patch.object(self.corpus, "search", return_value=[extra]), \
                patch.object(rag.gateway, "chat", chat):
            result = rag.ask(QUESTION, deadline=time.monotonic() + 80)
        return result, prompts

    def test_partial_retry_retains_confirmed_facts_and_specific_gap(self):
        result, prompts = self.replay_retry(draft("partial", [claim(FIRST, "P94"), claim(SECOND, "P104")], LAST_GAP))
        self.assertEqual(result.status, "partial")
        self.assertIn(FIRST, result.text)
        self.assertIn(SECOND, result.text)
        self.assertIn(LAST_GAP, result.text)
        self.assertNotIn(FIRST_GAP, result.text)
        self.assertEqual({c.id for c in result.citations}, {"P94", "P104"})
        first, second = prompts[0][1], prompts[2][1]
        self.assertEqual(first["anchors"], second["anchors"])
        self.assertEqual({p["id"] for p in second["passages"]}, {"P94", "P95", "P104"})
        self.assertEqual(second["previous_claims"], [{"index": 0, "text": FIRST, "evidence": ["P94"]}])
        self.assertEqual(prompts[3][1]["previous_claims"], second["previous_claims"])

    def test_compound_retry_can_add_information_without_more_claims(self):
        combined = "Оригинал передают в юротдел за 3 рабочих дня от подписания; скан размещают в СЭД за 1 рабочий день после поступления оригинала в архив."
        result, _ = self.replay_retry(draft("partial", [claim(combined, "P94", "P104")], LAST_GAP))
        self.assertIn(combined, result.text)
        self.assertEqual(len(result.claims), 1)
        self.assertEqual({c.id for c in result.citations}, {"P94", "P104"})

    def test_duplicates_and_paraphrases_are_not_information_gain(self):
        variants = [
            (draft("partial", [claim(FIRST, "P94"), claim(FIRST, "P94", "P104")], LAST_GAP), True),
            (draft("partial", [claim("Передача оригинала в юридический отдел занимает 3 рабочих дня от подписания.", "P94")], LAST_GAP), False),
        ]
        for candidate, adds in variants:
            with self.subTest(candidate=candidate):
                result, _ = self.replay_retry(candidate, adds_information=adds)
                self.assertEqual(result.text, f"{FIRST} [P94]\n\n{FIRST_GAP}")

    def test_answered_retry_cannot_drop_an_existing_confirmed_fact(self):
        result, _ = self.replay_retry(draft("answered", [claim(SECOND, "P104")]), previous_covered=[])
        self.assertEqual(result.status, "partial")
        self.assertIn(FIRST, result.text)
        self.assertNotIn(SECOND, result.text)
        self.assertIn(FIRST_GAP, result.text)

    def test_rejected_or_unknown_evidence_is_not_a_verified_supplement(self):
        for evidence, supported in (("P999999", [0]), ("P104", [0])):
            with self.subTest(evidence=evidence):
                result, _ = self.replay_retry(
                    draft("partial", [claim(FIRST, "P94"), claim(SECOND, evidence)], LAST_GAP), supported=supported)
                self.assertEqual([c.id for c in result.citations], ["P94"])
                self.assertNotIn(SECOND, result.text)

    def test_contradiction_or_changed_event_cannot_replace_previous_answer(self):
        for changed in (
            "Скан размещают через 5 рабочих дней после поступления оригинала.",
            "Для договора, подписанного 2 марта 2024 года, скан размещают за 1 рабочий день.",
        ):
            with self.subTest(changed=changed):
                result, prompts = self.replay_retry(
                    draft("partial", [claim(FIRST, "P94"), claim(changed, "P104")], LAST_GAP), consistent=False)
                self.assertNotIn(changed, result.text)
                self.assertIn(FIRST, result.text)
                self.assertEqual(prompts[0][1]["anchors"], prompts[2][1]["anchors"])

    def test_retry_failure_or_missing_relation_review_preserves_checked_answer(self):
        for options in (
            {"second_error": GatewayError("/chat/completions", 503)},
            {"review_error": GatewayError("/chat/completions", 503)},
            {"complete_review": False},
            {"previous_supported": [True]},
            {"previous_covered": [8]},
        ):
            with self.subTest(options=options):
                result, _ = self.replay_retry(
                    draft("partial", [claim(FIRST, "P94"), claim(SECOND, "P104")], LAST_GAP), **options)
                self.assertEqual([c.id for c in result.citations], ["P94"])
                self.assertIn(FIRST_GAP, result.text)

    def test_new_exception_can_revoke_old_claim_even_without_a_replacement(self):
        result, _ = self.replay_retry(
            draft("partial", [claim(FIRST, "P94")], LAST_GAP), supported=[],
            previous_supported=[], previous_covered=[], adds_information=False)
        self.assertEqual(result.status, "insufficient")
        self.assertFalse(result.citations)
        self.assertNotIn(FIRST, result.text)
        self.assertIn("Повторная проверка", result.text)

    def test_new_context_can_correct_a_rejected_old_claim(self):
        result, _ = self.replay_retry(
            draft("partial", [claim(SECOND, "P104")], LAST_GAP), previous_supported=[], previous_covered=[])
        self.assertEqual(result.status, "partial")
        self.assertEqual([c.id for c in result.citations], ["P104"])
        self.assertNotIn(FIRST, result.text)

    def test_scope_mismatch_is_rejected_even_with_positive_model_review(self):
        result, _ = self.replay_retry(draft("partial", [claim(FIRST, "P94"), claim(SECOND, "P104")], LAST_GAP))
        changed = replace(result, event_ranges=(("2024-03-02", "2024-03-02"),))
        self.assertIs(self.assistant()._select_retry(result, changed), result)

    def test_out_of_range_verifier_indices_fail_closed(self):
        rag = self.assistant(False)
        with patch.object(rag.gateway, "chat", side_effect=[
            draft("answered", [claim("Порог — свыше 3 млн рублей.", "P90")]),
            {"supported": [0, 99], "gap_ok": False},
        ]):
            result = rag.ask(COMPARISON)
        self.assertEqual(result.status, "error")
        self.assertFalse(result.citations)


if __name__ == "__main__":
    unittest.main()
