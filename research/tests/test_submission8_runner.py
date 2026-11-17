import csv
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from assistant import Answer
from proxy import Gateway, Settings
from solution import FALLBACK, run_questions, validate_submission, write_submission


class SubmissionRunnerTests(unittest.TestCase):
    def test_csv_contract_and_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.csv"
            questions = [('ID,"1"', "Вопрос"), ("2", "Вопрос")]
            write_submission(path, questions, [" ", "ю" * 10_001])
            validate_submission(path, questions)
            with path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.reader(stream)), [
                    ["id", "answer"], ['ID,"1"', FALLBACK], ["2", "ю" * 10_000],
                ])
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_submission(path, [questions[0], questions[0]], ["A", "B"])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                validate_submission(path, questions[:1])

    def test_timeout_and_failure_leave_verified_checkpoint_and_all_ids(self):
        stop = Event()

        def ask(question, *, deadline, soft_deadline, on_progress):
            self.assertLessEqual(soft_deadline, deadline)
            if question == "slow":
                on_progress(Answer("partial", "Подтверждённый срок.", []))
                stop.wait(5)
                return Answer("answered", "Слишком поздний результат.", [])
            if question == "broken":
                raise RuntimeError()
            return Answer("answered", "Ответ следующего вопроса.", [])

        questions = [("a", "slow"), ("b", "broken"), ("c", "next")]
        gateway = Gateway(Settings(workers=1))
        assistant = SimpleNamespace(gateway=gateway, ask=ask)
        answers = [FALLBACK] * len(questions)
        try:
            with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {
                "PSB_QUESTION_SOFT_SECONDS": "1", "PSB_QUESTION_HARD_SECONDS": "1",
            }):
                path = Path(folder) / "submission.csv"
                started = time.monotonic()
                run_questions(assistant, questions, answers, path, started, started + 3)
                validate_submission(path, questions)
            self.assertEqual(answers, ["Подтверждённый срок.", FALLBACK, "Ответ следующего вопроса."])
        finally:
            stop.set()

    def test_explicitly_revoked_checkpoint_does_not_return_after_failure(self):
        def ask(_, *, deadline, soft_deadline, on_progress):
            on_progress(Answer("partial", "Первый результат.", []))
            on_progress(Answer("insufficient", FALLBACK, []))
            return Answer("error", "Ошибка.", [])

        gateway = Gateway(Settings(workers=1))
        assistant = SimpleNamespace(gateway=gateway, ask=ask)
        with tempfile.TemporaryDirectory() as folder:
            started = time.monotonic()
            path = Path(folder) / "submission.csv"
            answers = [FALLBACK]
            run_questions(assistant, [("id", "вопрос")], answers, path, started, started + 3)
            self.assertEqual(answers, [FALLBACK])
