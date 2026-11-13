"""Hackathon runner: read independent questions and write a complete submission.csv."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from assistant import Answer, load_assistant
from proxy import env

LOG = logging.getLogger(__name__)
ARCHIVE_NAMES = ("knowledge-base.zip", "kb.zip", "knowledge_base.zip")
FALLBACK = "В предоставленной базе недостаточно данных для подтверждения ответа."
RUN_BUDGET = 1130


def find_kb(directory: Path) -> Path:
    for name in ARCHIVE_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"В {directory} не найден архив базы знаний: {', '.join(ARCHIVE_NAMES)}")


def read_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("Во входном CSV нужны столбцы id и question")
        seen = set()
        questions = []
        for row in reader:
            qid = row["id"]
            if qid is None or not qid.strip() or qid in seen or row["question"] is None:
                raise ValueError(f"Отсутствует вопрос или ID, либо ID повторяется (строка {reader.line_num})")
            seen.add(qid)
            questions.append((qid, row["question"]))
    return questions


def validate_submission(path: Path, questions: list[tuple[str, str]]) -> None:
    with path.open(encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ["id", "answer"]:
            raise ValueError("Неверные столбцы submission.csv")
        seen = set()
        for row, (qid, _) in zip(reader, questions, strict=True):
            if (len(row) != 2 or row[0] != qid or qid in seen
                    or not row[1].strip() or len(row[1]) > 10_000):
                raise ValueError(f"Неверная строка submission.csv ({reader.line_num})")
            seen.add(qid)


def write_submission(path: Path, questions: list[tuple[str, str]], answers: list[str]) -> None:
    tmp = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", newline="", delete=False)
    staged = Path(tmp.name)
    try:
        with tmp:
            writer = csv.writer(tmp, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(("id", "answer"))
            for (qid, _), answer in zip(questions, answers, strict=True):
                writer.writerow((qid, (answer.strip() or FALLBACK)[:10_000]))
            tmp.flush()
            os.fsync(tmp.fileno())
        validate_submission(staged, questions)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def prepare(kb: Path, deadline: float):
    result = Queue(maxsize=1)

    def load() -> None:
        try:
            result.put(load_assistant(kb, deadline=deadline))
        except Exception as exc:
            result.put(exc)

    Thread(target=load, daemon=True, name="psb-prepare").start()
    try:
        assistant = result.get(timeout=max(0, deadline - time.monotonic()))
    except Empty:
        LOG.warning("Подготовка базы превысила бюджет")
        return None
    if isinstance(assistant, Exception):
        LOG.warning("Сбой подготовки базы (%s)", type(assistant).__name__)
        return None
    return assistant


def run_questions(assistant, questions: list[tuple[str, str]], answers: list[str],
                  output: Path, started: float, deadline: float) -> None:
    completed = Queue()
    # Only the runner mutates CSV state. Expired workers cannot overwrite saved results.
    active: dict[int, float] = {}
    progress: dict[int, Answer] = {}
    statuses: Counter[str] = Counter()
    durations = []
    next_position = dirty = 0
    last_write = time.monotonic()

    def process(position: int, question: str) -> None:
        before = time.monotonic()
        try:
            answer = assistant.ask(question, deadline=deadline,
                                   on_progress=lambda value: completed.put((position, value, None)))
        except Exception as exc:
            LOG.warning("Ошибка обработки строки %d (%s)", position + 1, type(exc).__name__)
            answer = None
        completed.put((position, answer, time.monotonic() - before))

    def record(position: int, answer: Answer | None, duration: float, *, expired: bool = False) -> None:
        best = progress.pop(position, None)
        if answer is None or answer.status == "error":
            answer = best or answer
        if answer is not None:
            answers[position] = answer.csv_text() if answer.status != "error" else FALLBACK
        statuses["timeout" if expired else answer.status if answer is not None else "error"] += 1
        durations.append(duration)

    def apply_checkpoint(position: int, answer: Answer | None, duration: float | None) -> None:
        nonlocal dirty
        if position not in active:
            return
        if duration is not None:
            active.pop(position)
            record(position, answer, duration)
        elif answer is not None and answer.status != "error":
            progress[position] = answer
            answers[position] = answer.csv_text() or FALLBACK
        else:
            return
        dirty += 1

    try:
        while active or next_position < len(questions):
            now = time.monotonic()
            if now >= deadline:
                break
            while next_position < len(questions) and len(active) < assistant.settings.workers:
                active[next_position] = now
                Thread(target=process, args=(next_position, questions[next_position][1]),
                       daemon=True, name=f"psb-question-{next_position + 1}").start()
                next_position += 1
            wake = min(deadline, now + 1)
            try:
                apply_checkpoint(*completed.get(timeout=max(0, wake - time.monotonic())))
                # Consume available checkpoints before starting the next batch.
                while True:
                    apply_checkpoint(*completed.get_nowait())
            except Empty:
                pass
            now = time.monotonic()
            if dirty >= 8 or dirty and now - last_write >= 2:
                write_submission(output, questions, answers)
                dirty = 0
                last_write = time.monotonic()
    finally:
        try:
            while True:
                apply_checkpoint(*completed.get_nowait())
        except Empty:
            pass
        for position, began in active.items():
            record(position, None, time.monotonic() - began, expired=True)
        write_submission(output, questions, answers)
        validate_submission(output, questions)
    statuses["not_started"] = len(questions) - next_position
    LOG.info("Время %.1f с, статусы %s, запросы %s", time.monotonic() - started,
             dict(statuses), assistant.gateway.request_counts())
    if durations:
        p95 = sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
        LOG.info("Время обработки вопроса, среднее %.2f с, p50 %.2f с, p95 %.2f с",
                 statistics.mean(durations), statistics.median(durations), p95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, help="Путь к CSV, по умолчанию DATA_DIR/questions.csv")
    parser.add_argument("--kb", type=Path, help="Путь к ZIP базы знаний, по умолчанию поиск в DATA_DIR")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.monotonic()
    deadline = started + RUN_BUDGET
    directory = Path(env("DATA_DIR", default="data"))
    questions = read_questions(args.questions or directory / "questions.csv")
    answers = [FALLBACK] * len(questions)
    write_submission(args.output, questions, answers)
    try:
        before_prepare = time.monotonic()
        assistant = prepare(args.kb or find_kb(directory), min(deadline, before_prepare + 60))
        LOG.info("Подготовка базы %.2f с", time.monotonic() - before_prepare)
        if assistant is not None:
            if not assistant.settings.ready:
                LOG.warning("Прокси не настроен. Доступен поиск положений по базе.")
            run_questions(assistant, questions, answers, args.output, started, deadline)
    except FileNotFoundError:
        LOG.warning("Архив базы знаний отсутствует")
    finally:
        validate_submission(args.output, questions)
        LOG.info("Создан и проверен %s, %d строк за %.2f с", args.output.name, len(questions),
                 time.monotonic() - started)


if __name__ == "__main__":
    main()
