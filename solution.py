"""Hackathon runner: read independent questions and write a complete submission.csv."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from assistant import TECHNICAL, load_assistant
from proxy import env

LOG = logging.getLogger(__name__)
ARCHIVE_NAMES = ("knowledge-base.zip", "kb.zip", "knowledge_base.zip")


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


def write_submission(path: Path, questions: list[tuple[str, str]], answers: list[str]) -> None:
    tmp = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", newline="", delete=False)
    staged = Path(tmp.name)
    try:
        with tmp:
            writer = csv.writer(tmp, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(("id", "answer"))
            for (qid, _), answer in zip(questions, answers, strict=True):
                writer.writerow((qid, answer[:10_000]))
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, help="Путь к CSV, по умолчанию DATA_DIR/questions.csv")
    parser.add_argument("--kb", type=Path, help="Путь к ZIP базы знаний, по умолчанию поиск в DATA_DIR")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    directory = Path(env("DATA_DIR", default="data"))
    questions = read_questions(args.questions or directory / "questions.csv")
    # Leave time to replace the output atomically before the platform timeout.
    deadline = time.monotonic() + 1130
    answers = [TECHNICAL] * len(questions)
    write_submission(args.output, questions, answers)
    assistant = load_assistant(args.kb or find_kb(directory), deadline=deadline)
    if not assistant.settings.ready:
        LOG.warning("Для генерации нужны LLM_BASE_URL, LLM_API_KEY и LLM_MODEL")
        return

    pool = ThreadPoolExecutor(max_workers=assistant.settings.workers)
    pending = {}
    remaining = iter(enumerate(questions))

    def enqueue() -> None:
        for position, (_, question) in remaining:
            pending[pool.submit(assistant.ask, question, deadline=deadline)] = position
            break

    try:
        for _ in range(assistant.settings.workers):
            enqueue()
        finished = 0
        since_write = 0
        while pending and (time_left := deadline - time.monotonic()) > 0:
            done, _ = wait(pending, timeout=min(time_left, 8), return_when=FIRST_COMPLETED)
            for future in done:
                position = pending.pop(future)
                try:
                    answers[position] = future.result().csv_text() or TECHNICAL
                except Exception as exc:
                    LOG.warning("Ошибка обработки строки %d (%s)", position + 1, type(exc).__name__)
                finished += 1
                since_write += 1
                if time.monotonic() < deadline:
                    enqueue()
            if since_write >= 8 or (since_write and not done):
                write_submission(args.output, questions, answers)
                LOG.info("Обработано %d/%d", finished, len(questions))
                since_write = 0
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        write_submission(args.output, questions, answers)
    LOG.info("Создан %s, обработано %d/%d", args.output.name, finished, len(questions))


if __name__ == "__main__":
    main()
