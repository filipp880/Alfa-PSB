"""Compare search coverage on manually labelled passages from the supplied corpus.

The cases are evaluation fixtures, separate from the submitted runtime. The
method follows TREC's use of relevance judgments tied to the same corpus:
https://trec.nist.gov/data/reljudge_eng.html
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from unittest.mock import patch
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parent
HYPHENS = str.maketrans("ё‐‒–—−", "е-----")
ENDPOINTS = ("/chat/completions", "/embeddings", "/rerank")
NO_ANSWER_STATUSES = {"refusal", "insufficient", "clarification", "out_of_scope"}


def normalized(text: str) -> str:
    # Keep the number sign expected by the Russian fixture patterns.
    text = "№".join(unicodedata.normalize("NFKC", part) for part in text.split("№"))
    return " ".join(text.casefold().translate(HYPHENS).split())


def read_cases(path: Path, ids: list[str]) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8-sig"))
    known = {case["id"] for case in cases}
    if not cases or len(known) != len(cases):
        raise ValueError("Набор проверок пуст или содержит повторяющиеся ID.")
    unknown = set(ids) - known
    if unknown:
        raise ValueError(f"Неизвестные ID проверок: {', '.join(sorted(unknown))}")
    for case in cases:
        for pattern in case["must"]:
            re.compile(pattern)
        if any(not 0 <= index < len(case["evidence"]) for index in case.get("cite", [])):
            raise ValueError(f"Неверный индекс эталонной ссылки в {case['id']}.")
    return [case for case in cases if not ids or case["id"] in ids]


def gold_passages(corpus, evidence: list[dict]) -> list[set[str]]:
    """Match stable filename fragments and source spans across different indexes."""
    slots = []
    for item in evidence:
        file, quote = normalized(item["file"]), normalized(item["quote"])
        documents = [d for d in corpus.documents.values() if file in normalized(d.filename)]
        if len(documents) != 1:
            raise ValueError(f"Метка документа не уникальна или отсутствует: {item['file']!r}")
        document = documents[0]
        if not quote or quote not in normalized(document.text):
            raise ValueError(f"Эталонная цитата отсутствует в {document.filename}: {item['quote']!r}")
        slots.append({
            p.id for p in corpus.passages.values()
            if p.document_id == document.id and quote in normalized(p.text)
        })
    return slots


def hits(ids: set[str], slots: list[set[str]]) -> list[bool]:
    return [bool(ids & matches) for matches in slots]


def grade_answer(answer, case: dict, slots: list[set[str]], corpus) -> dict:
    status = answer.status if answer is not None else "error"
    text = normalized(answer.text) if answer is not None else ""
    missing_patterns = [pattern for pattern in case["must"] if not re.search(pattern, text)]
    citations = answer.citations if answer is not None else []
    valid_citations = [
        citation for citation in citations
        if (passage := corpus.passages.get(citation.id)) is not None
        and citation.document_id == passage.document_id
        and citation.quote and citation.quote in passage.text
    ]
    valid_ids = {citation.id for citation in valid_citations}
    required_cites = case.get("cite", range(len(slots)))
    missing_citations = [index for index in required_cites if not (valid_ids & slots[index])]
    return {
        "status": status,
        "status_ok": status in case["statuses"],
        "unwanted_no_answer": status in NO_ANSWER_STATUSES and status not in case["statuses"],
        "missing_patterns": missing_patterns,
        "missing_citations": missing_citations,
        "invalid_citations": len(citations) - len(valid_citations),
        "requires_citations": bool(required_cites),
    }


def passed(grade: dict) -> bool:
    return (grade["status_ok"] and not grade["missing_patterns"]
            and not grade["missing_citations"] and not grade["invalid_citations"])


def request_counts(assistant) -> Counter | None:
    read_counts = getattr(assistant.gateway, "request_counts", None)
    if not callable(read_counts):
        return None
    counts = read_counts()
    return Counter({endpoint: counts[endpoint] for endpoint in ENDPOINTS if endpoint in counts})


def request_delta(before: Counter | None, after: Counter | None) -> dict | None:
    return dict(after - before) if before is not None and after is not None else None


def evaluate_live_case(live, case: dict, slots: list[set[str]], cutoffs: list[int], timeout: float) -> dict:
    retrieve, select = live.retrieve, live.context
    answer_once = getattr(live, "_answer_once", None)
    rankings, contexts = [], []
    initial_answer = None
    retry_start = None

    def capture_retrieval(*args, **kwargs):
        found = retrieve(*args, **kwargs)
        rankings.append(found)
        return found

    def capture_context(*args, **kwargs):
        nonlocal retry_start
        if len(contexts) == 1:
            retry_start = request_counts(live)
        selected = select(*args, **kwargs)
        contexts.append(selected)
        return selected

    def capture_answer(*args, **kwargs):
        nonlocal initial_answer
        result = answer_once(*args, **kwargs)
        if initial_answer is None:
            initial_answer = result[0]
        return result

    before = request_counts(live)
    error_type = None
    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(live, "retrieve", capture_retrieval))
        stack.enter_context(patch.object(live, "context", capture_context))
        if callable(answer_once):
            stack.enter_context(patch.object(live, "_answer_once", capture_answer))
        try:
            answer = live.ask(case["question"], deadline=time.monotonic() + timeout)
        except Exception as exc:
            answer = None
            error_type = type(exc).__name__
    elapsed = time.perf_counter() - started
    after = request_counts(live)
    grade = grade_answer(answer, case, slots, live.corpus)
    initial = grade_answer(initial_answer, case, slots, live.corpus) if initial_answer is not None else None
    ranked = rankings[0] if rankings else []
    retried = len(contexts) > 1
    return {
        **grade,
        "elapsed_seconds": round(elapsed, 4),
        "exception_type": error_type,
        "ranked": {k: hits({p.id for p in ranked[:k]}, slots) for k in cutoffs},
        "context": hits({p.id for p in contexts[0]} if contexts else set(), slots),
        "additional_search": retried,
        "initial_grade": initial,
        "retry_recovered": not passed(initial) and passed(grade) if retried and initial is not None else None,
        "requests": request_delta(before, after),
        "retry_requests": request_delta(retry_start, after),
    }


def evaluate(args: argparse.Namespace, cases: list[dict]) -> dict:
    sys.path.insert(0, str(args.implementation))
    from assistant import dialogue, load_assistant

    local = load_assistant(args.kb, offline=True)
    live = None
    preparation = None
    if args.live and local.settings.ready:
        started = time.perf_counter()
        live = load_assistant(args.kb, deadline=time.monotonic() + args.timeout)
        preparation = {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "requests": request_delta(Counter(), request_counts(live)),
            "dense_active": bool(live.vectors),
        }

    results = []
    for case in cases:
        try:
            slots = gold_passages(local.corpus, case["evidence"])
        except ValueError as exc:
            raise ValueError(f"{case['id']}: {exc}") from exc
        question, history = dialogue(case["question"])
        started = time.perf_counter()
        kwargs = {}
        if hasattr(local, "scope_for"):
            kwargs["scope"] = local.scope_for(question, history)
        ranked = local.retrieve(question, history, time.monotonic() + args.timeout, **kwargs)
        context = local.context(ranked, **kwargs)
        result = {
            "id": case["id"],
            "group": case["group"],
            "ranked": {
                k: hits({p.id for p in ranked[:k]}, slots)
                for k in args.k
            },
            "context": hits({p.id for p in context}, slots),
            "index": [bool(slot) for slot in slots],
            "context_size": len(context),
            "context_chars": sum(len(local.corpus.context_text(p)) for p in context),
            "search_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if live is not None:
            result["live"] = evaluate_live_case(live, case, slots, args.k, args.timeout)
        results.append(result)
    return {
        "live_ready": live is not None,
        "live_preparation": preparation,
        "live_requests": request_delta(Counter(), request_counts(live)) if live is not None else None,
        "cases": results,
    }


@contextmanager
def implementation(path: Path):
    if path.is_dir():
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="psb-eval-") as temp, ZipFile(path) as archive:
        root = Path(temp).resolve()
        for entry in archive.infolist():
            relative = PurePosixPath(entry.filename)
            if entry.is_dir() or relative.suffix != ".py":
                continue
            target = root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Недопустимый путь модуля в ZIP: {entry.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
        yield root


def run_candidate(path: Path, args: argparse.Namespace) -> dict:
    with implementation(path) as directory:
        if not (directory / "assistant.py").is_file():
            raise ValueError(f"Не найден assistant.py в корне {path}")
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()), "--implementation", str(directory),
            "--kb", str(args.kb), "--cases", str(args.cases), "--timeout", str(args.timeout),
            "--k", *map(str, args.k),
        ]
        for case in args.case:
            command.extend(("--case", case))
        if args.live:
            command.append("--live")
        completed = subprocess.run(
            command,
            cwd=directory,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            stdout=subprocess.PIPE,
            encoding="utf-8",
            check=True,
        )
        return json.loads(completed.stdout)


def mean_recall(rows: list[list[bool]]) -> float | None:
    return statistics.mean(sum(row) / len(row) for row in rows) if rows else None


def metrics(run: dict, cutoffs: list[int]) -> dict:
    rows = run["cases"]
    with_evidence = [row for row in rows if row["context"]]
    online = [row["live"] for row in rows if "live" in row]
    online_evidence = [row for row in online if row["context"]]
    online_required_citations = [row for row in online if row["requires_citations"]]
    online_times = sorted(row["elapsed_seconds"] for row in online)
    retries = [row for row in online if row["additional_search"]]
    graded_retries = [row for row in retries if row["retry_recovered"] is not None]
    recovered = sum(row["retry_recovered"] for row in graded_retries)
    retry_requests: Counter = Counter()
    for row in retries:
        retry_requests.update(row["retry_requests"] or {})
    unwanted = Counter(row["status"] for row in online if row["unwanted_no_answer"])
    preparation = run["live_preparation"]
    return {
        "cases": len(rows),
        "cases_with_evidence": len(with_evidence),
        **{
            f"recall@{k}": mean_recall([row["ranked"][str(k)] for row in with_evidence])
            for k in cutoffs
        },
        "context_recall": mean_recall([row["context"] for row in with_evidence]),
        "context_complete": sum(all(row["context"]) for row in with_evidence),
        "index_complete": sum(all(row["index"]) for row in with_evidence),
        "median_context_size": statistics.median(row["context_size"] for row in rows),
        "median_context_chars": statistics.median(row["context_chars"] for row in rows),
        "median_search_ms": statistics.median(row["search_ms"] for row in rows),
        "live_cases": len(online),
        "live_status_ok": sum(row["status_ok"] for row in online),
        "live_facts_ok": sum(not row["missing_patterns"] for row in online),
        "live_citations_ok": sum(not row["missing_citations"] and not row["invalid_citations"] for row in online),
        "live_cases_requiring_citations": len(online_required_citations),
        "live_supported": sum(not row["missing_citations"] and not row["invalid_citations"]
                              for row in online_required_citations),
        "live_pass": sum(passed(row) for row in online),
        "live_technical_errors": sum(row["status"] == "error" for row in online),
        "live_unwanted_no_answer": sum(unwanted.values()),
        "live_unwanted_no_answer_statuses": dict(unwanted),
        **{
            f"live_recall@{k}": mean_recall([row["ranked"][str(k)] for row in online_evidence])
            for k in cutoffs
        },
        "live_context_recall": mean_recall([row["context"] for row in online_evidence]),
        "live_context_complete": sum(all(row["context"]) for row in online_evidence),
        "live_additional_search": len(retries),
        "live_retry_rate": len(retries) / len(online) if online else None,
        "live_retry_recovered": recovered,
        "live_retry_graded": len(graded_retries),
        "live_retry_recovery_rate": recovered / len(graded_retries) if graded_retries else None,
        "live_retry_requests": dict(retry_requests) if run["live_requests"] is not None else None,
        "live_p50_seconds": statistics.median(online_times) if online_times else None,
        "live_p95_seconds": online_times[math.ceil(0.95 * len(online_times)) - 1] if online_times else None,
        "live_elapsed_seconds": round(preparation["elapsed_seconds"] + sum(online_times), 4)
        if preparation is not None else None,
    }


def percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def format_requests(counts: dict | None) -> str:
    if counts is None:
        return "недоступно для этой версии"
    return ", ".join(f"{endpoint} {counts.get(endpoint, 0)}" for endpoint in ENDPOINTS)


def print_result(label: str, run: dict, cases: list[dict], args: argparse.Namespace) -> None:
    m = run["metrics"]
    print(f"\n{label}. Лексический поиск по {m['cases']} сценариям")
    print("  " + "   ".join(f"Recall@{k} {percent(m[f'recall@{k}'])}" for k in args.k))
    total = m["cases_with_evidence"]
    print(f"  Полнота контекста {percent(m['context_recall'])}, все фрагменты {m['context_complete']}/{total}")
    print(f"  Все фрагменты присутствуют в индексе в {m['index_complete']}/{total} сценариях")
    print(f"  Медиана {m['median_context_size']:g} фрагм. и {m['median_context_chars']:g} знаков в контексте, "
          f"{m['median_search_ms']:g} мс на поиск и сбор")
    if m["live_cases"]:
        live_total = m["live_cases"]
        print(f"  API: статус {m['live_status_ok']}/{live_total}, шаблоны {m['live_facts_ok']}/{live_total}, "
              f"источники {m['live_citations_ok']}/{live_total}, всё вместе {m['live_pass']}/{live_total}")
        print(f"  Ожидаемые ссылки с точными цитатами {m['live_supported']}/{m['live_cases_requiring_citations']}, "
              f"технических ошибок {m['live_technical_errors']}, "
              f"неожиданных отказов или уточнений {m['live_unwanted_no_answer']}")
        print("  API   " + "   ".join(f"Recall@{k} {percent(m[f'live_recall@{k}'])}" for k in args.k))
        print(f"  Полнота первого контекста API {percent(m['live_context_recall'])}, "
              f"все фрагменты {m['live_context_complete']}/{total}")
        print(f"  Повторная генерация после поиска {m['live_additional_search']}/{live_total}, "
              f"восстановлено {m['live_retry_recovered']}/{m['live_retry_graded']}, "
              f"запросы повтора {format_requests(m['live_retry_requests'])}")
        preparation = run["live_preparation"]
        print(f"  p50 {m['live_p50_seconds']:g} с, p95 {m['live_p95_seconds']:g} с, "
              f"подготовка и обработка {m['live_elapsed_seconds']:g} с")
        print(f"  Запросы всего {format_requests(run['live_requests'])}")
        index_status = "векторный индекс используется" if preparation["dense_active"] else "поиск без векторного индекса"
        print(f"  Подготовка {preparation['elapsed_seconds']:g} с, {index_status}, "
              f"запросы {format_requests(preparation['requests'])}")
        print("  Проверка шаблонов и источников приближённая. Она не заменяет экспертную оценку ответа.")
    elif args.live:
        print("  Прогон API пропущен. Требуются LLM_BASE_URL, LLM_API_KEY и LLM_MODEL.")
    if args.details:
        for case, result in zip(cases, run["cases"], strict=True):
            missing = [item["file"] + " / " + item["quote"]
                       for item, hit in zip(case["evidence"], result["context"], strict=True) if not hit]
            status = "без эталонных фрагментов" if not case["evidence"] else f"пропущено фрагментов {len(missing)}"
            print(f"    {result['id']}: {status}")
            for text in missing:
                print(f"      отсутствует {text}")
            if "live" in result:
                live = result["live"]
                print(f"      статус {live['status']}, шаблонов пропущено {len(live['missing_patterns'])}, "
                      f"источников пропущено {len(live['missing_citations'])}, "
                      f"некорректных цитат {live['invalid_citations']}")
                print(f"      время {live['elapsed_seconds']:g} с, запросы {format_requests(live['requests'])}")
                if live["initial_grade"] is not None and live["additional_search"]:
                    print(f"      первый статус {live['initial_grade']['status']}, "
                          f"проверки до и после повтора {int(passed(live['initial_grade']))} → {int(passed(live))}")
                if live["exception_type"]:
                    print(f"      исключение {live['exception_type']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка поиска и ответов на отдельных вопросах по базе кейса.",
        epilog="Метрики сравнивают локальные версии на одной разметке. "
               "Это не результат оценки организаторов.",
    )
    parser.add_argument("--candidate", type=Path, default=ROOT, help="Каталог проекта или ZIP")
    parser.add_argument("--baseline", type=Path, help="Вторая реализация для сравнения, каталог или ZIP")
    parser.add_argument("--kb", type=Path, default=ROOT / "data" / "knowledge-base.zip")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "cases.json")
    parser.add_argument("--case", action="append", default=[], help="ID для прогона, можно повторять")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10], help="Глубины Recall@k")
    parser.add_argument("--timeout", type=float, default=60, help="Лимит секунд на сценарий при прогоне через API")
    parser.add_argument("--live", action="store_true", help="Дополнительно вызвать настроенный API модели")
    parser.add_argument("--details", action="store_true", help="Вывести отсутствующие фрагменты по сценариям")
    parser.add_argument("--json", action="store_true", help="Вывести метрики и оценки в JSON")
    parser.add_argument("--implementation", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(k < 1 for k in args.k) or not 0 < args.timeout <= 120:
        parser.error("Глубины должны быть положительными, таймаут должен быть больше 0 и не более 120 с.")
    args.k = list(dict.fromkeys(args.k))
    args.kb, args.cases = args.kb.resolve(), args.cases.resolve()
    try:
        cases = read_cases(args.cases, args.case)
        if args.implementation:
            print(json.dumps(evaluate(args, cases), ensure_ascii=False))
            return
        paths = [("Текущая версия", args.candidate)]
        if args.baseline:
            paths.insert(0, ("Базовая версия", args.baseline))
        results = {}
        for label, path in paths:
            run = run_candidate(path.resolve(), args)
            run["metrics"] = metrics(run, args.k)
            results[label] = run
            if not args.json:
                print_result(label, run, cases, args)
        if args.json:
            print(json.dumps(results, indent=2))
    except (BadZipFile, OSError, ValueError, re.error, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
