"""Compare search coverage on manually labelled passages from the supplied corpus.

The cases are evaluation fixtures, separate from the submitted runtime. The
method follows TREC's use of relevance judgments tied to the same corpus:
https://trec.nist.gov/data/reljudge_eng.html
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
HYPHENS = str.maketrans("ё−–—‑", "е----")


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().translate(HYPHENS).split())


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


def grade_answer(answer, case: dict, slots: list[set[str]]) -> dict:
    text = normalized(answer.text)
    missing_patterns = [pattern for pattern in case["must"] if not re.search(pattern, text)]
    citations = {citation.id for citation in answer.citations}
    required_cites = case.get("cite", range(len(slots)))
    missing_citations = [index for index in required_cites if not (citations & slots[index])]
    return {
        "status": answer.status,
        "status_ok": answer.status in case["statuses"],
        "missing_patterns": missing_patterns,
        "missing_citations": missing_citations,
    }


def evaluate(args: argparse.Namespace, cases: list[dict]) -> dict:
    sys.path.insert(0, str(args.implementation))
    from assistant import dialogue, load_assistant

    local = load_assistant(args.kb, offline=True)
    live = None
    if args.live and local.settings.ready:
        live = load_assistant(args.kb, deadline=time.monotonic() + args.timeout)

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
            contexts = []
            select = live.context

            def capture_context(*args, **kwargs):
                selected = select(*args, **kwargs)
                contexts.append(selected)
                return selected

            started = time.perf_counter()
            with patch.object(live, "context", capture_context):
                answer = live.ask(case["question"], deadline=time.monotonic() + args.timeout)
            result["live"] = grade_answer(answer, case, slots)
            result["live"]["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            result["live"]["context"] = hits({p.id for p in contexts[0]} if contexts else set(), slots)
            result["live"]["additional_search"] = len(contexts) > 1
        results.append(result)
    return {"live_ready": live is not None, "cases": results}


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
    online_times = sorted(row["elapsed_seconds"] for row in online)
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
        "live_citations_ok": sum(not row["missing_citations"] for row in online),
        "live_pass": sum(
            row["status_ok"] and not row["missing_patterns"] and not row["missing_citations"]
            for row in online
        ),
        "live_context_recall": mean_recall([row["context"] for row in online if row["context"]]),
        "live_additional_search": sum(row["additional_search"] for row in online),
        "live_p95_seconds": online_times[math.ceil(0.95 * len(online_times)) - 1] if online_times else None,
    }


def percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


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
        print(f"  Полнота первого контекста API {percent(m['live_context_recall'])}, "
              f"повторный поиск {m['live_additional_search']}/{live_total}, p95 {m['live_p95_seconds']:g} с")
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
                      f"источников пропущено {len(live['missing_citations'])}")


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
