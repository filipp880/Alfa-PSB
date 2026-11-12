"""Compare search coverage on manually labelled passages from the supplied corpus.

The cases are evaluation fixtures, separate from the submitted runtime. The
method follows TREC's use of relevance judgments tied to the same corpus:
https://trec.nist.gov/data/reljudge_eng.html
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy
from dataclasses import asdict, replace
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
SUBSTANTIVE_STATUSES = {"answered", "partial"}
EVALUATION_STATUSES = SUBSTANTIVE_STATUSES | NO_ANSWER_STATUSES | {"error"}
PASSAGE_MARKER = re.compile(r"\[(P\d+)\]")
CITATION_RUN = re.compile(r"\[P\d+\](?:[\s,]*\[P\d+\])*")
CLAUSE_BREAK = re.compile(r"[;\n]|[!?](?=\s)|(?<!\b\w)(?<!\bул)(?<!\bстр)(?<!\bруб)\.(?=\s)")
MUTATION_FIELDS = ("member", "old", "new")


def normalized(text: str) -> str:
    # Keep the number sign expected by the Russian fixture patterns.
    text = "№".join(unicodedata.normalize("NFKC", part) for part in text.split("№"))
    return " ".join(text.casefold().translate(HYPHENS).split())


def read_cases(path: Path, ids: list[str]) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Набор проверок должен быть непустым списком.")
    for number, case in enumerate(cases):
        label = case.get("id", number) if isinstance(case, dict) else number

        def require(condition, field):
            if not condition:
                raise ValueError(f"Некорректное поле {field} в проверке {label!r}.")

        def strings(value):
            return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)

        require(isinstance(case, dict), "case")
        for field in ("id", "group", "question"):
            require(isinstance(case.get(field), str) and case[field].strip(), field)
        require(isinstance(case.get("evidence"), list), "evidence")
        for item in case["evidence"]:
            require(isinstance(item, dict) and all(isinstance(item.get(key), str) and item[key].strip()
                                                 for key in ("file", "quote")), "evidence")
        require(strings(case.get("statuses")) and bool(case["statuses"])
                and set(case["statuses"]) <= EVALUATION_STATUSES, "statuses")
        require(strings(case.get("must")), "must")
        require(strings(case.get("must_not", [])), "must_not")
        for field in ("expected", "pair"):
            if field in case:
                require(isinstance(case[field], str), field)
        assertions = case.get("assertions", [])
        require(isinstance(assertions, list), "assertions")
        require(isinstance(case.get("cite", []), list), "cite")
        patterns = case["must"] + case.get("must_not", [])
        for item in assertions:
            require(isinstance(item, dict) and strings(item.get("must"))
                    and isinstance(item.get("cite"), list), "assertions")
            patterns.extend(item["must"])
            if "subject" in item:
                require(isinstance(item["subject"], str) and item["subject"].strip(), "assertions.subject")
                patterns.append(item["subject"])
            if "same_clause" in item:
                require(isinstance(item["same_clause"], bool), "assertions.same_clause")
        for pattern in patterns:
            re.compile(pattern)
        indices = [*case.get("cite", []), *(index for item in assertions for index in item["cite"])]
        if any(type(index) is not int or not 0 <= index < len(case["evidence"]) for index in indices):
            raise ValueError(f"Неверный индекс эталонной ссылки в {case['id']}.")
        if "kb_mutation" in case:
            mutation = case["kb_mutation"]
            if (not isinstance(mutation, dict) or set(mutation) != set(MUTATION_FIELDS)
                    or not all(isinstance(mutation[field], str) for field in MUTATION_FIELDS)
                    or not mutation["old"] or mutation["old"] == mutation["new"]
                    or PurePosixPath(mutation["member"]).suffix.lower() not in {".txt", ".md"}):
                raise ValueError(f"Некорректное изменение базы в {case['id']}.")
    known = {case["id"] for case in cases}
    if len(known) != len(cases):
        raise ValueError("Набор проверок содержит повторяющиеся ID.")
    unknown = set(ids) - known
    if unknown:
        raise ValueError(f"Неизвестные ID проверок: {', '.join(sorted(unknown))}")
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


def cited_blocks(text: str, valid_ids: set[str]):
    """Attach each run of rendered citations to the claim it follows."""
    for paragraph in re.split(r"\n\s*\n", text):
        start = 0
        for markers in CITATION_RUN.finditer(paragraph):
            if block := paragraph[start:markers.start()].strip():
                yield block, valid_ids.intersection(PASSAGE_MARKER.findall(markers[0]))
            start = markers.end()
        if block := paragraph[start:].strip():
            yield block, set()


def subject_clauses(block: str, subjects: list[str]):
    """Keep values on their own side when a claim names several distinct subjects."""
    for raw_clause in CLAUSE_BREAK.split(block):
        clause = normalized(raw_clause)
        mentions = sorted((match.start(), match.end(), subject) for subject in subjects
                          for match in re.finditer(subject, clause))
        start = 0
        for previous, current in zip(mentions, mentions[1:]):
            if previous[2] != current[2] and previous[1] <= current[0]:
                yield clause[start:current[0]], previous[2]
                start = current[0]
        if mentions:
            yield clause[start:], mentions[-1][2]


def grade_answer(answer, case: dict, slots: list[set[str]], corpus) -> dict:
    status = answer.status if answer is not None else "error"
    answer_text = answer.text if answer is not None else ""
    text = normalized(answer_text)
    missing_patterns = [pattern for pattern in case["must"] if not re.search(pattern, text)]
    citations = answer.citations if answer is not None else []
    valid_citations = [
        citation for citation in citations
        if (passage := corpus.passages.get(citation.id)) is not None
        and citation.document_id == passage.document_id
        and citation.locator == passage.locator
        and citation.title == corpus.documents[passage.document_id].title
        and citation.quote and citation.quote in passage.text
    ]
    # CSV only expands markers present in the answer text. A source object kept
    # in metadata cannot support an answer whose rendered source was lost.
    rendered_ids = set(PASSAGE_MARKER.findall(answer_text))
    valid_ids = {citation.id for citation in valid_citations} & rendered_ids
    dangling_ids = rendered_ids - {citation.id for citation in citations}
    required_cites = case.get("cite", range(len(slots)))
    missing_citations = [index for index in required_cites if not (valid_ids & slots[index])]
    missing_assertions = []
    if assertions := case.get("assertions"):
        blocks = list(cited_blocks(answer_text, valid_ids))
        subjects = list(dict.fromkeys(item["subject"] for item in assertions if "subject" in item))
        for index, item in enumerate(assertions):
            candidates = [block for block, ids in blocks if all(ids & slots[slot] for slot in item["cite"])]
            if subject := item.get("subject"):
                candidates = [clause for block in candidates for clause, owner in subject_clauses(block, subjects)
                              if owner == subject]
            elif item.get("same_clause"):
                candidates = [normalized(clause) for block in candidates
                              for clause in re.split(r";|,(?!\d)", block)]
            else:
                candidates = list(map(normalized, candidates))
            if not any(all(re.search(pattern, block) for pattern in item["must"]) for block in candidates):
                missing_assertions.append(index)
    return {
        "status": status,
        "status_ok": status in case["statuses"],
        "unwanted_no_answer": status in NO_ANSWER_STATUSES and status not in case["statuses"],
        "expected_substantive": bool(SUBSTANTIVE_STATUSES.intersection(case["statuses"])),
        "missing_patterns": missing_patterns,
        "forbidden_patterns": [pattern for pattern in case.get("must_not", []) if re.search(pattern, text)],
        "missing_assertions": missing_assertions,
        "missing_citations": missing_citations,
        "invalid_citations": len(citations) - len(valid_citations) + len(dangling_ids),
        "requires_citations": bool(required_cites),
    }


def passed(grade: dict) -> bool:
    return (grade["status_ok"] and not grade["missing_patterns"] and not grade["forbidden_patterns"]
            and not grade["missing_citations"] and not grade["invalid_citations"] and not grade["missing_assertions"])


def request_counts(assistant) -> Counter | None:
    read_counts = getattr(assistant.gateway, "request_counts", None)
    if not callable(read_counts):
        return None
    counts = read_counts()
    return Counter({endpoint: counts[endpoint] for endpoint in ENDPOINTS if endpoint in counts})


def request_delta(before: Counter | None, after: Counter | None) -> dict | None:
    return dict(after - before) if before is not None and after is not None else None


def evaluate_live_case(live, case: dict, slots: list[set[str]], cutoffs: list[int], timeout: float,
                       *, audit: bool = False, reference_context: bool = False) -> dict:
    retrieve, select = live.retrieve, live.context
    answer_once = getattr(live, "_answer_once", None)
    rankings, contexts = [], []
    initial_answer = None
    attempts = []
    retry_start = None
    stages = []
    reference_passages = list(dict.fromkeys(
        min(slot, key=lambda pid: (len(live.corpus.passages[pid].text), pid)) for slot in slots if slot
    )) if reference_context else []

    def capture_retrieval(*args, **kwargs):
        found = retrieve(*args, **kwargs)
        rankings.append(found)
        return found

    def capture_context(*args, **kwargs):
        nonlocal retry_start
        if len(contexts) == 1:
            retry_start = request_counts(live)
        selected = ([live.corpus.passages[pid] for pid in reference_passages]
                    if reference_context else select(*args, **kwargs))
        contexts.append(selected)
        return selected

    def capture_answer(*args, **kwargs):
        nonlocal initial_answer
        result = answer_once(*args, **kwargs)
        attempts.append(result[0])
        if initial_answer is None:
            initial_answer = result[0]
        return result

    chat = live.gateway.chat

    def capture_chat(system, user, deadline, *, verify=False):
        stage = {"stage": "verify" if verify else "draft"}
        stage["anchors"] = deepcopy(user.get("anchors", {}))
        stage["passages"] = [
            {"id": p.get("id"), "applicability": p.get("applicability"), "text_visible": bool(p.get("text"))}
            for p in user.get("passages", []) if isinstance(p, dict)
        ]
        # Explicit payload allowlist: never serialize the gateway/settings object.
        stage["input"] = {key: deepcopy(user[key]) for key in ("gap", "previous_gap") if key in user}
        for key in ("claims", "previous_claims"):
            if isinstance(user.get(key), list):
                stage["input"][key] = [
                    {field: deepcopy(claim[field]) for field in ("index", "text", "evidence") if field in claim}
                    for claim in user[key] if isinstance(claim, dict)
                ]
        stages.append(stage)
        try:
            result = chat(system, user, deadline, verify=verify)
        except Exception as exc:
            stage["exception_type"] = type(exc).__name__
            raise
        keys = ("supported", "gap_ok", "previous_supported", "previous_covered", "adds_information", "consistent") \
            if verify else ("status", "claims", "gap", "look_for")
        output = {key: deepcopy(result[key]) for key in keys if key in result} if isinstance(result, dict) else {}
        stage["output_type"] = type(result).__name__
        if isinstance(output.get("claims"), list):
            output["claims"] = [
                {key: claim[key] for key in ("text", "evidence") if key in claim}
                for claim in output["claims"] if isinstance(claim, dict)
            ]
        stage["output"] = output
        return result

    before = request_counts(live)
    error_type = None
    started = time.perf_counter()
    with ExitStack() as stack:
        if reference_context:
            stack.enter_context(patch.object(live, "settings", replace(live.settings, retry_search=False)))
        stack.enter_context(patch.object(live, "retrieve", capture_retrieval))
        stack.enter_context(patch.object(live, "context", capture_context))
        if callable(answer_once):
            stack.enter_context(patch.object(live, "_answer_once", capture_answer))
        if audit:
            stack.enter_context(patch.object(live.gateway, "chat", capture_chat))
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

    def rendered_signature(value):
        return (value.status, value.text, value.citations) if value is not None else None

    decision = "not_attempted"
    if retried:
        matches_initial = rendered_signature(answer) == rendered_signature(initial_answer)
        matches_candidate = len(attempts) > 1 and rendered_signature(answer) == rendered_signature(attempts[-1])
        decision = ("unchanged" if matches_initial and matches_candidate else "retained_initial" if matches_initial
                    else "accepted_candidate" if matches_candidate else "reconciled" if answer is not None else "error")

    def attempt_summary(value):
        review = getattr(value, "retry_review", None)
        return {"status": value.status, "answer": value.text, "csv_answer": value.csv_text(),
                "citation_ids": [citation.id for citation in value.citations],
                "gap": getattr(value, "gap", None), "retry_review": asdict(review) if review is not None else None}

    return {
        **grade,
        "reference_context": reference_context,
        "elapsed_seconds": round(elapsed, 4),
        "exception_type": error_type,
        "ranked": {k: hits({p.id for p in ranked[:k]}, slots) for k in cutoffs},
        "context": hits({p.id for p in contexts[0]} if contexts else set(), slots),
        "additional_search": retried,
        "initial_grade": initial,
        "retry_recovered": not passed(initial) and passed(grade) if retried and initial is not None else None,
        "requests": request_delta(before, after),
        "retry_requests": request_delta(retry_start, after),
        **({"audit": {
            "answer": answer.text if answer is not None else "",
            "csv_answer": answer.csv_text() if answer is not None else "",
            "citations": [asdict(citation) for citation in answer.citations] if answer is not None else [],
            "ranked_ids": [[p.id for p in ranking] for ranking in rankings],
            "context_ids": [[p.id for p in selection] for selection in contexts],
            "stages": stages,
            "attempts": [attempt_summary(value) for value in attempts],
            "retry_decision": {"attempted": retried, "selected": decision,
                               "initial_status": initial_answer.status if initial_answer is not None else None,
                               "candidate_status": attempts[-1].status if len(attempts) > 1 else None,
                               "final_status": answer.status if answer is not None else "error"},
        }} if audit else {}),
    }


@contextmanager
def mutated_kb(path: Path, mutation: dict):
    """Copy the official archive and replace a single exact UTF-8 source span."""
    member = mutation["member"]
    old, new = mutation["old"].encode("utf-8"), mutation["new"].encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="psb-eval-kb-") as temp, ZipFile(path) as original:
        if original.namelist().count(member) != 1:
            raise ValueError(f"Не найден единственный файл изменения базы: {member!r}")
        source = original.read(member)
        if source.count(old) != 1:
            raise ValueError(f"Заменяемый текст должен встречаться ровно один раз в {member!r}")
        target = Path(temp) / path.name
        with ZipFile(target, "w") as changed:
            for entry in original.infolist():
                content = source.replace(old, new, 1) if entry.filename == member else original.read(entry)
                changed.writestr(copy(entry), content)
        yield target


def evaluate_case(local, live, case: dict, question: str, history: list[dict], args: argparse.Namespace) -> dict:
    try:
        slots = gold_passages(local.corpus, case["evidence"])
    except ValueError as exc:
        raise ValueError(f"{case['id']}: {exc}") from exc
    started = time.perf_counter()
    kwargs = {}
    if hasattr(local, "scope_for"):
        kwargs["scope"] = local.scope_for(question, history)
    ranked = local.retrieve(question, history, time.monotonic() + args.timeout, **kwargs)
    context = local.context(ranked, **kwargs)
    result = {
        "id": case["id"],
        "group": case["group"],
        "ranked": {k: hits({p.id for p in ranked[:k]}, slots) for k in args.k},
        "context": hits({p.id for p in context}, slots),
        "index": [bool(slot) for slot in slots],
        "context_size": len(context),
        "context_chars": sum(len(local.corpus.context_text(p)) for p in context),
        "search_ms": round((time.perf_counter() - started) * 1000, 2),
        "kb_mutated": "kb_mutation" in case,
    }
    if live is not None:
        result["live"] = evaluate_live_case(live, case, slots, args.k, args.timeout, audit=args.audit,
                                            reference_context=args.reference_context)
    return result


def total_requests(counts: list[Counter | None]) -> dict | None:
    if not counts or any(count is None for count in counts):
        return None
    total = Counter()
    for count in counts:
        total.update(count)
    return dict(total)


def evaluate(args: argparse.Namespace, cases: list[dict]) -> dict:
    sys.path.insert(0, str(args.implementation))
    from assistant import dialogue, load_assistant

    assistants = {}
    prepared_counts = []
    preparation_time = 0.0
    results = []
    with ExitStack() as stack:
        for case in cases:
            mutation = case.get("kb_mutation")
            key = tuple(mutation[field] for field in MUTATION_FIELDS) if mutation else ()
            if key not in assistants:
                path = stack.enter_context(mutated_kb(args.kb, mutation)) if mutation else args.kb
                local = load_assistant(path, offline=True)
                live = None
                if args.live and local.settings.ready:
                    started = time.perf_counter()
                    live = load_assistant(path, deadline=time.monotonic() + args.timeout)
                    preparation_time += time.perf_counter() - started
                    prepared_counts.append(request_counts(live))
                assistants[key] = local, live
            local, live = assistants[key]
            question, history = dialogue(case["question"])
            results.append(evaluate_case(local, live, case, question, history, args))
    live_assistants = [live for _, live in assistants.values() if live is not None]
    return {
        "live_ready": bool(live_assistants),
        "live_preparation": {
            "elapsed_seconds": round(preparation_time, 4),
            "requests": total_requests(prepared_counts),
            "dense_active": all(live.vectors for live in live_assistants),
            "dense_corpora": sum(bool(live.vectors) for live in live_assistants),
            "corpora": len(live_assistants),
        } if live_assistants else None,
        "live_requests": total_requests([request_counts(live) for live in live_assistants]),
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
        if args.audit:
            command.extend(("--audit", "--json"))
        if args.reference_context:
            command.append("--reference-context")
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


def selective_metrics(grades: list[dict]) -> dict:
    """Report fixture failures, not an unmeasured semantic error probability."""
    substantive = [grade for grade in grades if grade["status"] in SUBSTANTIVE_STATUSES]
    failed = sum(not passed(grade) for grade in substantive)
    answerable = [grade for grade in grades if grade.get("expected_substantive", False)]
    refusals = sum(grade["unwanted_no_answer"] for grade in answerable)
    return {
        "live_substantive_answers": len(substantive),
        "live_answer_coverage": len(substantive) / len(grades) if grades else None,
        "live_substantive_fixture_failures": failed,
        "live_substantive_fixture_failure_rate": failed / len(substantive) if substantive else None,
        "live_answerable_cases": len(answerable),
        "live_answerable_unwanted_no_answer": refusals,
        "live_answerable_unwanted_no_answer_rate": refusals / len(answerable) if answerable else None,
    }


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
        **selective_metrics(online),
        "live_status_ok": sum(row["status_ok"] for row in online),
        "live_facts_ok": sum(not row["missing_patterns"] and not row["forbidden_patterns"]
                             and not row["missing_assertions"] for row in online),
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
        if args.reference_context:
            print("  Генерация получила только размеченные эталонные фрагменты. Повторный поиск отключён.")
        live_total = m["live_cases"]
        print(f"  API: статус {m['live_status_ok']}/{live_total}, шаблоны {m['live_facts_ok']}/{live_total}, "
              f"источники {m['live_citations_ok']}/{live_total}, всё вместе {m['live_pass']}/{live_total}")
        print(f"  Ожидаемые ссылки с точными цитатами {m['live_supported']}/{m['live_cases_requiring_citations']}, "
              f"технических ошибок {m['live_technical_errors']}, "
              f"неожиданных отказов или уточнений {m['live_unwanted_no_answer']}")
        print(f"  Доля содержательных ответов {percent(m['live_answer_coverage'])}; "
              f"не прошли размеченные проверки {m['live_substantive_fixture_failures']}/"
              f"{m['live_substantive_answers']} ({percent(m['live_substantive_fixture_failure_rate'])}); "
              f"неожиданные отказы/уточнения на ответимых вопросах "
              f"{m['live_answerable_unwanted_no_answer']}/{m['live_answerable_cases']}")
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
        if preparation["corpora"] > 1:
            index_status = f"векторный индекс в {preparation['dense_corpora']}/{preparation['corpora']} версиях базы"
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
            variant = " (временная копия базы)" if result["kb_mutated"] else ""
            print(f"    {result['id']}{variant}: {status}")
            for text in missing:
                print(f"      отсутствует {text}")
            if "live" in result:
                live = result["live"]
                print(f"      статус {live['status']}, шаблонов пропущено {len(live['missing_patterns'])}, "
                      f"провалено проверок вывода {len(live['missing_assertions'])}, "
                      f"запрещённых совпадений {len(live['forbidden_patterns'])}, "
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
    parser.add_argument("--audit", action="store_true",
                        help="Включить в JSON тексты ответов, цитаты и структурированные результаты генерации и проверки")
    parser.add_argument("--reference-context", action="store_true",
                        help="Диагностический прогон генерации только с эталонными фрагментами разметки")
    parser.add_argument("--implementation", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.audit and not (args.live and args.json):
        parser.error("--audit требует --live и --json")
    if args.reference_context and not args.live:
        parser.error("--reference-context требует --live")
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
