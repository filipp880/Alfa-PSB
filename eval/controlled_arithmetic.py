"""Check operand/source/unit handling with scripted drafts and verdicts, not LLM reasoning."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "knowledge-base.zip"
MEMBER = "документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt"
CLAUSE = ("7.2. За просрочку оплаты Товара Покупатель уплачивает Поставщику неустойку в размере "
          "0,1% от неоплаченной суммы за каждый день просрочки, но не более 10% от неоплаченной суммы.")
WRONG_CLAUSE = "7.1. За просрочку поставки Товара Поставщик уплачивает Покупателю"
OPERANDS = re.compile(r"(\d+(?:,\d+)?)% от неоплаченной суммы за каждый день просрочки, но не более (\d+(?:,\d+)?)%")
GAP = "Не указана фактическая длительность просрочки во всех днях, известны только рабочие дни."


@dataclass(frozen=True)
class Case:
    id: str
    base: int
    days: int
    rate: str = "0.1"
    cap: str = "10"
    behavior: str = "correct"
    unit: str = "календарных дней"

    def numbers(self):
        principal = Decimal(self.base)
        raw = principal * Decimal(self.rate) * self.days / 100
        maximum = principal * Decimal(self.cap) / 100
        return raw, maximum, min(raw, maximum)

    def question(self):
        return (
            f"Договор поставки П-17/2024 от 12 февраля 2024 г. "
            f"На 15 ноября 2026 года покупатель просрочил оплату на {self.days} {self.unit}; "
            f"неоплаченная сумма за партию {self.base} руб. "
            "По какому пункту договора и в каком размере покупатель платит поставщику неустойку? "
            "Укажите расчёт с учётом предела."
        )


def money(amount):
    return format(amount.quantize(Decimal("0.01")), ",.2f").replace(",", " ").replace(".", ",")


def percentage(number):
    return format(Decimal(number), "f").replace(".", ",") + "%"


def fixture_cases():
    return [
        Case("base", 372500, 43),
        Case("principal", 784300, 43),
        Case("duration-at-cap", 372500, 100),
        Case("duration-over-cap", 372500, 167),
        Case("rate", 372500, 43, rate="0.17"),
        Case("cap", 372500, 167, cap="6"),
        Case("wrong-operands-source", 372500, 43, behavior="wrong_source"),
        Case("wrong-raw-without-cap", 372500, 167, behavior="raw_instead"),
        Case("wrong-working-days-unit", 372500, 12, behavior="days_unit", unit="рабочих дней"),
    ]


def make_variant(path, rate, cap):
    original = CLAUSE.encode("utf-8")
    replacement = CLAUSE.replace("0,1%", percentage(rate)).replace("10%", percentage(cap)).encode("utf-8")
    with ZipFile(BASE) as archive, ZipFile(path, "w") as out:
        for info in archive.infolist():
            raw = archive.read(info)
            if info.filename == MEMBER:
                assert raw.count(original) == 1
                raw = raw.replace(original, replacement, 1)
            out.writestr(copy(info), raw)


def draft_for(case, payload, trace):
    passages = payload["passages"]
    payment = next(p for p in passages if "7.2. За просрочку оплаты Товара Покупатель" in p["text"])
    trace["event_role"] = payload["anchors"]["event_role"]
    trace["event_ranges"] = payload["anchors"]["event_ranges"]
    trace["selected"] = payment["id"]
    trace["selected_applicability"] = payment["applicability"]
    trace["rate_and_cap_source"] = payment["text"]
    match = OPERANDS.search(payment["text"])
    assert match
    rate, cap = (Decimal(x.replace(",", ".")) for x in match.groups())
    assert (rate, cap) == (Decimal(case.rate), Decimal(case.cap))
    question_match = re.search(r"просрочил оплату на (\d+) ([а-я]+) дней; неоплаченная сумма за партию (\d+) руб", payload["question"])
    assert question_match
    duration, unit, principal = question_match.groups()
    assert (int(duration), int(principal)) == (case.days, case.base)
    principal, duration = Decimal(principal), Decimal(duration)
    trace["input_operands"] = {"principal_rub": str(principal), "days": str(duration), "unit": unit}
    raw, maximum = principal * rate * duration / 100, principal * cap / 100
    calculation = raw if case.behavior == "raw_instead" else min(raw, maximum)
    clause_id = payment["id"]
    locator = "7.2"
    if case.behavior == "wrong_source":
        clause_id = next(p["id"] for p in passages if WRONG_CLAUSE in p["text"])
        locator = "7.1"
    formula = (
        f"По п. {locator} покупатель платит поставщику {percentage(rate)} от неоплаченной суммы "
        f"за каждый день просрочки, максимум {percentage(cap)} от неоплаченной суммы."
    )
    amount = (
        f"По п. {locator} неустойка: {money(principal)} руб. × {percentage(rate)} × {duration} дн. "
        f"= {money(raw)} руб. Предел {percentage(cap)} — {money(maximum)} руб. "
        f"Итого к уплате {money(calculation)} руб."
    )
    claims = [{"text": amount, "evidence": [clause_id]}]
    if case.behavior == "days_unit":
        claims.insert(0, {"text": formula, "evidence": [payment["id"]]})
    trace["draft"] = claims
    return {"status": "partial" if case.behavior == "days_unit" else "answered", "claims": claims,
            "gap": GAP if case.behavior == "days_unit" else "", "look_for": ""}


def verify_for(case, payload, trace):
    passages = {p["id"]: p for p in payload["passages"]}
    raw, maximum, expected = case.numbers()
    accepted = []
    for index, claim in enumerate(payload["claims"]):
        [pid] = claim["evidence"]
        if pid not in passages or "7.2. За просрочку оплаты Товара Покупатель" not in passages[pid]["text"]:
            continue
        if not {percentage(case.rate), percentage(case.cap)} <= set(re.findall(r"\d+(?:,\d+)?%", claim["text"])):
            continue
        if case.behavior == "days_unit":
            if "за каждый день просрочки" in claim["text"] and "Итого к уплате" not in claim["text"]:
                accepted.append(index)
        elif f"= {money(raw)} руб." in claim["text"] and f"— {money(maximum)} руб." in claim["text"] and f"Итого к уплате {money(expected)} руб." in claim["text"]:
            accepted.append(index)
    result = {"supported": accepted, "gap_ok": payload["gap"] == GAP and case.behavior == "days_unit"}
    trace["verifier"] = result
    return result


def run(implementation, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="psb-arithmetic-") as temporary:
        tmp = Path(temporary)
        if implementation.is_file():
            extract = tmp / "implementation"
            extract.mkdir()
            with ZipFile(implementation) as archive:
                for entry in archive.infolist():
                    if entry.filename.endswith(".py") and "/" not in entry.filename:
                        (extract / entry.filename).write_bytes(archive.read(entry))
            implementation = extract
        sys.path.insert(0, str(implementation))
        from assistant import LegalAssistant
        from corpus import load_corpus
        from proxy import Settings
        from solution import write_submission

        rows = []
        timings = []
        settings = Settings(base_url="http://127.0.0.1", key="fixture", model="fixture",
                            embed_model="", rerank_model="", retry_search=False)
        corpora = {("0.1", "10"): load_corpus(BASE)}
        for case in fixture_cases():
            key = (case.rate, case.cap)
            if key not in corpora:
                variant = tmp / f"{case.id}.zip"
                make_variant(variant, *key)
                corpora[key] = load_corpus(variant)
            assistant = LegalAssistant(corpora[key], settings)
            trace = {"id": case.id, "question": case.question(), "chat_calls": Counter()}

            def chat(system, payload, deadline, *, verify=False):
                trace["chat_calls"]["verify" if verify else "draft"] += 1
                assert payload["question"] == case.question()
                return verify_for(case, payload, trace) if verify else draft_for(case, payload, trace)

            started = time.perf_counter()
            with patch.object(assistant.gateway, "chat", side_effect=chat):
                answer = assistant.ask(case.question())
            timings.append(time.perf_counter() - started)
            trace["status"] = answer.status
            trace["answer"] = answer.text
            trace["citations"] = [{"id": c.id, "locator": c.locator, "quote": c.quote} for c in answer.citations]
            trace["csv_answer"] = answer.csv_text()
            trace["result_oracle"] = {label: money(value) for label, value in zip(("raw", "cap", "final"), case.numbers(), strict=True)}
            trace["event_scope_ok"] = trace["event_ranges"] == [["2026-11-15", "2026-11-15"]]
            assert trace["selected_applicability"] != "inapplicable", trace
            assert trace["chat_calls"] == {"draft": 1, "verify": 1}, trace
            if case.behavior in {"wrong_source", "raw_instead"}:
                assert answer.status == "insufficient" and not getattr(answer, "claims", ()), trace
                assert "Итого к уплате" not in answer.csv_text(), trace
                if answer.citations:
                    assert "потенциально относящиеся" in answer.csv_text(), trace
                    assert [c.locator for c in answer.citations] == ["п. 7.2"], trace
            elif case.behavior == "days_unit":
                assert answer.status == "partial" and GAP in answer.csv_text(), trace
                assert "Итого к уплате" not in answer.csv_text(), trace
                assert [c.locator for c in answer.citations] == ["п. 7.2"], trace
            else:
                assert answer.status == "answered" and f"Итого к уплате {trace['result_oracle']['final']}" in answer.csv_text(), trace
                assert [c.locator for c in answer.citations] == ["п. 7.2"], trace
            rows.append(trace)

        final_csv = tmp / "submission.csv"
        write_submission(final_csv, [(row["id"], row["question"]) for row in rows], [row["csv_answer"] for row in rows])
        with final_csv.open(encoding="utf-8", newline="") as stream:
            contents = list(csv.reader(stream))
        assert contents == [["id", "answer"]] + [[row["id"], row["csv_answer"]] for row in rows]
        assert all(len(row["csv_answer"]) <= 10_000 for row in rows)
        report = {"cases": rows, "fixture_chat_calls": sum(sum(row["chat_calls"].values()) for row in rows),
                  "ask_wall_seconds": sum(timings), "ask_p50_seconds": statistics.median(timings),
                  "ask_p95_seconds": max(timings), "csv_rows": len(contents) - 1, "exact_event_scopes": sum(row["event_scope_ok"] for row in rows)}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{len(rows)}/{len(rows)} controlled cases, {report['fixture_chat_calls']} scripted chat calls, CSV {report['csv_rows']} rows")
        print(f"ask total {report['ask_wall_seconds']:.3f}s p50 {report['ask_p50_seconds']:.3f}s p95 {report['ask_p95_seconds']:.3f}s")
        print(f"exact event scope {report['exact_event_scopes']}/{len(rows)}")
        for row in rows:
            print(row["id"], row["status"], row["result_oracle"]["final"] if row["status"] == "answered" else "-",
                  row["verifier"]["supported"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.implementation.resolve(), args.output.resolve())
