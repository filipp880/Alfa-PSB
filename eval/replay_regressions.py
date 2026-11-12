"""Replay three fixed-draft regressions; this is not a language-model quality score.

Example: python eval/replay_regressions.py 3 4 5 . --output dist/psb6/regression-comparison.json
Targets are submission numbers, ZIP paths, or extracted runtime directories.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ("solution.py", "assistant.py", "corpus.py", "proxy.py", "telemetry.py", "requirements.txt")
SOURCE_TEXT = {
    "P80": "3.1. Согласование оформляется в листе согласования на бумажном носителе.",
    "P78": "2.3. Договоры на сумму свыше 1 000 000 (одного миллиона) рублей дополнительно согласуются с финансовым директором.",
    "P90": "2.3. Договоры на сумму свыше 3 000 000 (трёх миллионов) рублей дополнительно согласуются с финансовым директором.",
    "P94": "3.2. Подписанный оригинал договора передаётся в юридический отдел для хранения в течение 3 (трёх) рабочих дней с даты подписания.",
    "P104": "Сканированные копии договоров размещаются в СЭД не позднее 1 (одного) рабочего дня после поступления оригинала в архив.",
}


def fixtures():
    paper = "Согласование оформляется на бумажном листе согласования."
    changed = "Порог вырос с суммы свыше 1 млн руб. в прежней редакции до суммы свыше 3 млн руб. в новой."
    original = "Оригинал передают в юридический отдел в течение 3 рабочих дней с даты подписания."
    scan = "Скан размещают в СЭД не позднее 1 рабочего дня после поступления оригинала в архив."
    claim = lambda text, *ids: {"text": text, "evidence": list(ids)}
    return [
        ("signature-date", "Какой порядок согласования по п. 3.1 регламента действовал на дату подписания договора от "
         "28 февраля 2025 года, если товар доставили 2 марта 2025 года?",
         [{"status": "answered", "claims": [claim(paper, "P80")], "gap": ""}], {paper: {"P80"}}),
        ("historical-comparison", "Как изменился порог согласования с финансовым директором на 2 марта 2025 года? "
         "Назовите прежний и новый порог по п. 2.3 регламента.",
         [{"status": "answered", "claims": [claim(changed, "P78", "P90")], "gap": ""}],
         {changed: {"P78", "P90"}}),
        ("partial-retry", "В июне 2025 договор ООО «Меридиан» подписан. Назовите три отдельных срока и точки отсчёта: "
         "передача подписанного оригинала, размещение скана и выдача оригинала сотруднику другого подразделения. Как оформляется выдача?",
         [{"status": "insufficient", "claims": [claim(original, "P94")],
           "gap": "Нужно найти срок размещения скана и порядок выдачи оригинала.",
           "look_for": "Сканированные копии договоров размещаются в СЭД после поступления оригинала в архив"},
          {"status": "partial", "claims": [claim(original, "P94"), claim(scan, "P104")],
           "gap": "Уточните порядок выдачи оригинала сотруднику другого подразделения."}],
         {original: {"P94"}, scan: {"P104"}}),
    ]


def worker(target: Path, kb: Path, label: str) -> dict:
    sys.path.insert(0, str(target))
    from assistant import LegalAssistant
    from corpus import load_corpus
    from proxy import Settings

    corpus = load_corpus(kb)
    for pid, expected in SOURCE_TEXT.items():
        if pid not in corpus.passages or corpus.passages[pid].text != expected:
            raise ValueError(f"The fixed fixture requires the official source text at {pid}")
    results = []
    for name, question, drafts, required in fixtures():
        rag = LegalAssistant(corpus, Settings(base_url="http://localhost", key="fixture", model="fixture",
                                             retry_search=len(drafts) > 1))
        generated, audit, counts = iter(drafts), [], {"generation": 0, "verification": 0}
        gaps = {item.get("gap") for item in drafts if item.get("gap")}
        initial_anchors = None

        def chat(system, payload, deadline, *, verify=False):
            nonlocal initial_anchors
            available = {p["id"] for p in payload["passages"]}
            if not verify:
                counts["generation"] += 1
                draft = deepcopy(next(generated))
                if initial_anchors is None:
                    initial_anchors = deepcopy(payload.get("anchors"))
                audit.append({"stage": "draft", "anchors": payload.get("anchors"),
                              "sources": payload["passages"], "draft": deepcopy(draft)})
                return draft

            def supported(row):
                ids = required.get(row.get("text"))
                return ids is not None and set(row.get("evidence", [])) == ids and ids <= available

            counts["verification"] += 1
            accepted = [row for row in payload["claims"] if supported(row)]
            accepted_texts = {row["text"] for row in accepted}
            verdict = {"supported": [row["index"] for row in accepted], "gap_ok": payload.get("gap") in gaps}
            previous = payload.get("previous_claims", [])
            if previous:
                verdict.update(
                    previous_supported=[row["index"] for row in previous if supported(row)],
                    previous_covered=[row["index"] for row in previous if supported(row) and row["text"] in accepted_texts],
                    adds_information=bool(accepted_texts - {row["text"] for row in previous}),
                    consistent=payload.get("anchors") == initial_anchors and all(supported(row) for row in accepted),
                )
            audit.append({"stage": "verification", "claims": payload["claims"],
                          "previous_claims": previous, "verdict": verdict.copy()})
            return verdict

        rag.gateway.chat = chat
        started = time.monotonic()
        answer = rag.ask(question)
        scope = rag.scope_for(question, [])
        csv = answer.csv_text()
        citation_ids = {c.id for c in answer.citations}
        results.append({"name": name, "question": question,
                        "scope": {"event_role": scope.event_role, "comparison": scope.comparison,
                                  "date_ranges": [[str(a), str(b)] for a, b in scope.date_ranges]},
                        "audit": audit, "status": answer.status, "answer": csv,
                        "citations": sorted(citation_ids), "calls": counts,
                        "controlled_seconds": round(time.monotonic() - started, 6),
                        "fixed_facts_complete": all(text in csv and ids <= citation_ids for text, ids in required.items())})
    return {"version": label, "mode": "controlled-no-model", "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="Numbers (3 5 6), ZIP files, or runtime directories (.)")
    parser.add_argument("--kb", type=Path, default=ROOT / "data" / "knowledge-base.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker.resolve(), args.kb.resolve(), args.label), ensure_ascii=False))
        return
    if not args.targets:
        parser.error("Provide at least one submission number, ZIP file, or runtime directory")
    report = {"mode": "controlled-no-model", "python": sys.version.split()[0],
              "kb_sha256": hashlib.sha256(args.kb.read_bytes()).hexdigest(), "versions": []}
    for target in args.targets:
        source = (ROOT / "submissions" / f"{target}.zip" if target.isdecimal() else Path(target)).resolve()
        with TemporaryDirectory(prefix="psb-regression-") as temporary:
            runtime = source
            if source.is_file():
                runtime = Path(temporary)
                with ZipFile(source) as archive:
                    for filename in RUNTIME:
                        if archive.namelist().count(filename) != 1:
                            raise ValueError(f"Expected one top-level {filename} in {source}")
                        (runtime / filename).write_bytes(archive.read(filename))
            hashes = {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in RUNTIME}
            command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                       "--worker", str(runtime), "--label", target, "--kb", str(args.kb.resolve())]
            run = subprocess.run(command, cwd=runtime, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                 capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
            result = json.loads(run.stdout)
            result.update(source=str(source), runtime_sha256=hashes)
            report["versions"].append(result)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        for version in report["versions"]:
            print(version["version"], [(c["name"], c["status"], c["fixed_facts_complete"]) for c in version["cases"]])
    else:
        print(rendered)


if __name__ == "__main__":
    main()
