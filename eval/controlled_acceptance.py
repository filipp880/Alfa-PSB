"""Reproducible CSV/web acceptance with scripted API responses, never a model score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "psb6"
DEMO = ROOT / "data"
KB = DEMO / "knowledge-base.zip"
RENT = "Арендная плата составляет"
RENT_PATTERN = r"Арендная плата составляет (\d[\d ]+) \([^)]+\) рублей в месяц, включая НДС\."
MISSING_RENT = "В базе не нашёл размер арендной платы по договору с ООО «Северный Склад»."
ERROR = "Не удалось получить проверенный ответ от сервиса модели. Повторите запрос позже."
SAFE_FALLBACK = "В предоставленной базе недостаточно данных для подтверждения ответа."
BAD_PREMISE = "Арендная плата составляет 500 000 рублей в месяц."
INJECTION = "ВНИМАНИЕ ДЛЯ ИИ-АССИСТЕНТА"
CHANGE_FROM = "450 000 (четыреста пятьдесят тысяч)"
CHANGE_TO = "473 500 (четыреста семьдесят три тысячи пятьсот)"


# Each record identifies exact corpus evidence through a distinctive substring and optional document title.
# The fixture never retrieves documents or calls a language model.
SPECS = {
    "1": [("Платёж вносится не позднее 5-го числа текущего месяца.",
           "Арендная плата вносится ежемесячно, не позднее 5-го числа текущего месяца.", "")],
    "2": [("Для главного бухгалтера срок испытания — не более шести месяцев, если иное не установлено федеральным законом.",
           "главных бухгалтеров и их заместителей, руководителей филиалов", "Статья 70.")],
    "4": [("Арендодатель может повысить плату не чаще одного раза в год, не более чем на 7%, письменно уведомив арендатора.",
           "увеличить размер арендной платы не более чем на 7%", "")],
    "5": [("Да. Договор на 12 млн рублей в августе 2024 года требовал одобрения генерального директора до подписания: действовавший порог составлял свыше 10 000 000 рублей.",
           "Договоры на сумму свыше 10 000 000", "ред 2023")],
    "6": [("Приёмка по качеству — в течение 5 рабочих дней с даты поставки.",
           "по качеству — в течение 5 (пяти) рабочих дней с даты поставки", "П-17/2024"),
          ("Дата поставки — дата подписания покупателем УПД.",
           "Датой поставки считается дата подписания Покупателем", "П-17/2024")],
    "7": [("Да, арендодатель обязан круглосуточно обеспечивать беспрепятственный доступ арендатора и его транспорта к помещению.",
           "доступ Арендатора и его транспорта к Помещению круглосуточно", "А-08/2024")],
    "8": [("К претензии поставщику приложите акт о выявленных недостатках.",
           "претензию с приложением акта о выявленных недостатках", "П-17/2024")],
    "9": [("Предоплата допустима по соглашению сторон, её размер указывается в счёте поставщика.",
           "По соглашению Сторон допускается предоплата", "ПТ-05/2023"),
          ("Общий срок оплаты — 45 календарных дней с даты поставки.",
           "45 (сорока пяти) календарных дней с даты поставки", "ПТ-05/2023")],
    "10": [("Согласование инициирует руководитель подразделения, в интересах которого заключается договор.",
            "Инициатором согласования является руководитель подразделения", "ред 2025")],
    "11": [("Нет. Оригинал выдают сотруднику другого подразделения по заявке в СЭД максимум на 3 рабочих дня.",
            "по заявке в СЭД на срок не более 3 (трёх) рабочих дней", "архиве договоров")],
    "12": [("Да, во время испытания на сотрудника распространяются трудовое законодательство, коллективный договор, соглашения и локальные нормативные акты.",
            "В период испытания на работника распространяются положения трудового законодательства", "Статья 70.")],
    "13": [("Отзыв согласия сам по себе не означает немедленного удаления всех данных. Оператор вправе продолжить обработку при наличии оснований, на которые отсылает статья 9.",
            "вправе продолжить обработку персональных данных без согласия", "152-ФЗ")],
    "14": [("Да, срочный договор допустим, если в силу природных условий работа может выполняться только в течение определённого сезона.",
            "для выполнения сезонных работ, когда в силу природных условий", "Статья 59.")],
    "15": [("При прекращении договора помещение нужно вернуть по акту с учётом нормального износа.",
            "возвращает Помещение по акту", "А-08/2024")],
    "16": [("Упаковка и доставка до склада покупателя уже входят в цену товара в спецификации.",
            "стоимость упаковки и доставки до склада Покупателя", "П-17/2024")],
    "17": [("Гарантийный срок определяет производитель и указывает в гарантийном талоне.",
            "Гарантийный срок на Товар устанавливается производителем", "ПТ-05/2023")],
    "18": [("Подписанный оригинал договора передают в юридический отдел в течение 3 рабочих дней с даты подписания.",
            "Подписанный оригинал договора передаётся в юридический отдел", "ред 2025")],
}
GAPS = {
    "3": "В базе нет положения, по которому можно установить размер пособия по беременности и родам.",
    "13": "В базе нет упомянутых норм об основаниях обработки без согласия, а обстоятельства конкретной обработки неизвестны.",
    "17": "Для точного срока на конкретный товар нужен его гарантийный талон.",
}
RETRY_CASES = {
    "holdout-approval-february": {
        "claims": [
            ("Финансовый директор должен согласовать договор на 2 млн рублей: порог в действовавшей редакции — свыше 1 млн рублей.",
             "свыше 1 000 000 (одного миллиона) рублей дополнительно согласуются с финансовым директором", "ред 2023"),
            ("Согласование оформляется на бумажном листе согласования.",
             "в листе согласования на бумажном носителе", "ред 2023"),
        ],
        "hint": "Согласование оформляется в листе согласования на бумажном носителе",
        "gap": "Нужно найти положение о форме согласования в редакции 2023 года.",
    },
    "holdout-archive-three-clocks": {
        "claims": [
            ("Оригинал передают в юротдел в течение 3 рабочих дней с даты подписания.",
             "передаётся в юридический отдел для хранения в течение 3 (трёх) рабочих дней с даты подписания", "ред 2025"),
            ("Для выдачи оформляют заявку в СЭД, оригинал выдают на срок не более 3 рабочих дней.",
             "по заявке в СЭД на срок не более 3 (трёх) рабочих дней", "архиве договоров"),
            ("Скан размещают в СЭД не позднее 1 рабочего дня после поступления оригинала в архив.",
             "не позднее 1 (одного) рабочего дня после поступления оригинала в архив", "архиве договоров"),
        ],
        "hint": "Сканированные копии договоров размещаются в СЭД после поступления оригинала в архив",
        "gap": "Нужно найти срок размещения скана после передачи оригинала в архив.",
    },
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream, strict=True))


def last_question(question):
    return question.rsplit("\nПользователь: ", 1)[-1]


DEMO_INPUT = []
QUESTION_IDS = {}
RETRY_INPUT = [(case["id"], case["question"]) for case in
               json.loads((ROOT / "eval" / "holdout.json").read_text(encoding="utf-8"))
               if case["id"] in RETRY_CASES]
QUESTION_IDS.update({question: case_id for case_id, question in RETRY_INPUT})


def normalize(value):
    return " ".join(value.split())


def draft_answer(case_id, passages):
    claims = []
    missing = []
    gap = GAPS.get(case_id, "")
    retry_case = RETRY_CASES.get(case_id)
    specs = retry_case["claims"] if retry_case else SPECS.get(case_id, [])
    for claim, snippet, title in specs:
        passage = next((p for p in passages if p.get("applicability") != "inapplicable"
                        and snippet in p["text"] and title in p["metadata"]), None)
        if passage is None:
            missing.append(snippet)
        else:
            claims.append({"text": claim, "evidence": [passage["id"]]})
    if case_id in {"1", "4"}:
        rent = next(((p["id"], match[1]) for p in passages if
                     p.get("applicability") != "inapplicable" and "А-08/2024" in p["metadata"]
                     and (match := re.search(RENT_PATTERN, p["text"]))), None)
        if rent:
            claims.insert(0, {"text": f"Арендная плата составляет {rent[1]} рублей в месяц, включая НДС.",
                              "evidence": [rent[0]]})
        else:
            missing.append(RENT)
            gap = MISSING_RENT
    hint = ""
    if missing and retry_case:
        gap = retry_case["gap"]
        hint = retry_case["hint"]
    elif missing and not gap:
        gap = "В полученных положениях не хватает основания для полного ответа."
    status = ("partial" if gap else "answered") if claims else "insufficient"
    return {"status": status, "claims": claims, "gap": gap, "look_for": hint}, missing


class FixtureGateway(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != "Bearer loopback-fixture-only":
            self.send_error(401)
            return
        with self.server.lock:
            self.server.requests[self.path] += 1
        if self.path in {"/embeddings", "/rerank"}:
            self.send_error(503)
            return
        if self.path != "/chat/completions" or payload["model"] != "fixture-no-model":
            self.send_error(400)
            return
        query = json.loads(payload["messages"][-1]["content"])
        case_id = QUESTION_IDS.get(query["question"])
        if case_id is None:
            self.send_error(400)
            return
        passages = query["passages"]
        if any(INJECTION in p["text"] for p in passages):
            with self.server.lock:
                self.server.injected += 1
        draft, missing = draft_answer(case_id, passages)
        verify = "claims" in query
        phase = "verify" if verify else "draft"
        repair = "короткий законченный JSON" in payload["messages"][0]["content"]
        finish = "stop"
        if verify:
            allowed = {(c["text"], tuple(c["evidence"])) for c in draft["claims"]}
            answer = {"supported": [c["index"] for c in query["claims"]
                                    if (c["text"], tuple(c["evidence"])) in allowed],
                      "gap_ok": bool(query.get("gap") and query["gap"] == draft["gap"])}
            if query.get("previous_claims"):
                approved = {(c["text"], tuple(c["evidence"])) for c in query["claims"]
                            if c["index"] in answer["supported"]}
                previous = {(c["text"], tuple(c["evidence"])) for c in query["previous_claims"]}
                answer.update(
                    previous_supported=[c["index"] for c in query["previous_claims"]
                                        if (c["text"], tuple(c["evidence"])) in allowed],
                    previous_covered=[c["index"] for c in query["previous_claims"]
                                      if (c["text"], tuple(c["evidence"])) in approved],
                    adds_information=bool(approved - previous), consistent=True,
                )
        else:
            answer = draft
            if case_id == "4" and draft["claims"] and draft["claims"][0]["text"].startswith(RENT):
                answer["claims"] += [{"text": BAD_PREMISE, "evidence": draft["claims"][0]["evidence"]},
                                     {"text": "Пункт выдуман.", "evidence": ["P999999"]}]
        content = json.dumps(answer, ensure_ascii=False)
        if not verify and case_id == "3" and not repair:
            content = "   "  # DeepSeek JSON mode may return blank content with finish_reason=stop.
        if not verify and case_id == "18" and not repair:
            content, finish = '{"status":', "length"
        if self.server.force_error and case_id == "3" and not verify:
            content, finish = "", "content_filter"
        trace = {"id": case_id, "phase": phase, "repair": repair, "finish": finish, "missing": missing,
                 "input_claims": query.get("claims", []), "output": answer if content.strip() else None,
                 "anchors": query.get("anchors", {}),
                 "passages": [p["id"] for p in passages], "max_tokens": payload["max_tokens"]}
        with self.server.lock:
            self.server.traces.append(trace)
        result = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_gateway(*, force_error=False):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureGateway)
    server.requests, server.traces, server.lock = Counter(), [], Lock()
    server.injected, server.force_error = 0, force_error
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def environment(data, gateway, *, fail_search=False, retry_search=False):
    names = ["SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT"]
    return {**{name: os.environ[name] for name in names if name in os.environ},
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "NO_PROXY": "127.0.0.1,localhost", "DATA_DIR": str(data),
            "LLM_BASE_URL": f"http://127.0.0.1:{gateway.server_port}",
            "LLM_API_KEY": "loopback-fixture-only", "LLM_MODEL": "fixture-no-model",
            "MAX_WORKERS": "6", "MAX_TOKENS": "1500", "TEMPERATURE": "0", "TOP_K": "18",
            "EMB_MODEL": "fixture-503" if fail_search else "off",
            "RERANKER_MODEL": "fixture-503" if fail_search else "",
            "RAG_RETRY_SEARCH": str(int(retry_search))}


def checked_extract(archive, working):
    assert archive.stat().st_size <= 128 * 1024 * 1024
    with ZipFile(archive) as stream:
        names = stream.namelist()
        assert len(names) == len(set(names)) and stream.testzip() is None, names
        assert {"solution.py", "requirements.txt"} <= set(names), names
        assert all(Path(name).name == name and (name in {"requirements.txt", "corpus-index.json"}
                                                or name.endswith(".py"))
                   for name in names), names
        if "corpus-index.json" in names:
            manifest = json.loads(stream.read("corpus-index.json"))["manifest"]
            assert manifest["documents"] > 0 and manifest["passages"] > 0
        allowed = set((DEMO / "requirements.txt").read_text(encoding="utf-8-sig").splitlines())
        declared = stream.read("requirements.txt").decode("utf-8").splitlines()
        assert all(line in allowed for line in declared if line and not line.startswith("#")), declared
        stream.extractall(working)
    return {"names": names, "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def run_solution(working, env):
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "solution.py"], cwd=working, env=env, timeout=45,
                            capture_output=True, encoding="utf-8", check=True)
    elapsed = time.perf_counter() - start
    path = working / "submission.csv"
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8", errors="strict")
    rows = read_csv(path)
    assert rows[0] == ["id", "answer"], rows[0]
    expected = read_csv(Path(env["DATA_DIR"]) / "questions.csv")[1:]
    assert [row[0] for row in rows[1:]] == [row[0] for row in expected]
    assert all(len(row) == 2 and 0 < len(row[1].strip()) <= 10000 for row in rows[1:])
    metrics = re.search(r"p50 ([\d.]+) с, p95 ([\d.]+) с", result.stderr)
    return dict(rows[1:]), {"wall_seconds": elapsed,
                            "p50_seconds": float(metrics[1]) if metrics else None,
                            "p95_seconds": float(metrics[2]) if metrics else None,
                            "logs": result.stderr.splitlines(),
                            "contract": {"header": rows[0], "rows": len(rows)-1, "utf8": True,
                                         "ids_match_and_unique": True, "csv_strict": True,
                                         "min_chars": min(len(row[1]) for row in rows[1:]),
                                         "max_chars": max(len(row[1]) for row in rows[1:])}}


def prepare_working(temp, archive):
    working = Path(temp) / "work"
    data = Path(temp) / "input"
    data.mkdir()
    archive_info = checked_extract(archive, working)
    return working, data, archive_info


def run_demo(archive, *, fail_search=False, retry_search=False, force_error=False):
    with tempfile.TemporaryDirectory(prefix="psb-controlled-") as temp, start_gateway(force_error=force_error) as gw:
        working, data, archive_info = prepare_working(temp, archive)
        shutil.copyfile(DEMO / "demo_questions.csv", data / "questions.csv")
        shutil.copyfile(KB, data / "knowledge-base.zip")
        try:
            answers, metrics = run_solution(working, environment(data, gw, fail_search=fail_search,
                                                                 retry_search=retry_search))
            shutil.copyfile(working / "submission.csv", OUTPUT / f"controlled-{archive.stem}{'-failures' if fail_search else ''}.csv")
            expected = {case_id: all(text in answers[case_id] for text, *_ in specs)
                        for case_id, specs in SPECS.items()}
            expected["1"] &= "450 000 рублей в месяц" in answers["1"] and "п. 3.1" in answers["1"] and "п. 3.2" in answers["1"]
            expected["3"] = (answers["3"] in {ERROR, SAFE_FALLBACK} if force_error
                             else GAPS["3"] in answers["3"])
            expected["4"] &= ("450 000 рублей в месяц" in answers["4"] and BAD_PREMISE not in answers["4"]
                              and "Пункт выдуман" not in answers["4"])
            expected["13"] &= GAPS["13"] in answers["13"]
            expected["17"] &= GAPS["17"] in answers["17"]
            assert gw.injected == 0, "Embedded instruction reached the generation or verification prompt"
            assert all("[P" not in answer for answer in answers.values())
            assert set(expected) == set(answers)
            return {"archive": archive_info, **metrics, "calls": dict(gw.requests),
                    "calls_by_phase": dict(Counter(trace["phase"] for trace in gw.traces)),
                    "expected": dict(sorted(expected.items(), key=lambda pair: int(pair[0]))),
                    "expected_count": sum(expected.values()), "injection_prompt_occurrences": gw.injected,
                    "answers": answers, "trace": gw.traces}
        finally:
            gw.shutdown()


def mutate(source, target, mode):
    with ZipFile(source) as src, ZipFile(target, "w") as dst:
        changed = 0
        for entry in src.infolist():
            data = src.read(entry)
            if "Договор аренды А-08-2024" in entry.filename:
                original = data.decode("utf-8")
                if mode == "changed":
                    assert original.count(CHANGE_FROM) == 1
                    modified = original.replace(CHANGE_FROM, CHANGE_TO)
                else:
                    modified, count = re.subn(r"(?m)^3\.1\. Арендная плата составляет[^\r\n]+\r?\n?", "", original)
                    assert count == 1
                data = modified.encode("utf-8")
                changed += 1
            dst.writestr(entry, data)
        assert changed == 1


def run_web(working, data, env, question):
    for name in ["web.py", "static"]:
        source, target = ROOT / name, working / name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    with subprocess.Popen([sys.executable, "web.py", "--port", "0"], cwd=working, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8") as process:
        try:
            base = process.stdout.readline().split()[-1]
            with urlopen(base + "/", timeout=10) as response:
                assert b"/app.js" in response.read()
            body = json.dumps({"messages": [{"role": "user", "content": question}]}).encode("utf-8")
            request = Request(base + "/api/ask", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=15) as response:
                answer = json.load(response)
            documents = []
            for citation in answer["citations"]:
                with urlopen(base + "/api/documents/" + quote(citation["document_id"]), timeout=10) as response:
                    document = json.load(response)
                assert normalize(citation["quote"]) in normalize(document["text"])
                assert len(document["text"]) > len(citation["quote"])
                documents.append({"id": document["id"], "title": document["title"],
                                  "length": len(document["text"]), "citation": citation})
            return answer, documents
        finally:
            process.terminate()


def run_mutation(archive, mode):
    question = DEMO_INPUT[1][1]
    expected = {"original": "450 000", "changed": "473 500", "removed": MISSING_RENT}[mode]
    with tempfile.TemporaryDirectory(prefix="psb-perturbed-") as temp, start_gateway() as gw:
        working, data, _ = prepare_working(temp, archive)
        kb = data / "knowledge-base.zip"
        if mode == "original":
            shutil.copyfile(KB, kb)
        else:
            mutate(KB, kb, mode)
        with (data / "questions.csv").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows([("id", "question"), ('контроль,"1"', question)])
        try:
            env = environment(data, gw)
            answers, metrics = run_solution(working, env)
            csv_text = next(iter(answers.values()))
            web_answer, documents = run_web(working, data, env, question)
            assert expected in csv_text and expected in web_answer["text"], (csv_text, web_answer)
            assert "5-го числа" in csv_text and "5-го числа" in web_answer["text"]
            if mode == "removed":
                assert all(number not in csv_text and number not in web_answer["text"] for number in ("450 000", "473 500"))
                assert web_answer["status"] == "partial"
                assert {doc["citation"]["locator"] for doc in documents} == {"п. 3.2"}
            else:
                assert web_answer["status"] == "answered"
                assert {doc["citation"]["locator"] for doc in documents} == {"п. 3.1", "п. 3.2"}
                assert expected in next(doc["citation"]["quote"] for doc in documents if doc["citation"]["locator"] == "п. 3.1")
            return {"csv_text": csv_text, "web": web_answer, "documents": documents,
                    "calls_csv_and_web": dict(gw.requests), "metrics": metrics}
        finally:
            gw.shutdown()


def run_retry_probe(archive, enabled):
    with tempfile.TemporaryDirectory(prefix="psb-retry-") as temp, start_gateway() as gw:
        working, data, _ = prepare_working(temp, archive)
        shutil.copyfile(KB, data / "knowledge-base.zip")
        with (data / "questions.csv").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows([("id", "question"), *RETRY_INPUT])
        try:
            answers, metrics = run_solution(working, environment(data, gw, retry_search=enabled))
            complete = {case_id: all(text in answers[case_id] for text, *_ in spec["claims"])
                        for case_id, spec in RETRY_CASES.items()}
            return {"answers": answers, "complete": complete, **metrics, "calls": dict(gw.requests),
                    "trace": gw.traces}
        finally:
            gw.shutdown()


def main():
    global DEMO, KB, OUTPUT, DEMO_INPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--materials", type=Path, required=True,
                        help="Official materials directory: demo_questions.csv, knowledge-base.zip, requirements.txt")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--require-complete", action="store_true", help="Fail if a scripted demo expectation is missed")
    parser.add_argument("--mutations", action="store_true")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--retry-probe", action="store_true")
    args = parser.parse_args()
    DEMO, OUTPUT = args.materials.resolve(), args.output.resolve()
    KB = DEMO / "knowledge-base.zip"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DEMO_INPUT = read_csv(DEMO / "demo_questions.csv")
    assert DEMO_INPUT[0] == ["id", "question"] and len(DEMO_INPUT) == 19
    assert len({row[0] for row in DEMO_INPUT[1:]}) == 18
    QUESTION_IDS.update({last_question(question): case_id for case_id, question in DEMO_INPUT[1:]})
    import rapidfuzz
    assert rapidfuzz.__version__ == "3.14.1", rapidfuzz.__version__
    print(f"Controlled gateway only; Python {sys.version.split()[0]}, RapidFuzz {rapidfuzz.__version__}")
    initial_kb = hashlib.sha256(KB.read_bytes()).hexdigest()
    for archive in args.archives:
        out = {"demo": run_demo(archive)}
        if args.failures:
            out["failure_demo"] = run_demo(archive, fail_search=True, retry_search=True, force_error=True)
        if args.mutations:
            out["mutation"] = {mode: run_mutation(archive, mode) for mode in ["original", "changed", "removed"]}
        if args.retry_probe:
            off, on = (run_retry_probe(archive, enabled) for enabled in (False, True))
            out["retry_probe"] = {"off": off, "on": on,
                                  "additional_chat_calls": on["calls"]["/chat/completions"] - off["calls"]["/chat/completions"],
                                  "recovered": [case_id for case_id in on["complete"] if
                                                on["complete"][case_id] and not off["complete"][case_id]]}
        assert hashlib.sha256(KB.read_bytes()).hexdigest() == initial_kb
        out["official_kb_sha256"] = initial_kb
        path = OUTPUT / f"controlled-{archive.stem}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in ["demo", "failure_demo"]:
            if name in out:
                item = out[name]
                print(f"{archive.name} {name} {item['expected_count']}/18, "
                      f"wall {item['wall_seconds']:.3f}s, p50/p95 {item['p50_seconds']}/{item['p95_seconds']}, "
                      f"calls {item['calls']}, missing {[key for key, value in item['expected'].items() if not value]}")
        if args.retry_probe:
            probe = out["retry_probe"]
            print(f"{archive.name} retry probe extra chat calls {probe['additional_chat_calls']}, recovered {probe['recovered']}")
        print(path)
        if args.require_complete:
            assert all(out[name]["expected_count"] == 18 for name in ("demo", "failure_demo") if name in out), path


if __name__ == "__main__":
    main()
