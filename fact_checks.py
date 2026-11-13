"""Conservative comparisons of precise facts before publishing model output."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

import corpus


_NUMBER_WORDS = {
    0: r"нол[ьяюе]|нул[ьяюе]",
    1: r"один|одна|одно|одни|одного|одной|одному|одним|одну",
    2: r"два|две|двух|двум|двумя",
    3: r"три|тр[её]х|тр[её]м|тремя",
    4: r"четыре|четыр[её]х|четыр[её]м|четырьмя",
    5: r"пять|пяти|пятью",
    6: r"шесть|шести|шестью",
    7: r"семь|семи|семью",
    8: r"восемь|восьми|восемью",
    9: r"девять|девяти|девятью",
    10: r"десять|десяти|десятью",
    11: r"одиннадцать|одиннадцати",
    12: r"двенадцать|двенадцати",
    13: r"тринадцать|тринадцати",
    14: r"четырнадцать|четырнадцати",
    15: r"пятнадцать|пятнадцати",
    16: r"шестнадцать|шестнадцати",
    17: r"семнадцать|семнадцати",
    18: r"восемнадцать|восемнадцати",
    19: r"девятнадцать|девятнадцати",
    20: r"двадцать|двадцати",
    30: r"тридцать|тридцати",
    40: r"сорок|сорока",
    50: r"пятьдесят|пятидесяти",
    60: r"шестьдесят|шестидесяти",
    90: r"девяносто|девяноста",
}
_WORD_TO_NUMBER = {word: value for value, words in _NUMBER_WORDS.items()
                   for word in words.split("|") if "[" not in word}
_WORD_TO_NUMBER.update({"трех": 3, "трем": 3, "четырех": 4, "четырем": 4})
_WORD_NUMBER = "|".join(_NUMBER_WORDS.values())
_NUMERAL = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[,.]\d+)?"
_DURATION = (r"(?:(?:рабоч|календарн)\w*\s+)?"
             r"(?:дн(?:я|ей|ям|ями|ях)?|день|недел\w*|месяц\w*|час\w*|лет|год(?:а|ов)?)(?!\w)")
_NUMBER = re.compile(
    rf"(?<![\w./-])(?P<value>{_NUMERAL}|(?:{_WORD_NUMBER}))"
    r"(?:\s*(?P<scale>тыс(?:яч[аиу]?|\.)?|млн\.?|миллион\w*|млрд\.?|миллиард\w*))?"
    r"(?![\w/-]|\.\d)(?P<unit>\s*(?:\([^()]{1,45}\)\s*)?(?:руб(?:л\w*)?\b\.?|₽|usd\b|eur\b|доллар\w*\b|евро\b|%|процент\w*\b"
    rf"|{_DURATION}))?",
    re.I,
)
_RANGE = re.compile(
    rf"(?<!\w)(?P<first>{_NUMERAL}|(?:{_WORD_NUMBER}))(?:\s+(?:до|или|и)\s+|\s*[–—-]\s*)"
    rf"(?:{_NUMERAL}|(?:{_WORD_NUMBER}))\s+(?P<unit>{_DURATION})", re.I,
)
_IMPLICIT_UNIT = re.compile(r"\b(?:на\s+неделю|в\s+течение\s+(?P<day>(?:рабочего|календарного)\s+дня))\b", re.I)
_ORG = re.compile(r"\b(?P<form>(?i:ооо|пао|оао|зао|ао|нко|ип))\s+"
                  r"(?:[«\"“](?P<quoted>[^»\"”\n]{2,80})[»\"”]|(?P<bare>[А-ЯЁA-Z][\w-]*(?:\s+[А-ЯЁA-Z][\w-]*){0,3}))")
_GENERALIZATION = re.compile(r"\b(?:все[мх]?|любо(?:й|му|го)|кажд(?:ый|ому|ого))\s+(?:договор|контрагент|спецификац)\w*", re.I)
_SCOPE_LIMIT = re.compile(r"\b(?:только|исключительно|лишь|по\s+договорам|для\s+договоров|за\s+исключением)\b", re.I)
_TIMELINE = re.compile(r"\b(?:вступ\w*\s+в\s+силу|редакц\w*\s+от|действ\w*\s+с|утрат\w*\s+силу\s+с)\s*$", re.I)
_ARITHMETIC = re.compile(
    rf"(?P<left>{_NUMERAL})\s*(?P<lu>%|руб\.?)?\s*(?P<op>[×*+/−-])\s*(?P<right>{_NUMERAL})\s*(?P<ru>%|руб\.?|дн\.?)?"
    rf"(?:\s*(?P<op2>[×*+/−-])\s*(?P<operand>{_NUMERAL})\s*(?P<ou>%|дн\.?)?)?\s*=\s*(?P<result>{_NUMERAL})",
    re.I,
)
_CALCULATION_LIMIT = re.compile(r"\b(?:максимум|предел\w*|ограничени\w*|не\s+более)\b", re.I)
_CONTEXT_DATE = re.compile(r"\b(?:на|для|по состоянию на|подпис\w*|произош\w*|заключ\w*|после)\s*$", re.I)
_HYPOTHETICAL = re.compile(r"\b(?:допускается|допустим\w*|возможн\w*|можно|для|при)\b", re.I)
_CASE_AMOUNT = re.compile(r"\b(?:договор\w*|контракт\w*|плат[её]ж\w*|парт\w*|задолженност\w*)\s+(?:на|суммой|стоимостью)\s*$", re.I)
_THRESHOLD = re.compile(r"\b(?:свыше|менее|более|превыш\w*|порог\w*|предел\w*)\b", re.I)
_DASH = str.maketrans("ёЁ‐‑‒–—−", "ее------")
_CALENDAR_YEAR = re.compile(r"\b(?:19|20)\d{2}\s*(?:год(?:а|у|е|ом)?\b|г\.)", re.I)


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).translate(_DASH).casefold().split())


def _decimal(text: str) -> Decimal:
    value = normalized(text)
    if value in _WORD_TO_NUMBER:
        return Decimal(_WORD_TO_NUMBER[value])
    return Decimal(value.replace(" ", "").replace(",", "."))


def _typed_unit(unit: str) -> str:
    value = normalized(re.sub(r"\([^()]*\)", "", unit))
    if value.startswith(("руб", "₽")):
        return "rub"
    if value.startswith(("доллар", "usd")):
        return "usd"
    if value.startswith(("евро", "eur")):
        return "eur"
    if value.startswith(("%", "процент")):
        return "percent"
    if value.startswith("рабоч"):
        return "workday" if " д" in value else value
    if value.startswith("календарн"):
        return "calendar_day" if " д" in value else value
    if value.startswith(("дн", "день")):
        return "day"
    if value.startswith("недел"):
        return "week"
    if value.startswith("месяц"):
        return "month"
    if value.startswith(("год", "лет")):
        return "year"
    if value.startswith("час"):
        return "hour"
    return "number"


def _extract(text: str) -> tuple[dict[str, set[Decimal]], list[tuple[int, int, date]]]:
    dates = corpus.date_mentions(text)
    # Mask dates and provision numbers so that e.g. п. 5.3 never supplies a 5.3% rate.
    masked = list(text)
    for start, end, _ in dates:
        masked[start:end] = " " * (end - start)
    for match in _CALENDAR_YEAR.finditer(text):
        masked[match.start():match.end()] = " " * (match.end() - match.start())
    for match in re.finditer(r"(?:№|\b(?:пп?|ч|ст)\.|\b(?:пункт|подпункт|стать)\w*\s+)\s*[\w]+(?:[./-][\w]+)*", text, re.I):
        masked[match.start():match.end()] = " " * (match.end() - match.start())
    masked = "".join(masked)
    result: dict[str, set[Decimal]] = {}
    for match in _NUMBER.finditer(masked):
        raw = match["value"]
        if raw.isalpha() and not (match["unit"] or match["scale"]):
            continue
        try:
            number = _decimal(raw)
        except InvalidOperation:
            continue
        scale = normalized(match["scale"] or "")
        if scale:
            number *= 1000 if scale.startswith("тыс") else 10**9 if scale.startswith(("млрд", "миллиард")) else 10**6
        unit = _typed_unit(match["unit"] or "")
        result.setdefault(unit, set()).add(number)
        if scale and unit == "number":
            result.setdefault("scaled", set()).add(number)
    for match in _RANGE.finditer(masked):
        try:
            result.setdefault(_typed_unit(match["unit"]), set()).add(_decimal(match["first"]))
        except InvalidOperation:
            continue
    for match in _IMPLICIT_UNIT.finditer(masked):
        result.setdefault(_typed_unit(match["day"]) if match["day"] else "week", set()).add(Decimal(1))
    return result, dates


def numeric_facts(text: str) -> tuple:
    facts, dates = _extract(text)
    return (tuple((unit, tuple(sorted(numbers))) for unit, numbers in sorted(facts.items())),
            tuple(sorted({value for _, _, value in dates})))


def _contextual_money(text: str, currency: str) -> set[Decimal]:
    values: set[Decimal] = set()
    for match in _NUMBER.finditer(text):
        if _typed_unit(match["unit"] or "") != currency:
            continue
        prefix = text[max(0, match.start() - 40):match.start()]
        if _CASE_AMOUNT.search(prefix):
            money, _ = _extract(match[0])
            values.update(money.get(currency, set()))
    return values


def _organizations(text: str) -> set[tuple[str, str]]:
    parties = set()
    for match in _ORG.finditer(text):
        name = match["quoted"] or match["bare"]
        parties.add((normalized(match["form"]), normalized(name)))
    return parties


def _calculation_results(text: str, evidence: str, sources: dict[str, set], query: dict[str, set]) -> tuple[set[Decimal], set[Decimal]]:
    grounded = set().union(*sources.values(), *query.values())
    percentages = sources.get("percent", set()) | query.get("percent", set())
    results: set[Decimal] = set()
    operands: set[Decimal] = set()
    for match in _ARITHMETIC.finditer(text):
        left, right = _decimal(match["left"]), _decimal(match["right"])
        remaining = [(match["op"], right, match["ru"] == "%")]
        if match["operand"]:
            remaining.append((match["op2"], _decimal(match["operand"]), match["ou"] == "%"))
        left_percent = match["lu"] == "%"
        if left not in (percentages if left_percent else grounded | results):
            continue
        value = left / 100 if left_percent else left
        for operator, operand, percent in remaining:
            if operand not in (percentages if percent else grounded | results):
                break
            operand = operand / 100 if percent else operand
            if operator in ("*", "×"):
                value *= operand
            elif operator == "/" and operand:
                value /= operand
            elif operator == "+":
                value += operand
            elif operator in ("-", "−"):
                value -= operand
            else:
                break
        else:
            if value == _decimal(match["result"]):
                results.add(value)
                operands.update([left, *(operand for _, operand, _ in remaining)])
    if results and _CALCULATION_LIMIT.search(text) and _CALCULATION_LIMIT.search(evidence):
        # A calculation can state a contract cap as a separate sentence. Bind it to a
        # percentage and monetary operand already present in the cited rule and formula.
        limits = re.finditer(r"\b(?:предел\w*|максимум|ограничени\w*)\s+(\d+(?:[,.]\d+)?)\s*%", text, re.I)
        for match in limits:
            percentage = _decimal(match[1])
            if percentage not in sources.get("percent", set()):
                continue
            for principal in operands & (sources.get("rub", set()) | query.get("rub", set())):
                results.add(principal * percentage / 100)
    return results, operands


def fact_issue(text: str, sources: list[str], metadata: list[str], question: str,
               *, correction: bool = False) -> str:
    """Flag a precise mismatch. Semantic verification still checks who/what each value governs."""
    evidence = "\n".join(sources)
    descriptive = "\n".join(metadata)
    facts, dates = _extract(text)
    source_facts, source_dates = _extract(evidence)
    _, metadata_dates = _extract(descriptive)
    query_facts, query_dates = _extract(question)
    in_source = {value for _, _, value in (*source_dates, *metadata_dates)}
    in_query = {value for _, _, value in query_dates}
    # Month/year and year-only anchors are periods, never durations of thousands of years.
    # Exact boundary dates (e.g. "до 1 марта") are checked below against the
    # applicable source dates; the direction of the boundary is semantic.
    claim_periods = {period for period in corpus._event_dates(text)[0]
                     if date.min < period[0] < period[1] < date.max}
    allowed_periods = {period for value in (evidence, descriptive, question)
                       for period in corpus._event_dates(value)[0]}
    if not claim_periods.issubset(allowed_periods):
        return "Год или период не совпадает с источниками или датой события."
    for start, _, value in dates:
        if value in in_source:
            continue
        prefix = text[max(0, start - 42):start]
        if value in in_query and _CONTEXT_DATE.search(prefix) and not _TIMELINE.search(prefix):
            continue
        return "Дата не совпадает с источниками или заданной датой события."

    calculations, operands = _calculation_results(text, evidence, source_facts, query_facts)
    all_source_numbers = set().union(*source_facts.values())
    durations = {"workday", "calendar_day", "day", "week", "month", "year", "hour"}
    for unit, numbers in facts.items():
        supported = source_facts.get(unit, set())
        if unit in {"number", "scaled", "rub", "usd", "eur"}:
            supported |= calculations
        supported |= operands & query_facts.get(unit, set())
        if unit in {"number", "scaled"}:
            supported |= all_source_numbers | set().union(*query_facts.values())
        if unit in {"rub", "usd", "eur"}:
            supported |= source_facts.get("scaled", set())
            if (not correction and source_facts.get(unit) and _THRESHOLD.search(text)
                    and source_facts[unit] & numbers):
                supported |= _contextual_money(text, unit) & query_facts.get(unit, set())
        if unit == "day":
            supported |= source_facts.get("workday", set()) | source_facts.get("calendar_day", set())
            supported |= operands & (query_facts.get("workday", set()) | query_facts.get("calendar_day", set()))
        if numbers <= supported:
            continue
        # A case parameter may appear in a conditional application of the source rule.
        hypothetical = (unit in {"day", "month", "week", "year", "hour"} and not correction
                        and _HYPOTHETICAL.search(text) and any(key in source_facts for key in durations))
        if hypothetical and numbers <= supported | query_facts.get(unit, set()):
            continue
        return "Число, сумма или единица срока не подтверждены указанными источниками."

    source_refs = set(corpus.legal_references(descriptive + "\n" + evidence))
    source_numbers = {value for kind, value in source_refs
                      if kind in {"contract", "document", "amendment", "specification", "order", "letter", "note", "regulation"}}
    if any((kind, value) not in source_refs and not (kind == "document" and value in source_numbers)
           and ("document", value) not in source_refs for kind, value in corpus.legal_references(text)):
        return "Номер документа, пункта или статьи не совпадает с источниками."
    if not _organizations(text).issubset(_organizations(descriptive + "\n" + evidence)):
        return "Названный контрагент не совпадает с источниками."
    if _GENERALIZATION.search(text) and _SCOPE_LIMIT.search(evidence) and not _GENERALIZATION.search(evidence):
        return "Утверждение расширяет область действия источника."
    return ""
