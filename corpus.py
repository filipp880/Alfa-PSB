"""Load the supplied corpus and retrieve traceable legal provisions."""

from __future__ import annotations

from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from math import log1p
from pathlib import Path, PurePosixPath
import re
from typing import Iterator
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str
    filename: str
    kind: str = "corporate"
    source_url: str = ""
    snapshot_date: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    edition_note: str = ""
    reference: str = ""


@dataclass(frozen=True, slots=True)
class Passage:
    id: str
    document_id: str
    locator: str
    text: str
    heading: str = ""
    effective_from: date | None = None
    edition_note: str = ""


_MONTHS = "янв фев мар апр ма июн июл авг сен окт ноя дек".split()
_MONTH_PATTERN = (
    r"январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йяе]|июн[ьяе]|июл[ьяе]|"
    r"август[ае]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе]"
)
_DATE = re.compile(
    r"(?<![\d.])(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>\d{4})(?!\d)"
    rf"|\b(?P<text_day>\d{{1,2}})\s+(?P<text_month>{_MONTH_PATTERN})\s+(?P<text_year>\d{{4}})\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_YEAR = re.compile(rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:в|за|на)\s+(20\d{2})\b|\b(20\d{2})\s*(?:год[ауе]?\b|г\.)", re.I)
_WORD = re.compile(r"[а-яa-z]+|\d+(?:[.,]\d+)?")
_ENTITY = re.compile(r"\b[а-яa-z]+(?:-[а-яa-z]+)+\b")
_ORG_QUOTE = re.compile(r"[«\"“]([^»\"”\n]{2,80})[»\"”]")
_ID = re.compile(r"(?<!\w)([а-яa-z]{1,4})-\s*(\d+)(?:[/\-](20\d{2}))?(?!\w)", re.I)
_ARTICLE = re.compile(r"\b(?:статья|стать[еию]|ст\.)\s*(\d+(?:\.\d+)*)", re.I)
_POINT = re.compile(r"\b(?:пункт\w*|п\.)\s*(\d+(?:\.\d+)*)", re.I)
_RATE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:%|процент\w*)", re.I)
_AMOUNT = re.compile(r"\b(\d{1,3}(?:\s\d{3})+)(?=\s*(?:\(|руб))|\b(\d+)\s*(млн|тыс)\b", re.I)
_LAW_FILE = re.compile(r"^(.+?)_ст(\d+(?:\.\d+)*)", re.I)
_CLAUSE = re.compile(r"^[«\"]?(\d+(?:\.\d+)+(?:-\d+)?)\.\s+|^(\d+)\.\s+")
_SUBCLAUSE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?|[а-я])\)\s+", re.I)
_REVISION = re.compile(r"\b(?:действует\s+с|редакция\s+от|вступает\s+в\s+силу\s+с)\s+", re.I)
_EXPIRES = re.compile(r"\bутратил[ао]?\s+силу\s+с\s+", re.I)
_LAW_CHANGE = re.compile(
    r"^(?P<scope>Статья|Часть|Пункт|Подпункт|Абзац|Примечания?)\s+"
    r"(?:(?P<number>\d+(?:\.\d+)*(?:-\d+)?)\s+)?[^:;]{0,80}?"
    r"(?:измен[её]н\w*|дополнен\w*|изложен\w*|утратил\w*\s+силу)"
    r"(?:\s+подпунктом\s+(?P<new_subpoint>\d+(?:\.\d+)*))?\s+с\s+", re.I,
)
_TRANSITION = re.compile(r"^Положения (?:пункта|части)\s+(\d+(?:\.\d+)*)\s+\(в редакции\b", re.I)
_DEFAULT_RATE_QUERY = re.compile(r"\b(?:стандартн\w*|основн\w*|общ\w*|базов\w*)\s+ставк\w*", re.I)
_DEFAULT_RATE_PROVISION = re.compile(r"\b(?:если иное не установлено настоящей статьей|в случаях, не указанных в пунктах)\b", re.I)
_PAYMENT_DEADLINE = re.compile(r"\b(?:срок\w*\s+(?:оплат\w*|расчет\w*)|когда\s+(?:платить|оплатить))\b", re.I)
_LAW_NOTE = re.compile(
    r"^(?:ГАРАНТ\s*:|Информация об изменениях|См\.|Федеральн\w+ закон\w* от\b|"
    r"Об изменени\w* |О конституционно-правовом|О проверке конституционности)", re.I,
)
_ASSISTANT_COMMAND = re.compile(
    r"^\s*(?:внимание\s+для\s+(?:ии[- ]?)?ассистента\s*:|"
    r"(?:system|developer|системное сообщение)\s*:|"
    r"(?:игнорируй(?:те)?|забудь(?:те)?)\s+(?:все\s+)?(?:остальные|предыдущие|системные)\s+"
    r"(?:инструкции|сообщения|документы)\b)", re.I,
)
_STOP_WORDS = frozenset((
    "для при или как это что чем уже если был была было были будет может нужно необходимо"
    " также ли когда они она его ее их них эти этот такой так тогда есть него меня мы вам нас"
    " по от на до за из со во не но да то чтобы для мне надо сразу могу можем можно вопрос"
    " пользователь ассистент подскажите скажите хочу договор документы согласно"
).split())
_SUFFIXES = tuple(sorted(set((
    "иями ями ами иями ями иями остью остей ость ейший ейшая ейшие ировать ированного"
    " ющего ющим иями ому ему его ого ыми ими ыми ая яя ие ые ое ее ой ый ий ым им"
    " ых их ах ях ам ям ов ев ей ом ем ию ью ия ием ие ии ией илось ется ются ается"
    " аются ать ять ить еть ет ут ют ит ёт ал ел ым ам а я ы и у ю е о ь"
).split()), key=lambda suffix: (-len(suffix), suffix)))
_ALIASES = (
    ("аванс", "предоплата", "предварительная оплата"),
    ("гарантия", "гарантийный срок"),
    ("брак", "недостаток", "дефект"),
    ("проверить качество", "приемка по качеству"),
    ("ночь", "круглосуточно", "доступ", "пропускать"),
    ("съезжать", "возвращать помещение", "возврат помещения"),
    ("сдать ключи", "возврат помещения", "возвращает помещение"),
    ("возврат", "возвращает", "возвращать"),
    ("гендиректор", "генеральный директор"),
    ("юротдел", "юридический отдел"),
    ("оплата", "оплачивает", "платить", "платеж"),
    ("переработка", "сверхурочная работа"),
    ("персональные данные", "персданные", "пдн"),
    ("прибыль организации", "налог на прибыль"),
    ("НДФЛ", "налог на доходы физических лиц"),
    ("НДС", "налог на добавленную стоимость"),
    ("МРОТ", "минимальный размер оплаты труда", "минималка"),
)
_LAW_SEARCH_LABELS = {
    ("нк", "145"): "НДС налог на добавленную стоимость освобождение",
    ("нк", "164"): "НДС налог на добавленную стоимость",
    ("нк", "224"): "НДФЛ налог на доходы физических лиц",
    ("нк", "284"): "налог на прибыль организаций",
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().translate(str.maketrans("ё–—−‑", "е----"))


def _stem(word: str) -> str:
    if len(word) < 5 or not word[0].isalpha():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _words(text: str) -> list[str]:
    terms = (_stem(word) for word in _WORD.findall(_normalize(text)) if len(word) >= 2 and word not in _STOP_WORDS)
    return [term for term in terms if term not in _STOP_WORDS]


def _tokens(text: str) -> Counter[str]:
    normalized = _normalize(text)
    tokens = Counter(_words(normalized))
    tokens.update(f"ref:{_reference(match)}" for match in _ID.finditer(normalized))
    tokens.update(f"entity:{entity}" for entity in _ENTITY.findall(normalized))
    tokens.update(f"article:{value}" for value in _ARTICLE.findall(normalized))
    tokens.update(f"point:{value}" for value in _POINT.findall(normalized))
    tokens.update(f"rate:{value.replace(',', '.')}" for value in _RATE.findall(normalized))
    for separated, short, scale in _AMOUNT.findall(normalized):
        amount = int(re.sub(r"\s", "", separated)) if separated else int(short) * (1_000_000 if scale == "млн" else 1_000)
        tokens[f"amount:{amount}"] += 1
    return tokens


_SEARCH_ALIASES = tuple(tuple(tuple(_words(alias)) for alias in group) for group in _ALIASES)


def _query_terms(text: str) -> dict[str, float]:
    exact = _tokens(text)
    weighted = {token: 1.0 for token in exact}
    for group in _SEARCH_ALIASES:
        if any(alias and all(token in exact for token in alias) for alias in group):
            for alias in group:
                for token in alias:
                    weighted.setdefault(token, 0.53)
    return weighted


def _reference(match: re.Match[str]) -> str:
    prefix, number, year = match.groups()
    prefix = _normalize(prefix)
    prefix = {"a": "а", "p": "п", "k": "к", "pt": "пт"}.get(prefix, prefix)
    return f"{prefix}-{number}/{year}" if year else f"{prefix}-{number}"


def _month_number(text: str) -> int:
    month = text.lower()
    return next(index for index, root in enumerate(_MONTHS, 1) if month.startswith(root))


def _dates(text: str) -> list[date]:
    values = []
    for match in _DATE.finditer(text):
        try:
            if match["day"]:
                value = date(int(match["year"]), int(match["month"]), int(match["day"]))
            else:
                value = date(int(match["text_year"]), _month_number(match["text_month"]), int(match["text_day"]))
            values.append((match.start(), value))
        except ValueError:
            continue
    for match in _ISO_DATE.finditer(text):
        try:
            values.append((match.start(), date(*(int(part) for part in match.groups()))))
        except ValueError:
            continue
    return [value for _, value in sorted(values)]


def _first_date(text: str) -> date | None:
    return next(iter(_dates(text)), None)


def _date_after(pattern: re.Pattern[str], text: str) -> date | None:
    match = pattern.search(text)
    return _first_date(text[match.end():]) if match else None


def _date_range(text: str) -> tuple[date, date] | None:
    if value := _first_date(text):
        return value, value
    if match := _MONTH_YEAR.search(text):
        month = _month_number(match[1])
        year = int(match[2])
        return date(year, month, 1), date(year, month, monthrange(year, month)[1])
    if match := _YEAR.search(text):
        year = int(match[1] or match[2])
        return date(year, 1, 1), date(year, 12, 31)
    return None


def _decode_filename(name: str, flags: int) -> str:
    if flags & 0x800 or name.isascii():
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    # Older Windows tools often write Cyrillic in CP866 without setting a ZIP flag.
    candidates = (raw.decode("cp866"), raw.decode("cp1251", errors="replace"))
    return max(candidates, key=lambda value: len(re.findall(r"[а-яё]", value.lower())) - 3 * value.count("�"))


def _read_text(filename: str, raw: bytes) -> str:
    if filename.lower().endswith(".docx"):
        with ZipFile(BytesIO(raw)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        return "\n".join("".join(part.text or "" for part in para.iter(ns + "t")) for para in root.iter(ns + "p"))
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="replace")


def _make_document(doc_id: str, filename: str, text: str) -> Document:
    stem = PurePosixPath(filename).stem
    category, divider, short_name = stem.partition("__")
    short_name = short_name if divider else category
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    law_file = _LAW_FILE.match(short_name)
    if law_file or (lines[0].startswith("Источник:") and _ARTICLE.search(lines[0])):
        title = lines[0].removeprefix("Источник:").strip() if lines[0].startswith("Источник:") else short_name
        url = next((line.removeprefix("URL:").strip() for line in lines[:5] if line.startswith("URL:")), "")
        snapshot = next((_first_date(line) for line in lines[:6] if line.startswith("Дата выгрузки:")), None)
        return Document(doc_id, title, text, filename, "law", url, snapshot)

    title = short_name
    edition_lines = [line for line in lines[:6] if _REVISION.search(line) or _EXPIRES.search(line)]
    edition_note = " ".join(edition_lines)
    effective_from = _date_after(_REVISION, edition_note)
    effective_to = _date_after(_EXPIRES, edition_note)
    first_reference = next(_ID.finditer(_normalize("\n".join(lines[:4]))), None)
    reference = _reference(first_reference) if first_reference else ""
    if short_name.lower().startswith("договор") and (contract := _ID.search(lines[0])):
        title = _ID.sub(lambda _: "№ " + contract[0], title, count=1)
    elif short_name.lower().startswith("доп соглашение"):
        title = " ".join(lines[:2])
    if short_name.lower().startswith("письмо"):
        effective_from = next((_first_date(line) for line in lines if re.match(r"^С\s+\d", line)), effective_from)
    return Document(doc_id, title, text, filename, "corporate", effective_from=effective_from,
                    effective_to=effective_to, edition_note=edition_note, reference=reference)


def _lines(document: Document) -> list[str]:
    return [line.strip() for line in document.text.splitlines() if line.strip() and not _ASSISTANT_COMMAND.match(line)]


def _fragments(text: str, limit: int = 1350) -> list[str]:
    pieces = []
    while len(text) > limit:
        boundary = text.rfind("\n", limit // 2, limit + 1)
        if boundary < 0:
            boundary = text.rfind(" ", limit // 2, limit + 1)
        if boundary < 0:
            boundary = limit
        pieces.append(text[:boundary].rstrip())
        text = text[boundary:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def _corporate_parts(document: Document) -> Iterator[tuple[str, str, str, date | None, str]]:
    lines = _lines(document)
    if document.title.lower().startswith("дополнительное соглашение"):
        changed = re.search(r"\bизложить\s+пункт\s+(\d+(?:\.\d+)+)", document.text, re.I)
        locator = f"изменение п. {changed[1]}" if changed else "дополнительное соглашение"
        yield locator, "Условия изменения договора", "\n".join(lines), None, ""
        return
    if not any(re.match(r"^\d+\.\d+\.\s+", line) for line in lines):
        for index, line in enumerate(lines, 1):
            if index <= 3 or line.startswith(("Руководитель ", "Начальник ")):
                continue
            locator = "Приложение 1" if line.startswith("Приложение 1.") else f"абзац «{_lead(line)}»"
            yield locator, "", line, _first_date(line) if re.match(r"^С\s+\d", line) else None, ""
        return

    heading = "Преамбула"
    locator = "преамбула"
    pending = []
    for line in lines:
        if match := _CLAUSE.match(line):
            if pending:
                yield locator, heading, "\n".join(pending), None, ""
            pending = []
            if match[1]:
                locator = f"п. {match[1]}"
                pending.append(line)
            else:
                heading = line[match.end():]
                locator = f"раздел {match[2]}"
        else:
            pending.append(line)
    if pending:
        yield locator, heading, "\n".join(pending), None, ""


def _lead(text: str) -> str:
    return text if len(text) <= 57 else text[:55].rsplit(" ", 1)[0] + "…"


def _law_parts(document: Document) -> Iterator[tuple[str, str, str, date | None, str]]:
    lines = _lines(document)
    article = _ARTICLE.search(document.title)
    article_number = article[1] if article else ""
    article_label = f"ст. {article_number}" if article_number else "статья"
    article_heading = re.compile(rf"^Статья\s+{re.escape(article_number)}(?:[.\s]|$)", re.I)
    article_edition = []
    scoped_edition = defaultdict(list)
    unscoped_edition = []
    parent = ""
    part_label = "ч." if re.search(r"\bКоАП\b|\b152-ФЗ\b", document.title) else "п."
    for line in lines:
        if line.startswith(("Источник:", "URL:", "Дата выгрузки:")):
            continue
        if change_match := _LAW_CHANGE.match(line):
            change = _first_date(line[change_match.end():])
            if change:
                scope = change_match["scope"].lower()
                number = change_match["number"]
                subpoint = change_match["new_subpoint"]
                change_note = (change, line)
                if scope == "статья":
                    article_edition.append(change_note)
                elif scope in {"часть", "пункт"} and number and not subpoint:
                    scoped_edition[number].append(change_note)
                elif (scope == "подпункт" and number) or subpoint:
                    key = f"{number if subpoint else parent}/{subpoint or number}"
                    scoped_edition[key].append(change_note)
                else:
                    unscoped_edition.append(change_note)
                continue
        if _LAW_NOTE.match(line):
            continue
        if article_heading.match(line):
            line = article_heading.sub("", line, count=1).lstrip(". ")
            if not line or len(line) < 125 and not re.search(r"\b(?:установить|устанавливается)\b", line, re.I):
                continue
        scope = parent
        subpoint = ""
        if match := _CLAUSE.match(line):
            scope = parent = match[1] or match[2]
            locator = f"{article_label}, {part_label} {parent}"
        elif match := _SUBCLAUSE.match(line):
            subpoint = match[1]
            locator = f"{article_label}, {part_label} {parent}, подп. {subpoint}" if parent else f"{article_label}, п. {subpoint}"
        else:
            transition = _TRANSITION.match(line)
            if transition:
                scope = transition[1]
                locator = f"{article_label}, примечание о применении {part_label} {scope}"
            else:
                prefix = f"{article_label}, {part_label} {parent}" if parent else article_label
                locator = f"{prefix}, абзац «{_lead(line)}»"
        notes = article_edition + scoped_edition.get(scope, []) + unscoped_edition
        if subpoint:
            notes += scoped_edition.get(f"{scope}/{subpoint}", [])
        eligible = [value for value, _ in notes if not document.snapshot_date or value <= document.snapshot_date]
        effective = max(eligible, default=None)
        note = "\n".join(text for _, text in notes)
        yield locator, "", line, effective, note
        unscoped_edition = []


def _document_tokens(document: Document) -> Counter[str]:
    short_name = PurePosixPath(document.filename).stem.rsplit("__", 1)[-1]
    label = _LAW_FILE.match(short_name)
    law_terms = _LAW_SEARCH_LABELS.get((label[1].lower(), label[2]), "") if label else ""
    tokens = _tokens(f"{document.title} {short_name.replace('_', ' ')} {law_terms}")
    if document.reference:
        tokens[f"ref:{document.reference}"] = 1
    return tokens


class Corpus:
    def __init__(self, documents: dict[str, Document], passages: dict[str, Passage]) -> None:
        self.documents = documents
        self.passages = passages
        self._body = {key: _tokens(passage.text) for key, passage in passages.items()}
        self._headings = {key: _tokens(f"{passage.heading} {passage.locator}") for key, passage in passages.items()}
        self._titles = {key: _document_tokens(document) for key, document in documents.items()}
        self._length = {key: sum(terms.values()) for key, terms in self._body.items()}
        self._average_length = max(1.0, sum(self._length.values()) / len(passages))
        frequencies = Counter(token for terms in self._body.values() for token in terms)
        self._idf = {token: log1p((len(passages) - count + 0.5) / (count + 0.5))
                     for token, count in frequencies.items()}
        self._by_document = defaultdict(list)
        for passage in passages.values():
            self._by_document[passage.document_id].append(passage)

    def context_text(self, passage: Passage) -> str:
        document = self.documents[passage.document_id]
        metadata = [document.title, passage.locator, passage.heading]
        if document.snapshot_date:
            metadata.append(f"Дата выгрузки {document.snapshot_date:%d.%m.%Y}. В базе доступен текст этой редакции.")
        effective = passage.effective_from or document.effective_from
        if effective:
            metadata.append(f"Указанная дата действия редакции или изменения {effective:%d.%m.%Y}.")
        if document.effective_to:
            metadata.append(f"Редакция утратила силу с {document.effective_to:%d.%m.%Y}.")
        if document.edition_note:
            metadata.append(document.edition_note)
        if passage.edition_note:
            metadata.append(f"Примечания об изменениях из выгрузки (не текст нормы): {passage.edition_note}")
        return "\n".join(filter(None, (*metadata, passage.text)))

    def related(self, passage: Passage) -> list[Passage]:
        document = self.documents[passage.document_id]
        if document.kind == "law":
            if re.search(r"(?:п|ч)\.\s*\d", passage.locator):
                return [candidate for candidate in self._by_document[document.id]
                        if candidate.locator.startswith("ст.") and candidate != passage
                        and (candidate.locator.replace("примечание о применении ", "") == passage.locator
                             or passage.locator.replace("примечание о применении ", "") == candidate.locator)]
            return []
        clause = re.search(r"\bп\.\s*(\d+(?:\.\d+)+)", passage.locator)
        result = []
        for other in self.documents.values():
            if other.id == document.id or other.kind != "corporate":
                continue
            version = document.edition_note and other.edition_note and (
                document.title.split(" ред ")[0].casefold() == other.title.split(" ред ")[0].casefold()
            )
            same_reference = document.reference and document.reference == other.reference
            for candidate in self._by_document[other.id]:
                if (version or same_reference) and clause and re.search(
                    rf"\bп\.\s*{re.escape(clause[1])}(?![.\d])", candidate.locator
                ):
                    result.append(candidate)
                elif clause and document.edition_note and "в дополнение к порядку" in candidate.text.lower():
                    if re.search(rf"\bп\.\s*{re.escape(clause[1])}(?![.\d])", candidate.text):
                        operative = [item for item in self._by_document[other.id] if item.effective_from]
                        result.extend(operative)
                        result.append(candidate)
                        result.extend(item for item in self._by_document[other.id] if re.search(
                            r"\b(?:заключение|заключения)\s+службы\s+комплаенс\b", item.text, re.I
                        ))
                elif other.edition_note and "в дополнение к порядку" in passage.text.lower():
                    if re.search(rf"(?<!\w){re.escape(candidate.locator)}(?![.\d])", passage.text):
                        result.append(candidate)
        return list(dict.fromkeys(result))

    def search(self, query: str, limit: int = 24) -> list[Passage]:
        terms = _query_terms(query)
        if not terms or limit <= 0:
            return []
        query_date = _date_range(query)
        normalized = _normalize(query)
        default_rate = _DEFAULT_RATE_QUERY.search(normalized)
        payment_deadline = _PAYMENT_DEADLINE.search(normalized)
        entities = tuple(_ORG_QUOTE.findall(normalized))
        query_entities = {token for token in terms if token.startswith(("entity:", "ref:", "article:"))}
        document_scores = {}
        for key, document in self.documents.items():
            title = self._titles[key]
            overlap = sum(min(4.0, self._idf.get(token, 3.0)) * weight
                          for token, weight in terms.items() if token in title)
            exact = len(query_entities & title.keys()) * 4.0
            document_title = _normalize(document.title)
            exact += 4.0 * sum(bool(re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", document_title))
                               for entity in entities)
            factor = 1.0
            if document.effective_to and not query_date:
                factor = 0.65
            if query_date and (document.effective_from or document.effective_to):
                start, end = query_date
                incompatible = ((document.effective_from and end < document.effective_from)
                                or (document.effective_to and start >= document.effective_to))
                factor = 0.58 if incompatible else 1.25
            document_scores[key] = (min(12.0, overlap * 0.67) + exact, factor)

        scores = []
        for key, passage in self.passages.items():
            body = self._body[key]
            heading = self._headings[key]
            title = self._titles[passage.document_id]
            normalizer = 1.35 * (0.32 + 0.68 * self._length[key] / self._average_length)
            score = 0.0
            for token, weight in terms.items():
                if count := body.get(token, 0):
                    body_weight = weight * (0.4 if token in title else 1.0)
                    score += body_weight * self._idf[token] * (2.35 * count / (count + normalizer))
                if token in heading:
                    score += 0.46 * weight * min(4.0, self._idf.get(token, 3.0))
            if passage.locator == "преамбула" or passage.heading == "Реквизиты и подписи сторон":
                score *= 0.55
            if default_rate and _DEFAULT_RATE_PROVISION.search(_normalize(passage.text)):
                score += 9.0
            if payment_deadline and self.documents[passage.document_id].kind == "corporate":
                text = _normalize(passage.text)
                if re.search(r"\b(?:оплачива\w*|оплат\w*|внос\w*)\b", text) and re.search(
                    r"\b(?:в течение|не позднее|ежемесячно)\b", text
                ):
                    score += 5.0
            title_score, factor = document_scores[passage.document_id]
            if score > 0.0 or title_score >= 4.0:
                scores.append(((score + title_score) * factor, key))
        scores.sort(key=lambda item: -item[0])
        return [self.passages[key] for _, key in scores[:limit]]


def load_corpus(path: Path) -> Corpus:
    """Read documents without extracting archive members to the filesystem."""
    documents = {}
    passages = {}
    with ZipFile(path) as archive:
        entries = []
        for member in archive.infolist():
            filename = _decode_filename(member.filename, member.flag_bits).replace("\\", "/")
            if member.is_dir() or filename.startswith("__MACOSX/"):
                continue
            if PurePosixPath(filename).suffix.lower() in {".txt", ".md", ".docx"}:
                entries.append((filename, member))
        for filename, member in sorted(entries, key=lambda entry: entry[0].casefold()):
            text = _read_text(filename, archive.read(member)).strip()
            if not text:
                continue
            doc_id = f"D{len(documents) + 1}"
            document = _make_document(doc_id, PurePosixPath(filename).name, text)
            documents[doc_id] = document
            parts = _law_parts(document) if document.kind == "law" else _corporate_parts(document)
            for locator, heading, content, effective_from, edition_note in parts:
                for fragment in _fragments(content):
                    key = f"P{len(passages) + 1}"
                    passages[key] = Passage(key, doc_id, locator, fragment, heading, effective_from, edition_note)
    if not passages:
        raise ValueError("В архиве базы знаний нет читаемых положений.")
    return Corpus(documents, passages)
