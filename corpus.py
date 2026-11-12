"""Load the supplied corpus and retrieve traceable legal provisions."""

from __future__ import annotations

from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from difflib import SequenceMatcher
from io import BytesIO
from math import log1p
from pathlib import Path, PurePosixPath
import re
from typing import Iterator
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile

try:
    from rapidfuzz.fuzz import ratio as _name_similarity
except ImportError:
    def _name_similarity(left: str, right: str) -> float:
        return 100.0 * SequenceMatcher(None, left, right, autojunk=False).ratio()


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
    effective_event: str = ""


@dataclass(frozen=True, slots=True)
class Passage:
    id: str
    document_id: str
    locator: str
    text: str
    heading: str = ""
    effective_from: date | None = None
    edition_note: str = ""


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Evidence anchors and event dates extracted without treating them as facts."""

    document_ids: tuple[str, ...] = ()
    exact_document_ids: tuple[str, ...] = ()
    candidate_entities: tuple[str, ...] = ()
    ambiguous_entities: tuple[str, ...] = ()
    unresolved_references: tuple[str, ...] = ()
    points: tuple[str, ...] = ()
    date_ranges: tuple[tuple[date, date], ...] = ()
    event_role: str = ""
    comparison: bool = False


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
_DATE_WITHOUT_YEAR = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_PATTERN})\b"
    r"|(?<![\d./-])(?P<number>\d{1,2})[./](?P<numeric_month>\d{1,2})(?![\d./-])",
    re.I,
)
_SHARED_MONTH = re.compile(r"\b(\d{1,2})\s*(?:и\s+|,\s*|-\s*)$")
_MONTH_YEAR = re.compile(rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:в|за|на)\s+(20\d{2})\b|\b(20\d{2})\s*(?:год[ауе]?\b|г\.)", re.I)
_WORD = re.compile(r"[а-яa-z]+|\d+(?:[.,]\d+)?")
_ENTITY = re.compile(r"\b[а-яa-z]+(?:-[а-яa-z]+)+\b")
_ORG_QUOTE = re.compile(r"[«\"“]([^»\"”\n]{2,80})[»\"”]")
_ID = re.compile(r"(?<!\w)([а-яa-z]{1,4})\s*-\s*(\d+)(?:\s*[/\-]\s*(20\d{2}))?(?!\w)", re.I)
# NFKC turns the number sign (№) into "No", which is lowercased before matching.
_DOCUMENT_DATE = re.compile(
    r"\b(?:договор\w*|контракт\w*|регламент\w*|редакци\w*|(?:доп\w*\s*)?соглашени\w*)"
    rf"(?:\s+[а-яa-z]+){{0,3}}\s+(?:(?:no\s*)?{_ID.pattern}\s+|no\s*[а-яa-z\d/-]+\s+)?от\s*$"
)
_ARTICLE = re.compile(r"\b(?:стать(?:я|и|е|ю|ей|ями|ях|ям)|ст\.?)\s*(\d+(?:\.\d+)*)", re.I)
_POINT = re.compile(r"\b(?:пункт\w*|част[ьи]|п\.|ч\.)\s*(\d+(?:\.\d+)*)", re.I)
_NUMBER_TAIL = re.compile(r"\s*(?:,\s*|\b(?:и|или)\s+)(\d+(?:\.\d+)*)(?![\d./-])", re.I)
_ARTICLE_FAMILY = re.compile(
    r"\b(?P<tk>тк(?:\s+рф)?|трудов\w*\s+кодекс\w*)\b|"
    r"\b(?P<nk>нк(?:\s+рф)?|налогов\w*\s+кодекс\w*)\b|"
    r"\b(?P<koap>коап(?:\s+рф)?|кодекс\w*\s+(?:рф\s+)?об\s+административных\s+правонарушениях)\b|"
    r"\b(?P<gk>гк(?:\s+рф)?|гражданск\w*\s+кодекс\w*)\b|"
    r"\b(?P<uk>ук(?:\s+рф)?|уголовн\w*\s+кодекс\w*)\b|"
    r"\b(?P<fz>\d{1,4}\s*-?\s*фз)\b", re.I,
)
_FAMILY_NAMES = {"tk": "ТК РФ", "nk": "НК РФ", "koap": "КоАП РФ", "gk": "ГК РФ", "uk": "УК РФ"}
_LEGAL_ENTITY = re.compile(r"\b(?:ооо|ао|пао|зао|оао)\s*[«\"“]([^»\"”\n]{2,80})[»\"”]", re.I)
_LOCATOR_POINT = re.compile(r"\b(?:п|ч)\.\s*(\d+(?:\.\d+)*(?:-\d+)?)\b", re.I)
_LOCAL_POINT_LIST = re.compile(r"\bпункт\w*\s+([\d.,\s]+(?:и\s+\d+(?:\.\d+)*)?)\s+настоящей\s+статьи", re.I)
_AMENDMENT_DATE = re.compile(r"\bспецификаци\w*[^.!?\n]{0,90}?подписан\w*\s+после\s+", re.I)
_DATE_COMPARISON = re.compile(r"\b(?:сравн\w*|сопостав\w*|раньше\s+и\s+теперь|до\s+и\s+после)\b", re.I)
_DATE_CLAUSE = re.compile(r"[!?;\n]|\.(?=\s)")
_SIGNATURE = re.compile(r"\b(?:подпи[сш]\w*|заключ(?:ил\w*|а\w*|ен\w*|ить|ат)|заключени[яе]\s+договор\w*)\b")
_SPECIFICATION = re.compile(r"\bспецификаци\w*\b")
_REQUEST = re.compile(r"\b(?:запрос\w*|заявк\w*)\b")
_DELIVERY = re.compile(r"\b(?:достав\w*|отгруз\w*|поставк\w*|поставл\w*|постави(?:ли|ть|т|м)|приеха\w*)\b")
_FOLLOWING_ITEM = re.compile(r"^\s*[,;]?\s*(?:а\s+)?(?:втор\w*|трет\w*|друг\w*)\s*-\s*$")
_RATE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:%|процент\w*)", re.I)
_AMOUNT = re.compile(r"\b(\d{1,3}(?:\s\d{3}){1,5})(?=\s*(?:\(|руб))|\b(\d{1,15})\s*(млн|тыс)\b", re.I)
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
_RATE_QUERY = re.compile(r"\b(?:какая|какова|укажите)\s+ставк\w*\s+(?:ндс\b|налог\w*\s+на\s+прибыл\w*)", re.I)
_DEFAULT_RATE_PROVISION = re.compile(r"\b(?:если иное не установлено настоящей статьей|в случаях, не указанных в пунктах)\b", re.I)
_PAYMENT_DEADLINE = re.compile(r"\b(?:срок\w*\s+(?:оплат\w*|расчет\w*)|когда\s+(?:платить|оплатить))\b", re.I)
_LAW_NOTE = re.compile(
    r"^(?:ГАРАНТ\s*:|Информация об изменениях|См\.|Федеральн\w+ закон\w* от\b|"
    r"Об изменени\w* |О конституционно-правовом|О проверке конституционности|Обратите внимание|"
    r"[сc]рок давности привлечения к ответственности по ст\.)", re.I,
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
    ("оформить", "оформление", "оформляется"),
    ("скан", "сканированная копия", "скан-копия"),
    ("размещение", "размещаются", "разместить"),
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
    # NFKC normalizes the non-breaking hyphen (U+2011) to U+2010.
    return unicodedata.normalize("NFKC", text).lower().translate(str.maketrans("ё‐‒–—−", "е-----"))


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


def _date_mentions(text: str) -> list[tuple[int, int, date]]:
    values = []
    for match in _DATE.finditer(text):
        try:
            if match["day"]:
                value = date(int(match["year"]), int(match["month"]), int(match["day"]))
            else:
                value = date(int(match["text_year"]), _month_number(match["text_month"]), int(match["text_day"]))
            values.append((match.start(), match.end(), value))
        except ValueError:
            continue
    for match in _ISO_DATE.finditer(text):
        try:
            values.append((match.start(), match.end(), date(*(int(part) for part in match.groups()))))
        except ValueError:
            continue
    return sorted(values)


def _query_date_mentions(text: str, default_year: int | None) -> list[tuple[int, int, date]]:
    values = _date_mentions(text)
    years = {value.year for _, _, value in values}
    year = next(iter(years)) if len(years) == 1 else default_year if not years else None
    if year is not None:
        for match in _DATE_WITHOUT_YEAR.finditer(text):
            start, end = match.span()
            if any(start < other_end and end > other_start for other_start, other_end, _ in values):
                continue
            # Dotted numbers can also be clause locators. Require a temporal cue.
            if match["number"] and re.search(r"\b(?:пункт\w*|стать\w*|п\.|ч\.|ст\.)\s*$", text[:start]):
                continue
            if match["number"] and not re.search(r"(?:\bна|\bот|\bдо|\bпосле|\bи|\b-\s*|\s-)\s*$", text[:start]):
                continue
            day = int(match["day"] or match["number"])
            month = _month_number(match["month"]) if match["month"] else int(match["numeric_month"])
            try:
                values.append((start, end, date(year, month, day)))
            except ValueError:
                continue
    for start, _, value in tuple(values):
        if match := _SHARED_MONTH.search(text[:start]):
            day = int(match[1])
            if day > monthrange(value.year, value.month)[1] or not day:
                continue
            if not any(match.start() < end and match.end(1) > other_start for other_start, end, _ in values):
                values.append((match.start(), match.end(1), value.replace(day=day)))
    return sorted(values)


def _first_date(text: str) -> date | None:
    mentions = _date_mentions(text)
    return mentions[0][2] if mentions else None


def _date_after(pattern: re.Pattern[str], text: str) -> date | None:
    match = pattern.search(text)
    return _first_date(text[match.end():]) if match else None


def _event_role(left: str, right: str) -> str:
    if _DOCUMENT_DATE.search(left):
        return "document"
    for clause in (left, right):
        signature = _SIGNATURE.search(clause)
        request = _REQUEST.search(clause)
        delivery = _DELIVERY.search(clause)
        if _SPECIFICATION.search(clause) and (signature or re.search(r"\bот\s*$", clause)):
            return "specification"
        if signature:
            if re.search(r"\bдоп(?:олнительн\w*)?\s*соглашени\w*\b", clause):
                return "amendment_signature"
            return "signature"
        if request:
            return "request"
        if delivery:
            return "delivery"
    return ""


def _event_dates(query: str, default_year: int | None = None) -> tuple[tuple[tuple[date, date], ...], str, bool]:
    text = _normalize(query)
    mentions = [(start, end, (value, value)) for start, end, value in _query_date_mentions(text, default_year)]
    for pattern in (_MONTH_YEAR, _YEAR):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end, _ in mentions):
                continue
            if pattern is _MONTH_YEAR:
                month, year = _month_number(match[1]), int(match[2])
                period = date(year, month, 1), date(year, month, monthrange(year, month)[1])
            else:
                year = int(match[1] or match[2])
                period = date(year, 1, 1), date(year, 12, 31)
            mentions.append((start, end, period))
    dates = []
    mentions.sort()
    for index, (start, end, period) in enumerate(mentions):
        left = text[max(0, start - 20):start]
        right = text[end:end + 20]
        qualifier_match = re.search(r"\b(не позднее|не после|начиная с|не ранее|после|до)\s*$", left)
        qualifier = qualifier_match[1] if qualifier_match else ""
        if qualifier == "после" and period[1] < date.max:
            period = period[1] + timedelta(days=1), date.max
        elif qualifier in {"не позднее", "не после", "до"} and period[0] > date.min:
            inclusive = qualifier != "до" or bool(re.match(r"\s+включительно\b", right))
            period = date.min, period[1] if inclusive else period[0] - timedelta(days=1)
        elif qualifier in {"начиная с", "не ранее"}:
            period = period[0], date.max
        previous_end = mentions[index - 1][1] if index else 0
        next_start = mentions[index + 1][0] if index + 1 < len(mentions) else len(text)
        between = text[previous_end:start]
        before = _DATE_CLAUSE.split(between[-95:])[-1]
        after = _DATE_CLAUSE.split(text[end:min(next_start, end + 45)], maxsplit=1)[0].split(",", 1)[0]
        role = _event_role(before, after)
        if not role and dates and (re.fullmatch(r"\s*(?:и|или|,|-)\s*", between) or _FOLLOWING_ITEM.fullmatch(between)):
            role = dates[-1][1]
        dates.append((period, role))
    actual = [item for item in dates if item[1] != "document"]
    if actual:
        dates = actual
    compared = len(dates) > 1 and bool(_DATE_COMPARISON.search(text))
    if not compared and len(dates) > 1:
        desired_role = next((role for marker, role in (
            (r"на\s+дату\s+подписания\s+спецификаци\w*", "specification"),
            (r"на\s+дату\s+подписания\s+доп(?:олнительного)?\s*соглашени\w*", "amendment_signature"),
            (r"на\s+дату\s+(?:отгрузки|поставки|доставки)", "delivery"),
            (r"на\s+дату\s+(?:подписания|заключения)\s+договора", "signature"),
        ) if re.search(marker, text)), "")
        roles = {role for _, role in dates}
        if not desired_role and roles == {"request", "signature"}:
            desired_role = "signature"
        if not desired_role and roles == {"delivery", "specification"}:
            desired_role = "specification"
        selected = [item for item in dates if item[1] == desired_role] if desired_role else []
        if selected:
            dates = selected
    periods = tuple(dict.fromkeys(period for period, _ in dates))
    roles = {role for _, role in dates}
    return periods, next(iter(roles)) if len(roles) == 1 else "", compared or (len(periods) > 1 and len(roles) == 1)


def _number_groups(pattern: re.Pattern[str], text: str) -> Iterator[tuple[str, int, int]]:
    for match in pattern.finditer(text):
        end = match.end()
        yield match[1], match.start(), end
        while tail := _NUMBER_TAIL.match(text, end):
            end = tail.end()
            yield tail[1], tail.start(), end


def _family(match: re.Match[str]) -> str:
    return re.sub(r"[^\d]", "", match[0]) + "фз" if match.lastgroup == "fz" else str(match.lastgroup)


def _family_near(text: str, start: int, end: int) -> str:
    matches = []
    for match in _ARTICLE_FAMILY.finditer(text):
        left, right = min(end, match.end()), max(start, match.start())
        gap = text[left:right]
        if right - left > 48 or re.search(r"[;!?\n]", gap) or _ARTICLE.search(gap):
            continue
        if match.end() <= start and re.fullmatch(r"\s*[,\s]*(?:и|или|а)\s*", gap):
            continue
        direct = not gap.strip() or (match.start() >= end and re.fullmatch(r"\s*федеральн\w*\s+закон\w*\s+", gap))
        matches.append((0 if direct else max(0, right - left), match))
    return _family(min(matches, key=lambda item: item[0])[1]) if matches else ""


def _org_key(text: str) -> str:
    return " ".join(_WORD.findall(_normalize(text)))


def _org_stem(text: str) -> tuple[str, ...]:
    return tuple(_stem(term) for term in text.split())


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
    effective_event = ""
    if short_name.lower().startswith("договор") and (contract := _ID.search(lines[0])):
        title = _ID.sub(lambda _: "№ " + contract[0], title, count=1)
    elif short_name.lower().startswith("доп соглашение"):
        title = " ".join(lines[:2])
    if short_name.lower().startswith("письмо"):
        if operative := next((line for line in lines if re.match(r"^С\s+\d", line)), ""):
            effective_from = _first_date(operative) or effective_from
            if re.search(r"\bдо подписания (?:\w+ )?договора\b", operative, re.I):
                effective_event = "signature"
    return Document(doc_id, title, text, filename, "corporate", effective_from=effective_from,
                    effective_to=effective_to, edition_note=edition_note, reference=reference,
                    effective_event=effective_event)


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
    note_section = False
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
        if re.fullmatch(r"Примечани[ея]\s*[.:]?", line, re.I):
            parent = ""
            note_section = True
            continue
        if article_heading.match(line):
            line = article_heading.sub("", line, count=1).lstrip(". ")
            if not line or len(line) < 125 and not re.search(r"\b(?:установить|устанавливается)\b", line, re.I):
                continue
        scope = parent
        subpoint = ""
        if match := _CLAUSE.match(line):
            scope = parent = match[1] or match[2]
            locator = f"{article_label}, примечание {parent}" if note_section else f"{article_label}, {part_label} {parent}"
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
                if note_section and parent:
                    prefix = f"{article_label}, примечание {parent}"
                locator = f"{prefix}, абзац «{_lead(line)}»"
        notes = article_edition + ([] if note_section else scoped_edition.get(scope, [])) + unscoped_edition
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
        self._references = defaultdict(list)
        self._articles = defaultdict(list)
        self._article_families = {}
        self._organizations = defaultdict(list)
        self._org_names = {}
        for doc in documents.values():
            if doc.reference:
                self._references[doc.reference].append(doc.id)
                short = doc.reference.split("/", 1)[0]
                if short != doc.reference:
                    self._references[short].append(doc.id)
            if doc.kind == "law" and (number := _ARTICLE.search(doc.title)):
                normalized_title = _normalize(doc.title)
                families = {_family(match) for match in _ARTICLE_FAMILY.finditer(normalized_title)}
                family = next(iter(families)) if len(families) == 1 else _family_near(normalized_title, *number.span())
                if family:
                    self._article_families[doc.id] = family
                    self._articles[family, number[1]].append(doc.id)
                self._articles["", number[1]].append(doc.id)
            if doc.kind == "corporate":
                title = _org_key(doc.title)
                for name in _LEGAL_ENTITY.findall(doc.text[:1800]):
                    org = _org_key(name)
                    if re.search(rf"(?<!\w){re.escape(org)}(?!\w)", title):
                        self._organizations[org].append(doc.id)
                        self._org_names[org] = name
        self._by_position = {passage.id: position for items in self._by_document.values()
                             for position, passage in enumerate(items)}
        self._list_members = {}
        self._list_leads = {}
        for items in self._by_document.values():
            for position, lead in enumerate(items):
                if not lead.text.endswith(":"):
                    continue
                lead_section = _LOCATOR_POINT.search(lead.locator)
                members, size = [], 0
                for candidate in items[position + 1:position + 13]:
                    section = _LOCATOR_POINT.search(candidate.locator)
                    if (lead_section and (not section or section[1] != lead_section[1])) or (
                        lead.heading and candidate.heading != lead.heading
                    ) or size + len(candidate.text) > 4200:
                        break
                    members.append(candidate)
                    size += len(candidate.text)
                    if not candidate.text.endswith(";"):
                        break
                if len(members) >= 2 and members[0].text.endswith(";"):
                    self._list_members[lead.id] = members
                    for member in members:
                        self._list_leads.setdefault(member.id, lead)
        self._sections = defaultdict(list)
        self._amendments = defaultdict(list)
        self._amendment_dates = {}
        for passage in passages.values():
            point = _LOCATOR_POINT.search(passage.locator)
            if point:
                self._sections[passage.document_id, point[1]].append(passage)
            if point and passage.locator.startswith("изменение"):
                if boundary := _date_after(_AMENDMENT_DATE, passage.text):
                    self._amendment_dates[passage.id] = boundary
                    doc = self.documents[passage.document_id]
                    self._amendments[doc.reference, point[1]].append(boundary)

    def _organization_matches(self, query: str) -> tuple[set[str], set[str], set[str]]:
        query_words = _org_key(query).split()
        words = tuple(_stem(word) for word in query_words)
        exact, typo, ambiguous = set(), set(), set()
        for org in self._organizations:
            tokens = _org_stem(org)
            if any(words[start:start + len(tokens)] == tokens for start in range(len(words) - len(tokens) + 1)):
                exact.add(org)
        candidates = {_org_key(name) for name in (*_ORG_QUOTE.findall(query), *_ENTITY.findall(query))}
        if not exact:
            for org in self._organizations:
                first, *rest = _org_stem(org)
                for start, word in enumerate(words):
                    if word == first and rest:
                        phrase = " ".join(query_words[start:start + len(rest) + 1])
                        if _name_similarity(phrase, org) >= 86:
                            candidates.add(phrase)
            covered = {_org_stem(name)[0] for name in candidates if " " in name}
            candidates.update(word for word in query_words if len(word) >= 4 and _stem(word) not in covered
                              and sum(_org_stem(org)[0] == _stem(word) for org in self._organizations) > 1)
        for candidate in sorted(candidates):
            if len(candidate) < 4 or any(_org_stem(candidate) == _org_stem(org) for org in exact):
                continue
            prefix = {org for org in self._organizations if org.startswith(candidate)
                      or (" " not in candidate and _org_stem(org)[0] == _stem(candidate))}
            if len(prefix) > 1:
                if not prefix & exact:
                    ambiguous.update(prefix)
                continue
            ranked = sorted(((_name_similarity(candidate, org), org) for org in self._organizations), reverse=True)
            if not ranked or ranked[0][0] < 86:
                continue
            score, org = ranked[0]
            close = {name for similarity, name in ranked if score - similarity < 6 and similarity >= 82}
            if len(close) > 1:
                if not close & exact:
                    ambiguous.update(close)
            else:
                typo.add(org)
        return exact, typo - exact, ambiguous - exact

    def scope(self, query: str, *, default_year: int | None = None) -> SearchScope:
        normalized = _normalize(query)
        exact_docs, inferred_docs, ref_docs, unresolved = [], [], [], []
        for match in _ID.finditer(normalized):
            reference = _reference(match)
            if ids := self._references.get(reference):
                ref_docs.extend(ids)
            elif match[3] or re.search(r"(?:\bno\s*|договор\w*\s+)$", normalized[max(0, match.start() - 22):match.start()]):
                unresolved.append(reference)
        exact_docs.extend(ref_docs)
        for number, start, end in _number_groups(_ARTICLE, normalized):
            family = _family_near(normalized, start, end)
            if ids := self._articles.get((family, number)):
                (exact_docs if family or len(ids) == 1 else inferred_docs).extend(ids)
            else:
                display_family = _FAMILY_NAMES.get(family, f"{family[:-2]}-ФЗ" if family.endswith("фз") else "")
                unresolved.append(f"ст. {number}" + (f" ({display_family})" if display_family else ""))

        exact_org, typo_org, ambiguous_org = self._organization_matches(normalized)
        if ref_docs and ambiguous_org:
            matching = {name for name in ambiguous_org if set(self._organizations[name]) & set(ref_docs)}
            if len(matching) == 1:
                ambiguous_org.clear()
                exact_org.update(matching)
        exact_docs.extend(doc_id for org in sorted(exact_org) for doc_id in self._organizations[org])
        inferred_docs.extend(doc_id for org in sorted(typo_org | ambiguous_org) for doc_id in self._organizations[org])
        ranges, role, comparison = _event_dates(query, default_year)
        return SearchScope(
            document_ids=tuple(dict.fromkeys(exact_docs + inferred_docs)),
            exact_document_ids=tuple(dict.fromkeys(exact_docs)),
            candidate_entities=tuple(self._org_names[org] for org in sorted(exact_org | typo_org | ambiguous_org)),
            ambiguous_entities=tuple(self._org_names[org] for org in sorted(ambiguous_org)),
            unresolved_references=tuple(dict.fromkeys(unresolved)),
            points=tuple(dict.fromkeys(number for number, _, _ in _number_groups(_POINT, normalized))),
            date_ranges=ranges, event_role=role, comparison=comparison,
        )

    def applicability(self, passage: Passage, scope: SearchScope) -> str:
        """Assess whether the corpus has an explicit temporal boundary for this evidence."""
        if scope.comparison:
            return "comparison"
        if len(scope.date_ranges) != 1 or scope.event_role == "document":
            return "unknown"
        start, end = scope.date_ranges[0]
        document = self.documents[passage.document_id]
        if document.effective_event and scope.event_role not in {"", document.effective_event}:
            return "unknown"
        if scope.event_role == "specification":
            if boundary := self._amendment_dates.get(passage.id):
                return "inapplicable" if end <= boundary else "applicable" if start > boundary else "unknown"
            point = _LOCATOR_POINT.search(passage.locator)
            if point and (amendments := self._amendments.get((document.reference, point[1]))):
                return "inapplicable" if start > min(amendments) else "applicable" if end <= min(amendments) else "unknown"
        effective = passage.effective_from or document.effective_from
        expires = document.effective_to
        if (effective and end < effective) or (expires and start >= expires):
            return "inapplicable"
        if (effective and start < effective) or (expires and end >= expires):
            return "unknown"
        if document.snapshot_date and end > document.snapshot_date:
            return "unknown"
        return "applicable" if effective or expires else "unknown"

    def _section_start(self, doc_id: str, number: str) -> list[Passage]:
        section = self._sections.get((doc_id, number), [])
        start = next((item for item in section if "примечание о применении" not in item.locator), None)
        return [start] if start else []

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
        if boundary := self._amendment_dates.get(passage.id):
            metadata.append(f"Изменённое положение применяется к спецификациям, подписанным после {boundary:%d.%m.%Y}.")
        if document.edition_note:
            metadata.append(document.edition_note)
        if passage.edition_note:
            metadata.append(f"Примечания об изменениях из выгрузки (не текст нормы): {passage.edition_note}")
        if lead := self._list_leads.get(passage.id):
            metadata.append(f"Вводный абзац перечня {lead.locator}: {lead.text}")
        return "\n".join(filter(None, (*metadata, passage.text)))

    def neighbors(self, passage: Passage) -> list[Passage]:
        """Return the clause opener and close neighbors within the same section."""
        items = self._by_document[passage.document_id]
        position = self._by_position[passage.id]
        point = _LOCATOR_POINT.search(passage.locator)
        section = self._sections.get((passage.document_id, point[1]), []) if point else []
        lead = self._list_leads.get(passage.id)
        related = [lead] if lead else []
        if point:
            related.extend(item for item in self._section_start(passage.document_id, point[1])
                           if item != passage and item != lead)
        for candidate in items[max(0, position - 1):position + 2]:
            if candidate == passage or candidate in related:
                continue
            same_heading = passage.heading and passage.heading == candidate.heading and passage.heading != "Преамбула"
            if same_heading or candidate in section or (not point and passage.locator.startswith("ст.")):
                related.append(candidate)
        return related

    def list_context(self, passage: Passage) -> list[Passage]:
        """Expand a list opener into its consecutive, separately citable items."""
        return list(self._list_members.get(passage.id, ()))

    def related(self, passage: Passage) -> list[Passage]:
        document = self.documents[passage.document_id]
        result = []
        if document.kind == "law":
            if point := _LOCATOR_POINT.search(passage.locator):
                result.extend(candidate for candidate in self._sections[document.id, point[1]]
                              if candidate.locator.replace("примечание о применении ", "") == passage.locator
                              or passage.locator.replace("примечание о применении ", "") == candidate.locator)
            for match in _LOCAL_POINT_LIST.finditer(passage.text):
                for number in re.findall(r"\d+(?:\.\d+)*", match[1]):
                    result.extend(self._section_start(document.id, number))
            normalized = _normalize(passage.text)
            for number, start, end in _number_groups(_ARTICLE, normalized):
                family = _family_near(normalized, start, end) or self._article_families.get(document.id, "")
                point_ref = re.search(r"\b(?:пункт\w*|част\w*|п\.|ч\.)\s*(\d+(?:\.\d+)*)\s*$",
                                      normalized[max(0, start - 36):start])
                for doc_id in self._articles.get((family, number), []):
                    if doc_id != document.id:
                        result.extend(self._section_start(doc_id, point_ref[1]) if point_ref else self._by_document[doc_id][:2])
            return [item for item in dict.fromkeys(result) if item != passage][:8]
        clause = _LOCATOR_POINT.search(passage.locator)
        for number, _, _ in _number_groups(_POINT, _normalize(passage.text)):
            result.extend(self._section_start(document.id, number))
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
        return [item for item in dict.fromkeys(result) if item != passage][:8]

    def search(self, query: str, limit: int = 24, *, scope: SearchScope | None = None) -> list[Passage]:
        terms = _query_terms(query)
        if limit <= 0:
            return []
        scope = scope if scope is not None else self.scope(query)
        if not terms and not scope.document_ids:
            return []
        normalized = _normalize(query)
        default_rate = _DEFAULT_RATE_QUERY.search(normalized)
        generic_rate = _RATE_QUERY.search(normalized)
        payment_deadline = _PAYMENT_DEADLINE.search(normalized)
        entities = tuple(_ORG_QUOTE.findall(normalized))
        query_entities = {token for token in terms if token.startswith(("entity:", "ref:", "article:"))}
        exact_ids = set(scope.exact_document_ids)
        scope_ids = set(scope.document_ids)
        document_scores = {}
        for key, document in self.documents.items():
            title = self._titles[key]
            overlap = sum(min(4.0, self._idf.get(token, 3.0)) * weight
                          for token, weight in terms.items() if token in title)
            overlaps = query_entities & title.keys()
            if scope.unresolved_references and key not in scope_ids:
                overlaps = {token for token in overlaps if not token.startswith(("article:", "ref:"))}
            exact = len(overlaps) * 4.0
            document_title = _normalize(document.title)
            exact += 4.0 * sum(bool(re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", document_title))
                               for entity in entities)
            anchor_bonus = 7.0 if key in exact_ids else 4.5 if key in scope_ids else 0.0
            document_scores[key] = min(12.0, overlap * 0.67) + exact + anchor_bonus

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
            if (default_rate or generic_rate) and _DEFAULT_RATE_PROVISION.search(_normalize(passage.text)):
                score += 9.0 if default_rate else 6.0
            if payment_deadline and self.documents[passage.document_id].kind == "corporate":
                text = _normalize(passage.text)
                if re.search(r"\b(?:оплачива\w*|оплат\w*|внос\w*)\b", text) and re.search(
                    r"\b(?:в течение|не позднее|ежемесячно)\b", text
                ):
                    score += 5.0
            document = self.documents[passage.document_id]
            title_score = document_scores[passage.document_id]
            if scope.points and (not scope_ids or passage.document_id in scope_ids):
                point = _LOCATOR_POINT.search(passage.locator)
                if point and point[1] in scope.points:
                    score += 18.0 if passage in self._section_start(passage.document_id, point[1]) else 10.0
            factor = 0.65 if document.effective_to and not scope.date_ranges and not scope.comparison else 1.0
            if document.kind == "corporate" and scope.date_ranges:
                temporal = self.applicability(passage, scope)
                factor = {"inapplicable": 0.58, "applicable": 1.25}.get(temporal, 1.0)
            if score > 0.0 or title_score >= 4.0:
                scores.append(((score + title_score) * factor, key))
        scores.sort(key=lambda item: -item[0])
        ranked = [self.passages[key] for _, key in scores]
        if not scope_ids:
            return ranked[:limit]
        anchored = [item for item in ranked if item.document_id in scope_ids]
        outside = [item for item in ranked if item.document_id not in scope_ids]
        # Start with the strongest exact hit and keep candidates from each explicit source.
        best = anchored[:1]
        if len(scope_ids) > 1 and limit >= 4:
            by_document = {}
            for item in anchored:
                by_document.setdefault(item.document_id, item)
            best.extend(list(by_document.values())[:max(2, limit // 4)])
        reserve = min(len(outside), limit // 4)
        preferred = list(dict.fromkeys(best + anchored))[:limit - reserve]
        return (preferred + outside + [item for item in anchored if item not in preferred])[:limit]


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
