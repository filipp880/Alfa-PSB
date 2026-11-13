"use strict";

const element = (id) => document.getElementById(id);
const composer = element("composer");
const questionInput = element("question");
const sendButton = element("send");
const historyScroll = element("history-scroll");
const messages = element("messages");
const welcome = element("welcome");
const library = element("library");
const libraryIndex = element("library-index");
const libraryNotice = element("library-notice");
const documentList = element("document-list");
const documentSearch = element("document-search");
const documentView = element("document-view");
const documentTitle = element("document-title");
const documentText = element("document-text");
const documentLocation = element("document-location");
const documentFilename = element("document-filename");
const documentEdition = element("document-edition");

const answerStatuses = {
  answered: "Подтверждено",
  partial: "Частично подтверждено",
  insufficient: "Недостаточно данных",
  clarification: "Нужно уточнение",
  out_of_scope: "Вне базы знаний",
  refusal: "Отказ по основаниям",
  error: "Ошибка сервиса",
  offline: "Локальный поиск",
};

const editionStatuses = {
  applicable: "Применимая редакция",
  inapplicable: "Редакция неприменима",
  unknown: "Применимость не подтверждена",
  comparison: "Редакция для сравнения",
};

const claimVerdicts = {
  supported: "Подтверждено",
  rejected: "Отклонено",
  unclear: "Недостаточно подтверждений",
};

let turns = [];
let activeRequest = null;
let failedTurn = null;
let documents = null;
let documentsRequest = null;
let documentRequest = null;

function node(tag, className, content) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (content !== undefined) result.textContent = content;
  return result;
}

class RequestError extends Error {}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) {
    const knownError = response.headers.get("Content-Type")?.startsWith("application/json")
      && typeof result?.error === "string";
    throw new RequestError(knownError ? result.error : "Не удалось выполнить запрос. Повторите попытку.");
  }
  return result;
}

function updateComposer() {
  sendButton.disabled = Boolean(activeRequest) || !questionInput.value.trim();
}

function scrollToLatest() {
  historyScroll.scrollTop = historyScroll.scrollHeight;
}

function message(role, text) {
  const article = node("article", `message message-${role}`);
  article.append(node("h3", "message-heading", role === "user" ? "Вы" : "Ассистент"));
  const body = node("div", "message-body");
  body.append(node("p", "message-text", text));
  article.append(body);
  messages.append(article);
  return {article, body};
}

function questionPayload(question) {
  const history = [{role: "user", content: question}];
  let size = question.length;
  for (let index = turns.length - 2; index >= 0 && history.length < 15; index -= 2) {
    const pair = turns.slice(index, index + 2);
    const pairSize = pair[0].content.length + pair[1].content.length;
    if (size + pairSize > 24_000) break;
    history.unshift(...pair);
    size += pairSize;
  }
  return history;
}

function displayDate(iso) {
  const parts = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  return parts ? `${parts[3]}.${parts[2]}.${parts[1]}` : "";
}

function dateBadge(label, iso) {
  const date = displayDate(iso);
  if (!date) return null;
  const badge = node("span", "citation-badge", `${label} `);
  const time = node("time", "", date);
  time.dateTime = iso;
  badge.append(time);
  return badge;
}

function addEditionDates(details, citation, eventRanges) {
  const badges = node("div", "citation-badges");
  const label = citation.effective_to && eventRanges?.every(([start]) => start >= citation.effective_to)
    ? "Историческая редакция" : editionStatuses[citation.applicability];
  if (label && (citation.applicability !== "unknown" || eventRanges?.length || citation.effective_from || citation.effective_to)) {
    const badge = node("span", "citation-badge", label);
    badge.dataset.applicability = citation.applicability;
    badges.append(badge);
  }
  const dates = [
    dateBadge("В силе с", citation.effective_from),
    dateBadge("Утратила силу с", citation.effective_to),
    dateBadge("Выгрузка от", citation.snapshot_date),
  ].filter(Boolean);
  badges.append(...dates);
  if (badges.childElementCount) details.append(badges);
  if (citation.edition) details.append(node("p", "citation-edition", citation.edition));
}

function addSources(body, citations, status, eventRanges) {
  const anchors = new Map();
  if (!citations.length) return anchors;
  const section = node("section", "evidence");
  section.append(node("h4", "", status === "offline" ? "Найденные фрагменты" : "Источники ответа"));
  const list = node("ul", "evidence-list");
  for (const citation of citations) {
    const item = node("li");
    const details = node("details", "citation");
    const label = [citation.title, citation.locator].filter(Boolean).join(" · ");
    const summary = node("summary", "", label);
    details.append(summary);
    addEditionDates(details, citation, eventRanges);
    const openSource = node("button", "open-source");
    openSource.type = "button";
    openSource.append(node("span", "source-quote", citation.quote),
      node("span", "source-open-label", "Открыть в документе"));
    openSource.setAttribute("aria-label", `Открыть в документе: ${label}`);
    openSource.addEventListener("click", () => showDocument(citation.document_id, citation, eventRanges));
    details.append(openSource);
    item.append(details);
    list.append(item);
    const entry = {details, summary, label};
    anchors.set(citation.ref, entry);
  }
  section.append(list);
  body.append(section);
  return anchors;
}

function citationLink(source) {
  const button = node("button", "citation-ref", source.label);
  button.type = "button";
  button.addEventListener("click", () => {
    source.details.open = true;
    source.summary.focus({preventScroll: true});
    source.details.scrollIntoView({block: "nearest"});
  });
  return button;
}

function addDiagnostics(body, diagnostics) {
  if (!diagnostics) return;
  const panel = node("details", "diagnostics");
  const duration = new Intl.NumberFormat("ru", {maximumFractionDigits: 2})
    .format(diagnostics.elapsed_ms / 1000);
  panel.append(node("summary", "", "Как проверялся ответ"));
  panel.append(node("p", "diagnostic-summary",
    `${duration} с · ${diagnostics.retry_used ? "Был дополнительный поиск" : "Один проход поиска"}`));
  if (diagnostics.documents.length) {
    panel.append(node("h4", "", "Рассмотренные документы"));
    const list = node("ul", "diagnostic-list");
    for (const source of diagnostics.documents) {
      const edition = editionStatuses[source.applicability];
      list.append(node("li", "", edition ? `${source.title} · ${edition}` : source.title));
    }
    panel.append(list);
  }
  if (diagnostics.claims.length) {
    panel.append(node("h4", "", "Проверка утверждений"));
    const list = node("ul", "diagnostic-list");
    for (const claim of diagnostics.claims) {
      const item = node("li", "", claim.text);
      const reason = claim.reason ? ` · ${claim.reason}` : "";
      item.append(node("small", "diagnostic-verdict", `${claimVerdicts[claim.verdict]}${reason}`));
      list.append(item);
    }
    panel.append(list);
  }
  body.append(panel);
}

function renderAnswer(body, answer) {
  body.replaceChildren();
  if (Object.hasOwn(answerStatuses, answer.status)) {
    const status = node("p", "message-status", answerStatuses[answer.status]);
    status.dataset.status = answer.status;
    body.append(status);
  }
  const summary = node("div", "answer-summary");
  body.append(summary);
  const eventRanges = answer.event_ranges || [];
  const dates = eventRanges.map(([start, end]) => {
    const first = displayDate(start);
    const last = displayDate(end);
    return first && last && first !== last ? `${first} – ${last}` : first || last;
  }).filter(Boolean);
  if (dates.length) body.append(node("p", "event-dates", `Запрошенные даты · ${dates.join(" · ")}`));
  const anchors = addSources(body, answer.citations || [], answer.status, eventRanges);

  if (answer.claims?.length) {
    if (answer.intro) summary.append(node("p", "message-text", answer.intro));
    const claims = node("ul", "answer-claims");
    for (const claim of answer.claims) {
      const row = node("li", "answer-claim");
      row.append(node("p", "message-text", claim.text));
      const refs = node("div", "claim-sources");
      for (const id of claim.sources) {
        if (anchors.has(id)) refs.append(citationLink(anchors.get(id)));
      }
      if (refs.childElementCount) row.append(refs);
      claims.append(row);
    }
    summary.append(claims);
    if (answer.gap) {
      const gap = node("section", "answer-notice");
      gap.append(node("h4", "", "Не подтверждено базой"), node("p", "", answer.gap));
      summary.append(gap);
    }
  } else {
    summary.append(node("p", "message-text", answer.text));
  }
  if (answer.conflict) {
    const conflict = node("section", "answer-notice");
    conflict.append(node("h4", "", "Конфликт документов"), node("p", "", answer.conflict));
    summary.append(conflict);
  }
  addDiagnostics(body, answer.diagnostics);
}

function showFailure(user, assistant, question, text) {
  failedTurn = {user, assistant};
  renderAnswer(assistant.body, {status: "error", text, citations: []});
  const retry = node("button", "retry", "Повторить запрос");
  retry.type = "button";
  retry.addEventListener("click", () => submitQuestion(question));
  assistant.body.append(retry);
}

async function submitQuestion(question) {
  if (!question || activeRequest) return;
  if (failedTurn) {
    failedTurn.user.article.remove();
    failedTurn.assistant.article.remove();
    failedTurn = null;
  }
  const request = new AbortController();
  activeRequest = request;
  const user = message("user", question);
  const assistant = message("assistant", "Ищу подходящие положения в базе знаний…");
  assistant.article.classList.add("message-pending");
  welcome.hidden = true;
  updateComposer();
  scrollToLatest();

  try {
    const answer = await api("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: questionPayload(question)}),
      signal: request.signal,
    });
    if (request.signal.aborted) return;
    if (answer.status === "error") {
      showFailure(user, assistant, question, answer.text);
      return;
    }
    renderAnswer(assistant.body, answer);
    turns.push({role: "user", content: question}, {role: "assistant", content: answer.text});
    turns = turns.slice(-14);
  } catch (error) {
    if (!request.signal.aborted) {
      const text = error instanceof RequestError ? error.message : "Не удалось получить ответ. Повторите запрос.";
      showFailure(user, assistant, question, text);
    }
  } finally {
    if (activeRequest === request) {
      activeRequest = null;
      assistant.article.classList.remove("message-pending");
      updateComposer();
      if (assistant.article.getBoundingClientRect().height > historyScroll.clientHeight / 2) {
        assistant.article.scrollIntoView({block: "start"});
      } else {
        scrollToLatest();
      }
    }
  }
}

function resetDialog() {
  activeRequest?.abort();
  activeRequest = null;
  failedTurn = null;
  turns = [];
  messages.replaceChildren();
  welcome.hidden = false;
  questionInput.value = "";
  historyScroll.scrollTop = 0;
  updateComposer();
  questionInput.focus();
}

function renderDocuments() {
  if (documents === null) return;
  const normalize = (text) => text.normalize("NFKC").toLocaleLowerCase("ru")
    .replaceAll("ё", "е").replace(/[-‐‑‒–—−]/g, "");
  const terms = normalize(documentSearch.value).trim().split(/\s+/).filter(Boolean);
  const matches = documents.filter((item) => {
    const label = normalize(`${item.title} ${item.filename}`);
    return terms.every((term) => label.includes(term));
  });
  const entries = matches.map((item) => {
    const entry = node("li");
    const button = node("button", "", item.title);
    button.type = "button";
    button.addEventListener("click", () => showDocument(item.id));
    entry.append(button);
    return entry;
  });
  documentList.replaceChildren(...entries);
  libraryNotice.hidden = Boolean(matches.length);
  libraryNotice.textContent = matches.length ? "" : "По этому запросу ничего не найдено.";
}

function loadLibrary() {
  if (documents !== null || documentsRequest) return;
  libraryNotice.textContent = "Загружаю список документов…";
  libraryNotice.hidden = false;
  documentsRequest = api("/api/documents").then((result) => {
    documents = result.documents;
    element("library-count").textContent = `Документов в базе: ${documents.length}`;
    const mode = element("runtime-mode");
    mode.textContent = result.offline ? "Локальный поиск" : "API не настроен";
    mode.hidden = !result.offline && result.model_configured;
    renderDocuments();
  }).catch(() => {
    libraryNotice.textContent = "Не удалось загрузить базу. Откройте её снова, чтобы повторить попытку.";
  }).finally(() => {
    documentsRequest = null;
  });
}

function showLibraryIndex() {
  documentRequest?.abort();
  documentRequest = null;
  if (!library.open) library.showModal();
  libraryIndex.hidden = false;
  documentView.hidden = true;
  loadLibrary();
  documentSearch.focus({preventScroll: true});
}

async function showDocument(id, citation, eventRanges) {
  documentRequest?.abort();
  const request = new AbortController();
  documentRequest = request;
  if (!library.open) library.showModal();
  libraryIndex.hidden = true;
  documentView.hidden = false;
  documentView.setAttribute("aria-busy", "true");
  documentTitle.textContent = citation?.title || "Загружаю документ…";
  documentFilename.textContent = "";
  documentText.textContent = "";
  documentText.scrollTop = 0;
  documentEdition.replaceChildren();
  documentLocation.textContent = citation?.locator || "";
  documentLocation.hidden = !citation?.locator;
  if (citation) addEditionDates(documentEdition, citation, eventRanges);

  try {
    const item = await api(`/api/documents/${encodeURIComponent(id)}`, {signal: request.signal});
    if (request.signal.aborted) return;
    documentTitle.textContent = item.title;
    documentFilename.textContent = item.filename;
    documentTitle.focus({preventScroll: true});
    let cursor = 0;
    let firstMark = null;
    for (const [start, end] of citation?.highlights || []) {
      if (!(cursor <= start && start < end && end <= item.text.length)) continue;
      const mark = node("mark", "", item.text.slice(start, end));
      documentText.append(item.text.slice(cursor, start), mark);
      firstMark ||= mark;
      cursor = end;
    }
    documentText.append(item.text.slice(cursor));
    firstMark?.scrollIntoView({block: "center"});
  } catch (error) {
    if (!request.signal.aborted) {
      documentTitle.textContent = "Документ недоступен";
      documentText.textContent = error instanceof RequestError ? error.message : "Не удалось загрузить документ.";
      documentTitle.focus({preventScroll: true});
    }
  } finally {
    if (documentRequest === request) {
      documentView.removeAttribute("aria-busy");
      documentRequest = null;
    }
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || activeRequest) return;
  questionInput.value = "";
  submitQuestion(question);
  questionInput.focus();
});

questionInput.addEventListener("input", updateComposer);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!sendButton.disabled) composer.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    updateComposer();
    questionInput.focus();
  });
});

element("new-chat").addEventListener("click", resetDialog);
element("open-library").addEventListener("click", showLibraryIndex);
element("back-to-library").addEventListener("click", showLibraryIndex);
element("close-library").addEventListener("click", () => library.close());
documentSearch.addEventListener("input", renderDocuments);
library.addEventListener("close", () => {
  documentRequest?.abort();
  documentRequest = null;
});
library.addEventListener("click", (event) => {
  if (event.target === library) library.close();
});

loadLibrary();
