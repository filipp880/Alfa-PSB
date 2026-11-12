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

const answerStatuses = {
  partial: "Частичный ответ",
  insufficient: "Недостаточно данных",
  clarification: "Нужно уточнение",
  out_of_scope: "Вне базы знаний",
  refusal: "Ответ ограничен",
  error: "Ошибка сервиса",
  offline: "Локальный поиск",
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

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Не удалось выполнить запрос.");
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

function addSources(body, citations) {
  const anchors = new Map();
  if (!citations.length) return anchors;
  const section = node("section", "evidence");
  section.append(node("h4", "", "Источники ответа"));
  const list = node("ul", "evidence-list");
  for (const citation of citations) {
    const item = node("li");
    const details = node("details", "citation");
    const label = [citation.title, citation.locator].filter(Boolean).join(" · ");
    const summary = node("summary", "", label);
    details.append(summary, node("blockquote", "", citation.quote));
    const openSource = node("button", "open-source", "Открыть в документе");
    openSource.type = "button";
    openSource.addEventListener("click", () => showDocument(citation.document_id, citation));
    details.append(openSource);
    item.append(details);
    list.append(item);
    if (!anchors.has(citation.id)) anchors.set(citation.id, []);
    anchors.get(citation.id).push({details, summary});
  }
  section.append(list);
  body.append(section);
  return anchors;
}

function renderAnswer(body, answer) {
  body.replaceChildren();
  if (Object.hasOwn(answerStatuses, answer.status)) {
    body.append(node("p", "message-status", answerStatuses[answer.status]));
  }
  const text = node("p", "message-text");
  body.append(text);
  const anchors = addSources(body, answer.citations || []);
  for (const part of answer.text.split(/(\[P\d+\])/g)) {
    const sources = anchors.get(part.slice(1, -1));
    if (!sources) {
      text.append(document.createTextNode(part));
      continue;
    }
    const button = node("button", "citation-ref", part);
    button.type = "button";
    button.setAttribute("aria-label", `Показать источник ${part.slice(1, -1)}`);
    button.addEventListener("click", () => {
      for (const source of sources) source.details.open = true;
      sources[0].summary.focus();
    });
    text.append(button);
  }
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
      showFailure(user, assistant, question, error instanceof TypeError
        ? "Не удалось связаться с сервером. Проверьте, что прототип запущен."
        : error.message);
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

async function showDocument(id, citation) {
  documentRequest?.abort();
  const request = new AbortController();
  documentRequest = request;
  if (!library.open) library.showModal();
  libraryIndex.hidden = true;
  documentView.hidden = false;
  documentView.scrollTop = 0;
  documentView.setAttribute("aria-busy", "true");
  documentTitle.textContent = citation?.title || "Загружаю документ…";
  documentFilename.textContent = "";
  documentText.textContent = "";
  documentLocation.textContent = citation?.locator || "";
  documentLocation.hidden = !citation?.locator;

  try {
    const item = await api(`/api/documents/${encodeURIComponent(id)}`, {signal: request.signal});
    if (request.signal.aborted) return;
    documentTitle.textContent = item.title;
    documentFilename.textContent = item.filename;
    documentTitle.focus({preventScroll: true});
    const position = citation?.quote ? item.text.indexOf(citation.quote) : -1;
    if (position >= 0) {
      const mark = node("mark", "", citation.quote);
      documentText.replaceChildren(item.text.slice(0, position), mark, item.text.slice(position + citation.quote.length));
      mark.scrollIntoView({block: "center"});
    } else {
      documentText.textContent = item.text;
    }
  } catch (error) {
    if (!request.signal.aborted) {
      documentTitle.textContent = "Документ недоступен";
      documentText.textContent = error instanceof TypeError ? "Не удалось связаться с сервером." : error.message;
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
