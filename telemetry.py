"""Optional metadata-only OpenInference traces for a local OTLP collector."""

from __future__ import annotations

import logging
import math
import os
import re
from contextlib import AbstractContextManager, contextmanager, nullcontext
from urllib.parse import urlsplit


LOG = logging.getLogger(__name__)
_DEFAULT_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
_DEFAULT_PROJECT = "alfa-psb"
_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,79}")
_SOURCE_ID = re.compile(r"[DP][0-9]{1,8}")
_KINDS = {"CHAIN", "EMBEDDING", "GUARDRAIL", "LLM", "RERANKER", "RETRIEVER"}
_STATUSES = {
    "answered", "partial", "insufficient", "clarification", "out_of_scope", "refusal",
    "error", "offline", "ok", "skipped", "unavailable", "fallback", "disabled", "empty",
    "ready", "timeout", "invalid", "success", "failed", "not_configured",
    "deadline", "content_filter", "truncated", "interrupted", "network_error", "invalid_json", "invalid_response",
}


class TraceSpan:
    def __init__(self, active=None):
        self._active = active

    def set(self, key: str, value: object) -> None:
        if self._active is None or not key.startswith("psb.") or not _NAME.fullmatch(key):
            return
        if type(value) is bool or (type(value) is int and -(2**63) <= value < 2**63):
            self._active.set_attribute(key, value)
        elif type(value) is float and math.isfinite(value):
            self._active.set_attribute(key, value)
        elif isinstance(value, str) and (value in _STATUSES or _SOURCE_ID.fullmatch(value)):
            self._active.set_attribute(key, value)
            if value in {"error", "failed", "timeout"}:
                self._active.set_status(_ERROR_STATUS)
        elif isinstance(value, (list, tuple)) and value:
            identifiers = tuple(value[:64])
            if all(isinstance(item, str) and _SOURCE_ID.fullmatch(item) for item in identifiers):
                self._active.set_attribute(key, identifiers)


def _configure():
    if os.getenv("PSB_TRACE", "").strip().casefold() not in {"1", "true", "yes", "on"}:
        return None, None

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import Status, StatusCode
        endpoint = os.getenv("PSB_TRACE_ENDPOINT", "").strip() or _DEFAULT_ENDPOINT
        target = urlsplit(endpoint)
        if (target.scheme not in {"http", "https"} or not target.hostname or target.username
                or target.password or target.query or target.fragment):
            raise ValueError("Invalid collector endpoint")
        project = os.getenv("PSB_TRACE_PROJECT", "").strip() or _DEFAULT_PROJECT
        if not _NAME.fullmatch(project):
            raise ValueError("Invalid collector project name")
        exporter = OTLPSpanExporter(endpoint=endpoint, headers={"x-project-name": project}, timeout=1.5)
        provider = TracerProvider(resource=Resource({
            "service.name": _DEFAULT_PROJECT,
            "openinference.project.name": project,
        }))
        provider.add_span_processor(BatchSpanProcessor(
            exporter, max_queue_size=256, max_export_batch_size=64, schedule_delay_millis=300,
        ))
        return provider.get_tracer("alfa-psb"), Status(StatusCode.ERROR)
    except ImportError:
        LOG.warning("Phoenix tracing disabled: install requirements-dev.txt to enable export")
        return None, None
    except Exception as exc:
        LOG.warning("Phoenix tracing disabled: %s", type(exc).__name__)
        return None, None


_TRACER, _ERROR_STATUS = _configure()
_NOOP = nullcontext(TraceSpan())


@contextmanager
def _record(name: str, kind: str):
    with _TRACER.start_as_current_span(
        name, attributes={"openinference.span.kind": kind},
        record_exception=False, set_status_on_exception=False,
    ) as active:
        try:
            yield TraceSpan(active)
        except Exception:
            active.set_status(_ERROR_STATUS)
            raise


def span(name: str, *, kind: str = "CHAIN") -> AbstractContextManager[TraceSpan]:
    """Start a span using a fixed code-level name and metadata-only attributes."""
    if _TRACER is None or not _NAME.fullmatch(name) or kind not in _KINDS:
        return _NOOP
    return _record(name, kind)
