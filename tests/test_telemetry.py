import unittest
from unittest.mock import Mock, call, patch

import telemetry


class TelemetryTests(unittest.TestCase):
    def test_exports_only_allowed_metadata(self):
        active = Mock()
        trace = telemetry.TraceSpan(active)
        trace.set("psb.answer.status", "partial")
        trace.set("psb.context.candidates", ["P123", "P4"])
        trace.set("psb.question.length", 45)
        trace.set("psb.gateway.repair", "invalid_json")
        trace.set("psb.question", "Personal details from a query")
        trace.set("psb.context.candidates", ["P123", "Personal details from a document"])
        trace.set("input.value", "Personal details in a different attribute")
        self.assertEqual(active.set_attribute.call_args_list, [
            call("psb.answer.status", "partial"),
            call("psb.context.candidates", ("P123", "P4")),
            call("psb.question.length", 45),
            call("psb.gateway.repair", "invalid_json"),
        ])

    def test_noop_preserves_application_exception(self):
        with patch.object(telemetry, "_TRACER", None):
            with self.assertRaisesRegex(ValueError, "application failure"):
                with telemetry.span("assistant.answer") as trace:
                    trace.set("psb.answer.status", "error")
                    raise ValueError("application failure")
