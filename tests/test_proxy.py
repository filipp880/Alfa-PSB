import json
import time
import unittest
from http.client import IncompleteRead
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from proxy import Gateway, GatewayError, Settings


def completion(reason, content):
    return {"choices": [{"finish_reason": reason, "message": {"content": content}}]}


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.gateway = Gateway(Settings(base_url="http://fixture.local", key="fixture", model="fixture"))

    def test_repairs_even_parseable_incomplete_completions(self):
        for reason in ("length", "insufficient_system_resource", "aborted", "stop"):
            with self.subTest(reason=reason):
                initial = "   " if reason == "stop" else '{"claims":[]}'
                with patch.object(self.gateway, "_post", side_effect=[
                    completion(reason, initial), completion("stop", '{"claims":["complete"]}'),
                ]) as post:
                    result = self.gateway.chat("Верни JSON", {}, time.monotonic() + 20)
                self.assertEqual(result, {"claims": ["complete"]})
                self.assertEqual(post.call_count, 2)
                repair = post.call_args.args[2]
                self.assertEqual(repair["thinking"], {"type": "disabled"})
                self.assertEqual(repair["max_tokens"], 3000 if reason == "length" else 1500)

    def test_second_incomplete_completion_is_a_technical_error(self):
        with patch.object(self.gateway, "_post", return_value=completion("length", '{}')) as post:
            with self.assertRaisesRegex(GatewayError, "truncated"):
                self.gateway.chat("Верни JSON", {}, time.monotonic() + 20)
        self.assertEqual(post.call_count, 2)

    def test_retries_incomplete_http_body(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                gateway = Gateway(self.gateway.settings)
                opener = MagicMock()
                short, complete = MagicMock(), MagicMock()
                short.__enter__.return_value = short
                complete.__enter__.return_value = complete
                opener.open.side_effect = [short, complete]
                fragment = b'{"choices":'
                if chunked:
                    short.read.side_effect = IncompleteRead(fragment, 15)
                else:
                    short.read.return_value = fragment
                    short.length = 15
                complete.length = 0
                complete.read.return_value = json.dumps(completion("stop", '{"ok":true}')).encode()
                with patch("proxy.build_opener", return_value=opener), patch("proxy.time.sleep"):
                    result = gateway.chat("Верни JSON", {}, time.monotonic() + 20)
                self.assertEqual(result, {"ok": True})
                self.assertEqual(gateway.request_counts(), {"/chat/completions": 2})

    def test_extracts_json_with_surrounding_text(self):
        data = completion("stop", 'Вот результат: {"claims":[{"text":"факт"}]} Готово.')
        with patch.object(self.gateway, "_post", return_value=data) as post:
            self.assertEqual(self.gateway.chat("Верни JSON", {}, time.monotonic() + 20),
                             {"claims": [{"text": "факт"}]})
        self.assertEqual(post.call_count, 1)

    def test_repair_carries_error_and_forces_zero_temperature(self):
        data = completion("stop", '{"claims": [')
        with patch.object(self.gateway, "_post", side_effect=[data, completion("stop", '{"claims":[]}')]) as post:
            self.gateway.chat("Верни JSON", {"question": "Запрос"}, time.monotonic() + 20)
        repair = post.call_args.args[2]
        self.assertEqual(repair["temperature"], 0)
        self.assertIn("Return only valid JSON", repair["messages"][0]["content"])
        self.assertEqual(json.loads(repair["messages"][1]["content"]),
                         {"question": "Запрос", "json_error": "invalid_json", "malformed_response": '{"claims": ['})

    def test_bad_token_does_not_retry_even_with_503(self):
        opener = MagicMock()
        opener.open.side_effect = HTTPError("http://fixture.local", 503, "Unavailable", {},
                                            BytesIO(b'{"error":"token_not_found_in_db"}'))
        with patch("proxy.build_opener", return_value=opener):
            for reason in ("token_not_found_in_db", "disabled"):
                with self.assertRaises(GatewayError) as failure:
                    self.gateway.chat("Верни JSON", {}, time.monotonic() + 20)
                self.assertEqual(failure.exception.reason, reason)
        self.assertEqual(opener.open.call_count, 1)

    def test_rate_limit_reduces_workers_and_repeated_errors_disable_reranking(self):
        opener = MagicMock()
        opener.open.side_effect = [
            HTTPError("http://fixture.local", 429, "Limited", {}, BytesIO()) for _ in range(6)
        ]
        with patch("proxy.build_opener", return_value=opener), patch("proxy.time.sleep"):
            for _ in range(3):
                with self.assertRaises(GatewayError):
                    self.gateway.rerank("запрос", ["текст"], time.monotonic() + 20)
            with self.assertRaisesRegex(GatewayError, "disabled"):
                self.gateway.rerank("запрос", ["текст"], time.monotonic() + 20)
        self.assertEqual(opener.open.call_count, 6)
        self.assertEqual(self.gateway.worker_limit(), 1)
        self.assertEqual(self.gateway.metrics()["transport_retries"], 3)
