import json
import time
import unittest
from http.client import IncompleteRead
from unittest.mock import MagicMock, patch

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
