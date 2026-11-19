"""Tests for streaming zero-output fast fail, 300s timeout cap, and in-stage recovery."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ADV_DIR = Path(__file__).resolve().parents[1] / "mcp_servers/scene_editing/core/adversarial"
if str(ADV_DIR) not in sys.path:
    sys.path.insert(0, str(ADV_DIR))

from streaming_requests import (NoOutputTimeout, StreamingRequestArchive, deadline_events,
                               StreamingRequestFailure, RequestDeadlineExceeded, request_timeout_details)
from defense_request_recovery import retryable_request


class StreamingRetryAndTimeoutTests(unittest.TestCase):
    def test_deadline_events_fails_fast_when_zero_output_received(self):
        class HangingStream:
            def __enter__(self):
                time.sleep(0.5)
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(())

            def close(self):
                return None

        started = time.monotonic()
        with self.assertRaises(NoOutputTimeout):
            list(deadline_events(HangingStream, seconds=5.0, no_output_seconds=0.08))
        self.assertLess(time.monotonic() - started, 0.4)

    def test_streaming_archive_retries_zero_output_then_succeeds(self):
        calls = []

        class HangingThenOkClient:
            def __init__(self):
                self.chat = NS(completions=NS(create=self.create))

            def create(self, **kwargs):
                calls.append(kwargs["timeout"])
                if len(calls) == 1:
                    class Hanging:
                        def __enter__(self):
                            time.sleep(0.5)
                            return self

                        def __exit__(self, *args):
                            return False

                        def __iter__(self):
                            return iter(())

                        def close(self):
                            return None

                    return Hanging()

                chunk = NS(
                    model="test-model",
                    id="resp-1",
                    usage=NS(model_dump=lambda: {"prompt_tokens": 5}),
                    choices=[NS(index=0, delta=NS(reasoning_content=None, content='{"ok":true}', tool_calls=None), finish_reason="stop")],
                )

                class OkStream:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def __iter__(self):
                        return iter([chunk])

                    def close(self):
                        return None

                return OkStream()

        profile = {
            "api_key": "fixture-secret-key",
            "base_url": "https://example.invalid/v1",
            "model": "kimi-k3",
            "timeout": 300,
        }
        with tempfile.TemporaryDirectory() as tmp:
            archive = StreamingRequestArchive(Path(tmp) / "requests", HangingThenOkClient(), profile)
            text, attempt = archive.complete(
                "turn_01",
                [{"role": "user", "content": "hi"}],
                total_seconds=10,
                no_output_seconds=0.08,
                retries=2,
                retry_backoff_s=(0.01,),
            )
            self.assertEqual(text, '{"ok":true}')
            self.assertEqual(attempt.name, "attempt_002")
            err = json.loads((attempt.parent / "attempt_001/error.json").read_text())
            self.assertEqual(err["error_type"], "NoOutputTimeout")
            self.assertEqual(err["failure_kind"], "no_output_timeout")
            self.assertIs(err["output_received"], False)
            self.assertTrue(all(t <= 10.0 for t in calls))

    def test_streaming_archive_does_not_retry_after_partial_output(self):
        calls = []

        class PartialThenFailClient:
            def __init__(self):
                self.chat = NS(completions=NS(create=self.create))

            def create(self, **kwargs):
                calls.append(1)

                class PartialStream:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def __iter__(self):
                        yield NS(
                            model="test-model",
                            id="resp-1",
                            usage=None,
                            choices=[NS(index=0, delta=NS(reasoning_content="thinking", content=None, tool_calls=None), finish_reason=None)],
                        )
                        raise TimeoutError("mid-stream disconnect")

                    def close(self):
                        return None

                return PartialStream()

        profile = {"api_key": "", "base_url": "https://example.invalid/v1", "model": "kimi-k3", "timeout": 300}
        with tempfile.TemporaryDirectory() as tmp:
            archive = StreamingRequestArchive(Path(tmp) / "requests", PartialThenFailClient(), profile)
            with self.assertRaisesRegex(StreamingRequestFailure, "Streaming request failed") as caught:
                archive.complete("turn_01", [{"role": "user", "content": "hi"}], retries=2, retry_backoff_s=(0.01,))
            self.assertEqual(len(calls), 1)
            self.assertEqual(caught.exception.error_type,"TimeoutError")
            self.assertEqual(caught.exception.phase,"reading_response_stream")
            self.assertIsInstance(caught.exception.__cause__,TimeoutError)
            self.assertEqual(caught.exception.failure_kind,'transport_timeout')
            self.assertIs(caught.exception.output_received,True)

    def test_defense_request_recovery_accepts_no_output_timeout_on_latest_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "defense"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"stop_reason": "request_or_execution_failed", "selected_candidate_id": None, "model_actions": 1})
            )
            attempt2 = run_dir / "requests/turn_02_abc/attempt_002"
            attempt2.mkdir(parents=True)
            (attempt2 / "error.json").write_text(
                json.dumps({"error_type": "NoOutputTimeout", "chunks_received": 0, "output_received": False})
            )
            found = retryable_request(run_dir)
            self.assertIsNotNone(found)
            self.assertEqual(found["turn"], 2)
            self.assertEqual(found["error_type"], "NoOutputTimeout")


if __name__ == "__main__":
    unittest.main()
