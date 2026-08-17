import io
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from tools.approximator_server import ApproximatorHandler


class RecordingUploadTests(unittest.TestCase):
    def make_handler(self, payload, content_type="video/webm", kind=None):
        handler = object.__new__(ApproximatorHandler)
        headers = Message()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(payload))
        if kind:
            headers["X-Recording-Kind"] = kind
        handler.headers = headers
        handler.rfile = io.BytesIO(payload)
        handler.response = None
        handler._send_json = lambda status, body: setattr(handler, "response", (status, body))
        return handler

    def test_saves_webm_in_recordings_directory(self):
        payload = b"\x1aE\xdf\xa3webm-test"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings = root / "outputs" / "recordings"
            with (
                patch("tools.approximator_server.ROOT_DIR", root),
                patch("tools.approximator_server.RECORDINGS_DIR", recordings),
            ):
                handler = self.make_handler(payload)
                handler._save_recording()

            self.assertEqual(handler.response[0], 201)
            self.assertTrue(handler.response[1]["ok"])
            saved = list(recordings.glob("training-*.webm"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_bytes(), payload)

    def test_saves_trajectory_recording_with_its_own_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings = root / "outputs" / "recordings"
            with (
                patch("tools.approximator_server.ROOT_DIR", root),
                patch("tools.approximator_server.RECORDINGS_DIR", recordings),
            ):
                handler = self.make_handler(b"trajectory", kind="trajectory")
                handler._save_recording()
            self.assertEqual(handler.response[0], 201)
            self.assertEqual(len(list(recordings.glob("trajectory-*.webm"))), 1)

    def test_rejects_non_webm_upload(self):
        handler = self.make_handler(b"not a video", "application/octet-stream")
        handler._save_recording()
        self.assertEqual(handler.response[0], 415)
        self.assertFalse(handler.response[1]["ok"])

    def test_config_reports_private_recorder_state(self):
        handler = object.__new__(ApproximatorHandler)
        handler.path = "/api/config"
        handler.response = None
        handler._send_json = lambda status, body: setattr(handler, "response", (status, body))
        with patch("tools.approximator_server.RECORDING_ENABLED", False):
            handler.do_GET()
        self.assertEqual(handler.response, (200, {"record_training_enabled": False}))
