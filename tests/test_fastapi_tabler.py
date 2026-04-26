import importlib.util
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from lib.config import METHOD_PROMPTMATCH
from lib.web_context import HybridScorerContext


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("hybrid_entry", os.path.join(ROOT, "Hybrid-Scorer.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastApiTablerTests(unittest.TestCase):
    def make_context(self):
        with patch("lib.web_context.require_cuda"), patch("lib.web_context.configure_torch_cpu_threads"):
            return HybridScorerContext(ROOT, "HybridScorer", "vtest", "")

    def test_view_state_shape(self):
        ctx = self.make_context()
        payload = ctx.to_payload()
        self.assertIn("view", payload)
        self.assertIn("controls", payload)
        self.assertEqual(payload["inputs"]["method"], METHOD_PROMPTMATCH)
        self.assertIn("main", payload["controls"]["sliders"])

    def test_override_split_in_dto(self):
        ctx = self.make_context()
        image_path = os.path.join(ROOT, "images", "Flux_corset_00001_.png")
        fname = os.path.basename(image_path)
        ctx.state["scores"] = {
            fname: {"pos": 0.1, "neg": None, "path": image_path, "failed": False},
        }
        ctx.state["overrides"] = {fname: "SELECTED"}
        ctx.inputs["main_threshold"] = 0.9
        payload = ctx.to_payload()
        self.assertEqual(len(payload["view"]["left"]["items"]), 1)
        self.assertTrue(payload["view"]["left"]["items"][0]["overridden"])

    def test_setup_page(self):
        entry = load_entrypoint()
        client = TestClient(entry.create_setup_required_app(["missing-package"]))
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Setup Update Required", response.text)

    def test_media_unknown_is_404(self):
        entry = load_entrypoint()
        with patch("lib.web_context.require_cuda"), patch("lib.web_context.configure_torch_cpu_threads"):
            client = TestClient(entry.create_fastapi_app())
        self.assertEqual(client.get("/media/not-registered").status_code, 404)

    def test_threshold_endpoint_smoke(self):
        entry = load_entrypoint()
        with patch("lib.web_context.require_cuda"), patch("lib.web_context.configure_torch_cpu_threads"):
            client = TestClient(entry.create_fastapi_app())
        response = client.post("/api/thresholds", json={"main_threshold": 0.2, "aux_threshold": 0.1})
        self.assertEqual(response.status_code, 200)
        self.assertIn("view", response.json())


if __name__ == "__main__":
    unittest.main()
