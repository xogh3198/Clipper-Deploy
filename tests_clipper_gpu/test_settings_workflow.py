"""Model/batch settings must reach the independent worker/frontend."""
import contextlib
import io
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

from clipper_gpu import deploy
from clipper_gpu.config import ROOT, load, compose_spec

MODEL = {"MODEL_NAME": "resnet-18", "MODEL_ID": "microsoft/resnet-18",
         "MODEL_REVISION": "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"}


class SettingsWorkflowTests(unittest.TestCase):
    def test_swap_model_gpu_and_batch_configuration(self):
        with patch.dict(os.environ, {**MODEL, "GPU": "3", "BATCH_SIZE": "4", "SLO_MS": "400"}):
            c = load()
        worker = compose_spec(c)["services"]["worker"]
        self.assertEqual(worker["environment"]["MODEL_ID"], MODEL["MODEL_ID"])
        self.assertEqual(worker["environment"]["MODEL_REVISION"], MODEL["MODEL_REVISION"])
        self.assertEqual(worker["environment"]["CLIPPER_MODEL_NAME"], MODEL["MODEL_NAME"])
        self.assertEqual(worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"], ["3"])
        self.assertEqual(c["BATCH_SIZE"], "4")

    def test_download_uses_configured_non_resnet50_model(self):
        with patch.dict(os.environ, MODEL), patch.object(sys, "argv", ["clipper", "download", "--models", "resnet-18"]), \
             patch.object(deploy, "paths"), patch.object(deploy, "verify_images"), patch.object(deploy, "oneoff") as call:
            self.assertEqual(deploy.main(), 0)
            self.assertEqual(call.call_args.args[0]["MODEL_NAME"], "resnet-18")

    def test_wrong_cli_model_is_not_ignored(self):
        with patch.object(sys, "argv", ["clipper", "download", "--models", "resnet-18"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                deploy.main()

    def test_setup_links_and_fences(self):
        text = (ROOT / "setting_readme.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("```") % 2, 0)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" not in target:
                self.assertTrue((ROOT / target.split("#")[0]).is_file(), target)
        self.assertIn("setting_readme.md", (ROOT / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
