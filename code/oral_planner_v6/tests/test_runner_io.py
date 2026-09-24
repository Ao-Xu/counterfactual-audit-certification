"""Regression checks for packaging fixes, without collecting new outcomes."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oral_planner_v6 import run


class RunnerIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "fresh" / "results"
        replace_out = patch.object(run, "OUT", self.out)
        replace_out.start()
        self.addCleanup(replace_out.stop)

    def lock(self, manifest=None):
        self.out.mkdir(parents=True, exist_ok=True)
        data = {"manifest_sha": manifest or (run.R / "manifest.sha256").read_text().strip(),
                "pilots": [], "oracles": []}
        raw = json.dumps(data).encode()
        (self.out / "budget_lock.json").write_bytes(raw)
        (self.out / "budget_lock.sha256").write_text(hashlib.sha256(raw).hexdigest())

    def test_fresh_output_directory(self):
        run.write("progress.json", {"stage": "unit-test"})
        self.assertEqual(json.loads((self.out / "progress.json").read_text()), {"stage": "unit-test"})
        self.assertFalse((self.out / "progress.json.tmp").exists())

    def test_lock_tampering_rejected(self):
        self.lock()
        self.assertEqual(run.read_lock()["pilots"], [])
        (self.out / "budget_lock.json").write_text("{}")
        with self.assertRaisesRegex(AssertionError, "budget lock changed"):
            run.read_lock()

    def test_different_manifest_rejected(self):
        self.lock("0" * 64)
        with self.assertRaisesRegex(AssertionError, "another manifest"):
            run.read_lock()

    def test_existing_outcomes_refused(self):
        self.lock()
        (self.out / "final.json").write_text("{}")
        (self.out / "benchmarks.json").write_text("{}")
        for call, message in [(run.pilots, "immutable"), (run.finals, "already exist"),
                              (run.benchmark, "benchmark exists")]:
            with self.subTest(stage=call.__name__), self.assertRaisesRegex(AssertionError, message):
                call()

    def test_benchmark_checks_lock_before_sampling(self):
        self.lock()
        (self.out / "budget_lock.json").write_text("{}")
        with patch.object(run, "draw_stats") as draw:
            with self.assertRaisesRegex(AssertionError, "budget lock changed"):
                run.benchmark()
            draw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
