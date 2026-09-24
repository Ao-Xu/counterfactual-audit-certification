"""Source package smoke tests."""

import unittest

from scripts.build_archive import ROOT, select_files


class ArchiveTests(unittest.TestCase):
    def test_curated_source_tree(self):
        names = {path.relative_to(ROOT).as_posix() for path in select_files()}
        self.assertIn("planning/run.py", names)
        self.assertIn("scripts/build_archive.py", names)
        self.assertIn("language_models/experiments/hans_suite/run_hans_suite.py", names)
        self.assertNotIn("CODE_INDEX.json", names)
        self.assertTrue(all("results" not in name.split("/") for name in names))
        self.assertTrue(all("figure_archive" not in name.split("/") for name in names))


if __name__ == "__main__":
    unittest.main()
