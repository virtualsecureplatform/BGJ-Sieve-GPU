import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "bkz_scheduler.py"
SPEC = importlib.util.spec_from_file_location("bkz_scheduler", str(MODULE_PATH))
SCHEDULER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEDULER)


class BkzSchedulerTests(unittest.TestCase):
    def test_algorithm_9_d4f(self):
        self.assertEqual(SCHEDULER.default_d4f(40), 0)
        self.assertEqual(SCHEDULER.default_d4f(100), 19)
        self.assertEqual(SCHEDULER.default_d4f(150), 22)

    def test_block_size_is_translated_to_bsd(self):
        stages = SCHEDULER.normalize_stages({"stages": [{"block_size": 100, "jump": 9}]})
        self.assertEqual(stages[0]["d4f"], 19)
        self.assertEqual(stages[0]["bsd"], 81)
        self.assertEqual(stages[0]["block_size"], 100)

    def test_sieve_dimension_preserves_hd_sieve_semantics(self):
        stages = SCHEDULER.normalize_stages({
            "stages": [{"sieve_dimension": 115, "d4f": 30, "tours": 2}]
        })
        self.assertEqual(stages[0]["bsd"], 115)
        self.assertEqual(stages[0]["block_size"], 145)
        self.assertEqual(stages[0]["tours"], 2)

    def test_final_sieve_arguments(self):
        command = SCHEDULER.final_sieve_command("hd_sieve", "basis", {
            "target_sieving_dimension": 120, "min_lifting_dimension": 118
        })
        self.assertEqual(command[-4:], ["--TSD", "120", "--MLD", "118"])


if __name__ == "__main__":
    unittest.main()
