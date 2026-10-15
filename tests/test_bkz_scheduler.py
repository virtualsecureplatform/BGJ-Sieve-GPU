import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import subprocess


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "bkz_scheduler.py"
SPEC = importlib.util.spec_from_file_location("bkz_scheduler", str(MODULE_PATH))
SCHEDULER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEDULER)


class BkzSchedulerTests(unittest.TestCase):
    def test_mpi_pause_does_not_publish_a_completed_pump(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            basis, schedule, work = root/'input', root/'plan.json', root/'work'
            basis.write_text('[[3 4]\n[0 1]]\n')
            schedule.write_text(json.dumps({'stages': [
                {'sieve_dimension': 60, 'd4f': 20, 'dual_hash_ratio': 0, 'jump': 9}]}))
            with mock.patch.object(SCHEDULER.subprocess, 'run',
                                   return_value=subprocess.CompletedProcess([], 75)) as runner:
                with contextlib.redirect_stdout(io.StringIO()):
                    status = SCHEDULER.main(['--input', str(basis), '--schedule', str(schedule),
                        '--binary', '/bin/true', '--backend', 'mpi', '--work-dir', str(work)])
            self.assertEqual(status, 75)
            self.assertIn('--nodes=2', runner.call_args[0][0])
            self.assertEqual(list(work.glob('bkz-checkpoints/*/stage-001-*.basis')), [])

    def test_mpi_rejects_dual_hash_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = root/'plan.json'
            schedule.write_text(json.dumps({'stages': [{'block_size': 80}]}))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    SCHEDULER.main(['--input', str(root/'input'), '--schedule', str(schedule),
                        '--backend', 'mpi', '--work-dir', str(root/'work'), '--dry-run'])

    def test_mpi_command_has_two_ranks_and_stable_checkpoint_root(self):
        command = SCHEDULER.mpi_command(["solver", "--task", "bkz"], "/checkpoint", 64)
        self.assertIn("--nodes=2", command)
        self.assertIn("--ntasks-per-node=1", command)
        self.assertEqual(command[-6:], ["--checkpoint-root", "/checkpoint", "--resume", "auto",
                                      "--mpi-frame-mib", "64"])

    def test_mpi_identity_does_not_reuse_serial_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            basis = Path(directory)/"basis"
            basis.write_text("[[3 4][0 1]]")
            serial = SCHEDULER.run_identity(basis, [])
            self.assertEqual(serial, SCHEDULER.run_identity(basis, [], backend="serial"))
            self.assertNotEqual(serial, SCHEDULER.run_identity(basis, [], backend="mpi"))
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

    def test_pump_checkpoint_stage_passes_stop_index(self):
        stage = SCHEDULER.normalize_stages({
            "stages": [{"sieve_dimension": 143, "d4f": 21,
                        "jump": 9, "start_index": 9, "stop_index": 18}]
        })[0]
        command = SCHEDULER.command_for("hd_sieve", "in", "out", stage)
        self.assertEqual(command[-2:], ["--STOP_INDEX", "18"])

    def test_stop_index_must_exceed_start(self):
        with self.assertRaises(ValueError):
            SCHEDULER.normalize_stages({
                "stages": [{"block_size": 164, "start_index": 9,
                            "stop_index": 9}]
            })

    def test_final_sieve_arguments(self):
        command = SCHEDULER.final_sieve_command("hd_sieve", "basis", {
            "target_sieving_dimension": 120, "min_lifting_dimension": 118
        })
        self.assertEqual(command[-4:], ["--TSD", "120", "--MLD", "118"])

    def test_first_vector_norm2(self):
        with tempfile.TemporaryDirectory() as directory:
            basis = Path(directory) / "basis"
            basis.write_text("[[3 4]\n[0 1]]\n")
            self.assertEqual(SCHEDULER.first_vector_norm2(basis), 25)


if __name__ == "__main__":
    unittest.main()
