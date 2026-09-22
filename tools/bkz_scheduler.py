#!/usr/bin/env python3
"""Checkpointed progressive pnj-BKZ driver for hd_sieve.

This program is intentionally only an orchestrator.  All lattice reduction is
performed by hd_sieve and should be run inside a Slurm allocation.
"""

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


PAPER_SCHEDULES = {
    # Table 3 of Wang, Wang, and Wang, ASIA CCS 2023.
    130: [41, 53, 63, 69, 72, 77, 81, 89, 97, 104, 118],
    140: [40, 51, 61, 69, 78, 88, 95, 103, 110, 114, 117, 120, 123, 128],
    150: [40, 51, 64, 73, 84, 97, 103, 110, 118, 124, 131, 138],
    154: [40, 51, 57, 65, 69, 76, 81, 86, 90, 97, 103, 112, 118, 124, 130, 136, 142],
}


def shell_join(arguments):
    """shlex.join for the older Python versions on some clusters."""
    return " ".join(shlex.quote(str(argument)) for argument in arguments)


def run_identity(source, stages, final_sieve=None):
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    digest.update(json.dumps(stages, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps(final_sieve, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:16]


def default_d4f(block_size):
    """Algorithm 9's default f, with f_extra=0."""
    return min(max(0, (block_size - 40) // 2), int(11.5 + 0.075 * block_size))


def load_plan(path, preset):
    if bool(path) == bool(preset):
        raise ValueError("specify exactly one of --schedule or --paper-preset")
    if preset:
        dim = int(preset)
        if dim not in PAPER_SCHEDULES:
            raise ValueError("paper preset must be one of: " + ", ".join(map(str, PAPER_SCHEDULES)))
        return {
            "name": "asia-ccs-2023-svp-%d-j9" % dim,
            "stages": [{"block_size": beta, "jump": 9} for beta in PAPER_SCHEDULES[dim]],
        }
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict) or not isinstance(plan.get("stages"), list):
        raise ValueError("schedule must be a JSON object containing a stages list")
    return plan


def normalize_stages(plan):
    defaults = plan.get("defaults", {})
    result = []
    for number, raw in enumerate(plan["stages"], 1):
        if not isinstance(raw, dict):
            raise ValueError("stage %d must be an object" % number)
        stage = dict(defaults)
        stage.update(raw)
        d4f = int(stage.get("d4f", default_d4f(int(stage.get("block_size", 0)))))
        if "sieve_dimension" in stage:
            bsd = int(stage["sieve_dimension"])
            beta = bsd + d4f
        elif "block_size" in stage:
            beta = int(stage["block_size"])
            bsd = beta - d4f
        else:
            raise ValueError("stage %d needs block_size or sieve_dimension" % number)
        jump = int(stage.get("jump", 1))
        tours = int(stage.get("tours", 1))
        start = int(stage.get("start_index", 0))
        bdh = int(stage.get("dual_hash_ratio", 300))
        if beta < 40 or bsd < 40:
            raise ValueError("stage %d has unsupported sieving dimension %d (< 40)" % (number, bsd))
        if d4f < 0 or jump < 1 or tours < 1 or start < 0 or bdh < 0:
            raise ValueError("stage %d contains a negative value or zero jump/tours" % number)
        result.append({"number": number, "block_size": beta, "bsd": bsd,
                       "d4f": d4f, "jump": jump, "tours": tours,
                       "start_index": start, "bdh": bdh})
    if not result:
        raise ValueError("schedule has no stages")
    return result


def checkpoint_name(stage, tour):
    return "stage-%03d-tour-%02d-b%03d-j%02d-f%02d.basis" % (
        stage["number"], tour, stage["block_size"], stage["jump"], stage["d4f"])


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def command_for(binary, input_name, output_name, stage):
    return [str(binary), "--input", input_name, "--task", "bkz",
            "--BSD", str(stage["bsd"]), "--JUMP", str(stage["jump"]),
            "--D4F", str(stage["d4f"]), "--BDH", str(stage["bdh"]),
            "--STI", str(stage["start_index"]), "--output", output_name]


def final_sieve_command(binary, input_name, settings):
    command = [str(binary), "--input", input_name, "--task", "sieve"]
    for field, option in (("target_sieving_dimension", "--TSD"),
                          ("current_sieving_dimension", "--CSD"),
                          ("min_lifting_dimension", "--MLD")):
        if field in settings:
            value = int(settings[field])
            if value < 1:
                raise ValueError("final_sieve.%s must be positive" % field)
            command.extend((option, str(value)))
    return command


def first_vector_norm2(path):
    """Return the squared norm of the first row in an fplll-style basis."""
    values = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            stripped = stripped.lstrip("[").rstrip("]")
            if stripped:
                values = [int(value) for value in stripped.split()]
                break
    if not values:
        raise ValueError("cannot read first basis vector from %s" % path)
    return sum(value * value for value in values)


def print_plan(plan, stages):
    print("schedule: %s" % plan.get("name", "unnamed"))
    print("stage tour block BSD D4F jump start BDH")
    for stage in stages:
        for tour in range(1, stage["tours"] + 1):
            print("%5d %4d %5d %3d %3d %4d %5d %3d" % (
                stage["number"], tour, stage["block_size"], stage["bsd"],
                stage["d4f"], stage["jump"], stage["start_index"], stage["bdh"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--schedule", type=Path)
    choice.add_argument("--paper-preset", choices=map(str, sorted(PAPER_SCHEDULES)))
    parser.add_argument("--binary", type=Path, default=Path("app/hd_sieve"))
    parser.add_argument("--work-dir", required=True, type=Path,
                        help="job-local directory containing .pool/.bucket/.sol/.uid")
    parser.add_argument("--output", type=Path, help="copy the last checkpoint here")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = load_plan(args.schedule, args.paper_preset)
        stages = normalize_stages(plan)
        final_sieve = plan.get("final_sieve")
        if final_sieve is not None and not isinstance(final_sieve, dict):
            raise ValueError("final_sieve must be an object")
        target_norm2 = plan.get("target_norm2")
        if target_norm2 is not None:
            target_norm2 = int(target_norm2)
            if target_norm2 < 1:
                raise ValueError("target_norm2 must be positive")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print_plan(plan, stages)

    source = args.input.resolve()
    binary = args.binary.resolve()
    work_dir = args.work_dir.resolve()
    scratch_input = "bkz-scheduler-input.basis"
    scratch_output = "bkz-scheduler-output.basis"
    if args.dry_run:
        first = Path("bkz-checkpoints") / "RUN_ID" / "stage-000-input.basis"
        current = first.name
        for stage in stages:
            for tour in range(1, stage["tours"] + 1):
                target = checkpoint_name(stage, tour)
                print("DRY-RUN copy %s -> %s" % (current, scratch_input))
                print("DRY-RUN " + shell_join(
                    command_for(binary, scratch_input, scratch_output, stage)))
                print("DRY-RUN move %s -> %s" % (scratch_output, target))
                current = target
        if final_sieve is not None:
            print("DRY-RUN copy %s -> %s" % (current, scratch_input))
            print("DRY-RUN " + shell_join(final_sieve_command(binary, scratch_input, final_sieve)))
        return 0

    if not source.is_file():
        parser.error("input does not exist: %s" % source)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("hd_sieve is not executable: %s" % binary)
    identity = run_identity(source, stages, final_sieve)
    checkpoints = work_dir / "bkz-checkpoints" / identity
    first = checkpoints / "stage-000-input.basis"
    print("run identity: %s" % identity, flush=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    if not first.exists():
        atomic_copy(source, first)

    current = first
    target_reached = False
    for stage in stages:
        for tour in range(1, stage["tours"] + 1):
            target = checkpoints / checkpoint_name(stage, tour)
            if target.exists():
                print("resume: keeping completed %s" % target.name, flush=True)
                current = target
                if target_norm2 is not None and first_vector_norm2(current) <= target_norm2:
                    target_reached = True
                    break
                continue
            local_input = work_dir / scratch_input
            local_output = work_dir / scratch_output
            # hd_sieve derives dot-prefixed temporary names from --input, so it
            # must receive a slash-free job-local name.
            for scratch in (local_input, local_output):
                try:
                    scratch.unlink()
                except FileNotFoundError:
                    pass
            atomic_copy(current, local_input)
            command = command_for(binary, scratch_input, scratch_output, stage)
            print("running: " + shell_join(command), flush=True)
            subprocess.run(command, cwd=work_dir, check=True)
            if not local_output.is_file() or local_output.stat().st_size == 0:
                raise RuntimeError("hd_sieve returned success without %s" % local_output)
            os.replace(local_output, target)
            local_input.unlink()
            current = target
            if target_norm2 is not None:
                norm2 = first_vector_norm2(current)
                print("checkpoint first-vector norm2: %d (target %d)" %
                      (norm2, target_norm2), flush=True)
                if norm2 <= target_norm2:
                    print("target reached; stopping progressive BKZ", flush=True)
                    target_reached = True
                    break
        if target_reached:
            break

    final_marker = checkpoints / "final-sieve.complete"
    if final_sieve is not None and not target_reached and final_marker.exists():
        print("resume: final sieve already completed", flush=True)
    elif final_sieve is not None and not target_reached:
        local_input = work_dir / scratch_input
        atomic_copy(current, local_input)
        command = final_sieve_command(binary, scratch_input, final_sieve)
        print("running final sieve: " + shell_join(command), flush=True)
        subprocess.run(command, cwd=work_dir, check=True)
        local_input.unlink()
        final_marker.touch()

    if args.output:
        output = args.output.resolve()
        if output.exists() and not output.samefile(current):
            if not filecmp.cmp(str(output), str(current), shallow=False):
                raise FileExistsError("refusing to overwrite output: %s" % output)
        if not output.exists():
            atomic_copy(current, output)
        print("completed: %s" % output)
    else:
        print("completed: %s" % current)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, RuntimeError, FileExistsError) as error:
        print("bkz_scheduler: %s" % error, file=sys.stderr)
        sys.exit(1)
