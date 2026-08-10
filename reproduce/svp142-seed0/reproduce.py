#!/usr/bin/env python3
"""Prepare, run, and verify the official SVP-142 seed-0 instance.

Expensive operations are available only through explicit ``prepare`` and
``run`` subcommands so that ``plan`` and ``verify-known`` are safe diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request


DIMENSION = 142
LATTICE_SEED = 0
SIEVE_SEED = "0"
DEFAULT_TSD = 132
DEFAULT_MLD = 118
BKZ_BETA = 60
BKZ_LOOPS = 8
TARGET_NORM2 = 9_075_417
GENERATOR_URL = "https://www.latticechallenge.org/svp-challenge/generator.php"
RAW_SHA256 = "bd449bb1bccbd9c927a2895bf1ca1f5d62a7864c0c4e7e305f1da854dca96674"
STRATEGY_SHA256 = "f516b0a6f0c580cff72e1e2c3562c44dc6f17e8f99613e9e4020e35481b27a18"
DEFAULT_STRATEGY = Path("/usr/share/libfplll9/strategies/default.json")
HERE = Path(__file__).resolve().parent
KNOWN_VECTOR = HERE / "known-vector.txt"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SOL_RE = re.compile(
    r"\[pos (\d+)\] length = ([0-9.]+)\(([0-9.]+) gh, [0-9.eE+-]+ old\), "
    r"vec = \[([^\]]*)\]"
)


class ReproductionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_paths(input_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        input_dir / "L_142_0.raw",
        input_dir / "L_142_0.lll",
        input_dir / "L_142_0.p60l8.pre",
        input_dir / "preprocess-manifest.json",
    )


def read_matrix(path: Path) -> list[list[int]]:
    rows = re.findall(r"\[([^\[\]]*)\]", path.read_text())
    matrix = [[int(value) for value in row.split()] for row in rows if row.strip()]
    if len(matrix) != DIMENSION or any(len(row) != DIMENSION for row in matrix):
        raise ReproductionError(f"expected a {DIMENSION}x{DIMENSION} basis in {path}")
    return matrix


def validate_matrix(path: Path) -> None:
    read_matrix(path)


def download_raw(input_dir: Path) -> Path:
    raw, _, _, _ = input_paths(input_dir)
    if raw.exists() and sha256(raw) == RAW_SHA256:
        validate_matrix(raw)
        print(f"[svp142] official raw basis cached and verified: {RAW_SHA256}")
        return raw
    input_dir.mkdir(parents=True, exist_ok=True)
    tmp = raw.with_name(f"{raw.name}.tmp.{os.getpid()}")
    body = urllib.parse.urlencode(
        {"dimension": DIMENSION, "seed": LATTICE_SEED, "sent": "true"}
    ).encode("ascii")
    request = urllib.request.Request(
        GENERATOR_URL,
        data=body,
        headers={"User-Agent": "BGJ-Sieve-GPU SVP-142 recipe"},
    )
    print("[svp142] downloading official dimension 142, seed 0 basis")
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = sha256(tmp)
    if actual != RAW_SHA256:
        raise ReproductionError(
            f"downloaded raw basis hash {actual}, expected {RAW_SHA256}; retained {tmp}"
        )
    validate_matrix(tmp)
    tmp.replace(raw)
    return raw


def fplll_version() -> str:
    binary = shutil.which("fplll")
    if not binary:
        raise ReproductionError("fplll is not installed")
    proc = subprocess.run([binary, "--version"], text=True, capture_output=True, check=False)
    lines = (proc.stdout or proc.stderr).splitlines()
    return lines[0].strip() if lines else ""


def run_to_file(command: list[str], output: Path, accept_nonzero: bool = False) -> int:
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    print("[svp142] " + " ".join(command) + f" > {output}")
    with tmp.open("wb") as destination:
        proc = subprocess.run(command, stdout=destination, check=False)
    if proc.returncode and not accept_nonzero:
        raise ReproductionError(
            f"command exited {proc.returncode}; partial output retained at {tmp}"
        )
    validate_matrix(tmp)
    tmp.replace(output)
    return proc.returncode


def verify_vector_text(text: str, raw: Path) -> dict[str, object]:
    values = text.strip()
    if values.startswith("[") and values.endswith("]"):
        values = values[1:-1]
    try:
        vector = [int(value) for value in values.split()]
    except ValueError as exc:
        raise ReproductionError(f"vector contains a non-integer: {exc}") from exc
    if len(vector) != DIMENSION:
        raise ReproductionError(f"vector has {len(vector)} coordinates, expected {DIMENSION}")
    basis = read_matrix(raw)
    determinant = basis[0][0]
    remainder = (
        vector[0] - sum(vector[i] * basis[i][0] for i in range(1, DIMENSION))
    ) % determinant
    norm2 = sum(value * value for value in vector)
    return {
        "ok": remainder == 0 and norm2 <= TARGET_NORM2,
        "in_lattice": remainder == 0,
        "norm2": norm2,
        "norm": math.sqrt(norm2),
        "target_norm2": TARGET_NORM2,
    }


def prepare(input_dir: Path, strategy: Path, force: bool) -> Path:
    raw = download_raw(input_dir)
    _, lll, pre, manifest_path = input_paths(input_dir)
    version = fplll_version()
    if version != "fplll 5.5.0":
        raise ReproductionError(f"requires fplll 5.5.0, found {version or 'unknown'}")
    if not strategy.is_file() or sha256(strategy) != STRATEGY_SHA256:
        raise ReproductionError(f"unexpected or missing fplll strategy: {strategy}")

    expected = {
        "dimension": DIMENSION,
        "lattice_seed": LATTICE_SEED,
        "raw_sha256": RAW_SHA256,
        "fplll_version": version,
        "strategy_sha256": STRATEGY_SHA256,
        "pipeline": "LLL -> BKZ-60 (pruned fplll strategy, max 8 loops)",
        "bkz_beta": BKZ_BETA,
        "bkz_max_loops": BKZ_LOOPS,
    }
    if not force and manifest_path.is_file() and lll.is_file() and pre.is_file():
        cached = json.loads(manifest_path.read_text())
        if all(cached.get(key) == value for key, value in expected.items()):
            if cached.get("lll_sha256") == sha256(lll) and cached.get("pre_sha256") == sha256(pre):
                validate_matrix(lll)
                validate_matrix(pre)
                print(f"[svp142] reusing verified preprocessing: {cached['pre_sha256']}")
                return pre

    run_to_file(["fplll", "-a", "lll", str(raw)], lll)
    rc = run_to_file(
        [
            "fplll", "-a", "bkz", "-b", str(BKZ_BETA), "-s", str(strategy),
            "-bkzmaxloops", str(BKZ_LOOPS), str(lll),
        ],
        pre,
        accept_nonzero=True,
    )
    if rc:
        print(f"[svp142] fplll returned {rc} at the tour limit; complete output accepted")
    manifest = dict(expected)
    manifest.update(
        {
            "lll_sha256": sha256(lll),
            "pre_sha256": sha256(pre),
            "known_vector": verify_vector_text(KNOWN_VECTOR.read_text(), raw),
        }
    )
    tmp = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(manifest_path)
    print(f"[svp142] preprocessing complete: {manifest['pre_sha256']}")
    return pre


def load_inputs(input_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    raw, _, pre, manifest_path = input_paths(input_dir)
    if not manifest_path.is_file():
        raise ReproductionError(f"missing preprocessing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("raw_sha256") != RAW_SHA256 or sha256(raw) != RAW_SHA256:
        raise ReproductionError("raw basis does not match the official pinned basis")
    if manifest.get("pre_sha256") != sha256(pre):
        raise ReproductionError("preprocessed basis does not match its manifest")
    validate_matrix(raw)
    validate_matrix(pre)
    return raw, pre, manifest


def preflight(binary: Path) -> None:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ReproductionError(f"missing executable solver: {binary}")
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise ReproductionError("nvidia-smi was not found")
    proc = subprocess.run(
        [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    gpus = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(gpus) != 4:
        raise ReproductionError(f"this binary requires exactly four visible GPUs; found {len(gpus)}")
    print("[svp142] GPUs: " + "; ".join(gpus))


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_sieve(
    input_dir: Path,
    run_dir: Path,
    binary: Path,
    timeout: float,
    tsd: int,
    mld: int,
) -> int:
    if not (mld <= tsd <= DIMENSION) or mld < 100:
        raise ReproductionError(f"invalid sieve dimensions MLD={mld}, TSD={tsd}")
    raw, pre, manifest = load_inputs(input_dir)
    preflight(binary)
    run_dir.mkdir(parents=True, exist_ok=False)
    for subdir in (".pool/0", ".pool/1", ".bucket/0", ".bucket/1", ".sol/0", ".sol/1", ".uid/0", ".uid/1"):
        (run_dir / subdir).mkdir(parents=True)
    command = [
        "stdbuf", "-oL", "-eL", str(binary.resolve()), "--task", "sieve",
        "--input", str(pre.resolve()), "--TSD", str(tsd), "--MLD", str(mld),
    ]
    env = dict(os.environ)
    env.update({"HD_SIEVE_SEED": SIEVE_SEED, "HD_CUDA_BLOCKING_SYNC": "0"})
    result: dict[str, object] = {
        "status": "running",
        "dimension": DIMENSION,
        "lattice_seed": LATTICE_SEED,
        "sieve_seed": int(SIEVE_SEED),
        "tsd": tsd,
        "mld": mld,
        "target_norm2": TARGET_NORM2,
        "raw_sha256": RAW_SHA256,
        "pre_sha256": manifest["pre_sha256"],
        "binary_sha256": sha256(binary),
        "command": " ".join(command),
        "progression": [],
    }
    print(f"[svp142] run directory: {run_dir}")
    print(f"[svp142] target norm^2: {TARGET_NORM2}")
    print("[svp142] " + " ".join(command))
    started = time.monotonic()
    proc = subprocess.Popen(
        command, cwd=run_dir, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
    )
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    best_norm2: int | None = None
    try:
        with (run_dir / "run.log").open("w") as log:
            while result["status"] == "running":
                elapsed = time.monotonic() - started
                running = proc.poll() is None
                if running and elapsed >= timeout:
                    result["status"] = "timeout"
                    break
                events = selector.select(timeout=1.0 if running else 0)
                if not events:
                    if not running:
                        break
                    continue
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(f"[{elapsed:10.2f}] {line}")
                    log.flush()
                    print(line, end="", flush=True)
                    match = SOL_RE.search(ANSI_RE.sub("", line))
                    if not match or int(match.group(1)) != 0:
                        continue
                    verification = verify_vector_text(match.group(4), raw)
                    norm2 = int(verification["norm2"])
                    result["progression"].append(
                        {"t": round(elapsed, 2), "displayed_length": float(match.group(2)), "norm2": norm2}
                    )
                    if verification["in_lattice"] and (best_norm2 is None or norm2 < best_norm2):
                        best_norm2 = norm2
                        result["best_verification"] = verification
                        result["t_best"] = round(elapsed, 2)
                    if verification["ok"]:
                        result["status"] = "solved"
                        result["verification"] = verification
                        result["t_solution"] = round(elapsed, 2)
                        break
            if result["status"] == "running":
                result["status"] = f"exited_rc{proc.wait()}"
    finally:
        stop_process(proc)
        proc.wait()
        selector.close()
    result["t_total"] = round(time.monotonic() - started, 2)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[svp142] {result['status']}; result: {result_path}")
    return 0 if result["status"] == "solved" else 1


def show_plan(input_dir: Path, run_dir: Path) -> None:
    print(f"""SVP-142 seed-0 plan
1. Download and hash-pin the official 142x142 seed-0 basis in {input_dir}.
2. Preprocess with fplll 5.5.0: LLL -> pruned BKZ-60, maximum 8 loops.
3. Build the four-A100 binary with the 112/96/24 GiB host-cache profile.
4. Sieve from MLD 118 through TSD 132 in {run_dir}.
5. Stop after independently verifying a lattice vector with norm^2 <= {TARGET_NORM2}.

No expensive command was executed by this plan.""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--input-dir", type=Path, required=True)
    plan.add_argument("--run-dir", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input-dir", type=Path, required=True)
    prep.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    prep.add_argument("--force", action="store_true")
    verify = sub.add_parser("verify-known")
    verify.add_argument("--input-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=10500)
    run.add_argument("--tsd", type=int, default=DEFAULT_TSD)
    run.add_argument("--mld", type=int, default=DEFAULT_MLD)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            show_plan(args.input_dir.resolve(), args.run_dir.resolve())
            return 0
        if args.action == "prepare":
            prepare(args.input_dir.resolve(), args.strategy.resolve(), args.force)
            return 0
        if args.action == "verify-known":
            raw = download_raw(args.input_dir.resolve())
            result = verify_vector_text(KNOWN_VECTOR.read_text(), raw)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.action == "run":
            return run_sieve(
                args.input_dir.resolve(), args.run_dir.resolve(), args.binary.resolve(),
                args.timeout, args.tsd, args.mld,
            )
    except (OSError, ReproductionError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"[svp142] error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
