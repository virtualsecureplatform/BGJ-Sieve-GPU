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


DIMENSION = int(os.environ.get("SVP_DIMENSION", "142"))
LATTICE_SEED = int(os.environ.get("SVP142_LATTICE_SEED", "0"))
SIEVE_SEED = os.environ.get("SVP_SIEVE_SEED", "0")
DEFAULT_TSD = 132
DEFAULT_MLD = 118
BKZ_BETA = int(os.environ.get("SVP_BKZ_BETA", "64" if DIMENSION >= 143 else "60"))
BKZ_LOOPS = int(os.environ.get("SVP_BKZ_LOOPS", "8"))
PREPROCESS_MODE = os.environ.get(
    "SVP_PREPROCESS_MODE", "lll-deeplll-bkz" if DIMENSION >= 143 else "lll-bkz"
)
DEEPLLL_DEPTH = int(os.environ.get("SVP_DEEPLLL_DEPTH", "4"))
BLASTER_APP = os.environ.get("SVP_BLASTER_APP", "")
TARGET_NORM2 = int(os.environ.get("SVP142_TARGET_NORM2", "9075417"))
GENERATOR_URL = "https://www.latticechallenge.org/svp-challenge/generator.php"
RAW_SHA256 = os.environ.get(
    "SVP142_RAW_SHA256",
    "bd449bb1bccbd9c927a2895bf1ca1f5d62a7864c0c4e7e305f1da854dca96674",
)
STRATEGY_SHA256 = "f516b0a6f0c580cff72e1e2c3562c44dc6f17e8f99613e9e4020e35481b27a18"
DEFAULT_STRATEGY = Path("/usr/local/share/fplll/strategies/default.json")
HERE = Path(__file__).resolve().parent
KNOWN_VECTOR = Path(os.environ.get("SVP142_KNOWN_VECTOR", str(HERE / "known-vector.txt")))
LEGACY_GENERATOR = os.environ.get("SVP142_LEGACY_GENERATOR", "")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SOL_RE = re.compile(
    r"\[pos (\d+)\] length = ([0-9.]+)\(([0-9.]+) gh, [0-9.eE+-]+ old\), "
    r"vec = \[([^\]]*)\]"
)
TIMING_RE = re.compile(
    r"^\[svp142-timing\] phase=(?P<phase>[A-Za-z0-9_-]+) "
    r"event=(?P<event>start|end) epoch=(?P<epoch>[0-9]+(?:\.[0-9]+)?)"
    r"(?: elapsed_seconds=(?P<elapsed>[0-9]+(?:\.[0-9]+)?))?"
    r"(?: returncode=(?P<returncode>-?[0-9]+))?\s*$"
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
    if PREPROCESS_MODE == "lll-potlllbkz":
        lll_name = f"L_{DIMENSION}_{LATTICE_SEED}.lll-heuristic-mpfr256"
    elif PREPROCESS_MODE == "potlll-bkz":
        lll_name = f"L_{DIMENSION}_{LATTICE_SEED}.potlll-heuristic-mpfr256"
    else:
        lll_name = f"L_{DIMENSION}_{LATTICE_SEED}.lll"
    if PREPROCESS_MODE == "lll-deeplll-bkz":
        suffix = f"d{DEEPLLL_DEPTH}.p{BKZ_BETA}l{BKZ_LOOPS}"
    elif PREPROCESS_MODE == "lll-potlllbkz":
        suffix = f"lll-heuristic-mpfr256.potlllbkz.p{BKZ_BETA}l{BKZ_LOOPS}"
    elif PREPROCESS_MODE == "potlllbkz":
        suffix = f"potlllbkz.p{BKZ_BETA}l{BKZ_LOOPS}"
    elif PREPROCESS_MODE == "potlll-bkz":
        suffix = f"potlll-heuristic-mpfr256.bkz.p{BKZ_BETA}l{BKZ_LOOPS}"
    else:
        suffix = f"p{BKZ_BETA}l{BKZ_LOOPS}"
    legacy_manifest = BKZ_BETA == 60 and BKZ_LOOPS == 8 and (
        PREPROCESS_MODE != "lll-deeplll-bkz" or DEEPLLL_DEPTH == 4
    )
    manifest_tag = PREPROCESS_MODE if legacy_manifest else suffix
    return (
        input_dir / f"L_{DIMENSION}_{LATTICE_SEED}.raw",
        input_dir / lll_name,
        input_dir / f"L_{DIMENSION}_{LATTICE_SEED}.{suffix}.pre",
        input_dir / f"preprocess-manifest-{manifest_tag}.json",
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
    if raw.exists() and (RAW_SHA256 == "GENERATE" or sha256(raw) == RAW_SHA256):
        validate_matrix(raw)
        print(f"[svp142] raw basis cached and verified: {sha256(raw)}")
        return raw
    input_dir.mkdir(parents=True, exist_ok=True)
    tmp = raw.with_name(f"{raw.name}.tmp.{os.getpid()}")
    if RAW_SHA256 == "GENERATE":
        actual = ""
    else:
        body = urllib.parse.urlencode(
            {"dimension": DIMENSION, "seed": LATTICE_SEED, "sent": "true"}
        ).encode("ascii")
        request = urllib.request.Request(
            GENERATOR_URL,
            data=body,
            headers={"User-Agent": "BGJ-Sieve-GPU SVP-142 recipe"},
        )
        print(f"[svp142] downloading official dimension {DIMENSION}, seed {LATTICE_SEED} basis")
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256(tmp)
    if RAW_SHA256 == "GENERATE" or actual != RAW_SHA256:
        if not LEGACY_GENERATOR:
            raise ReproductionError(
                "raw basis generation requires SVP142_LEGACY_GENERATOR"
            )
        generator = Path(LEGACY_GENERATOR)
        if not generator.is_file() or not os.access(generator, os.X_OK):
            raise ReproductionError(f"missing legacy generator: {generator}")
        print(f"[svp142] official response unusable ({actual}); using pinned NTL 9.3 generator")
        with tmp.open("wb") as output:
            proc = subprocess.run(
                [str(generator), str(DIMENSION), str(LATTICE_SEED)],
                stdout=output,
                check=False,
            )
        if proc.returncode:
            raise ReproductionError(f"legacy generator exited {proc.returncode}")
        actual = sha256(tmp)
    if RAW_SHA256 != "GENERATE" and actual != RAW_SHA256:
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


def run_to_file(
    command: list[str],
    output: Path,
    accept_nonzero: bool = False,
    phase: str | None = None,
) -> int:
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    started_wall = time.time()
    started_mono = time.monotonic()
    if phase:
        print(
            f"[svp142-timing] phase={phase} event=start "
            f"epoch={started_wall:.3f}",
            flush=True,
        )
    print("[svp142] " + " ".join(command) + f" > {output}")
    with tmp.open("wb") as destination:
        proc = subprocess.run(command, stdout=destination, check=False)
    if phase:
        ended_wall = time.time()
        print(
            f"[svp142-timing] phase={phase} event=end "
            f"epoch={ended_wall:.3f} elapsed_seconds={time.monotonic() - started_mono:.2f} "
            f"returncode={proc.returncode}",
            flush=True,
        )
    if proc.returncode and not accept_nonzero:
        raise ReproductionError(
            f"command exited {proc.returncode}; partial output retained at {tmp}"
        )
    validate_matrix(tmp)
    tmp.replace(output)
    return proc.returncode


def extract_timing(log_paths: list[Path]) -> dict[str, object]:
    """Extract timestamp-based preprocessing durations from recipe logs."""
    reports: list[dict[str, object]] = []
    for log_path in log_paths:
        phases: dict[str, dict[str, object]] = {}
        for line_number, line in enumerate(log_path.read_text().splitlines(), 1):
            match = TIMING_RE.match(line.strip())
            if not match:
                continue
            phase = match.group("phase")
            event = match.group("event")
            entry = phases.setdefault(phase, {})
            epoch = float(match.group("epoch"))
            if event == "start":
                entry["start_epoch"] = epoch
            else:
                entry["end_epoch"] = epoch
                if match.group("returncode") is not None:
                    entry["returncode"] = int(match.group("returncode"))
                if match.group("elapsed") is not None:
                    entry["reported_elapsed_seconds"] = float(match.group("elapsed"))
            entry["last_line"] = line_number

        complete: dict[str, dict[str, object]] = {}
        for phase, entry in phases.items():
            start = entry.get("start_epoch")
            end = entry.get("end_epoch")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            duration = end - start
            if duration < 0:
                raise ReproductionError(f"timing timestamps go backwards for {phase} in {log_path}")
            entry = dict(entry)
            entry["elapsed_seconds"] = round(duration, 3)
            entry.pop("last_line", None)
            complete[phase] = entry
        reports.append(
            {
                "log": str(log_path),
                "phases": complete,
                "preprocess_seconds": round(
                    sum(float(entry["elapsed_seconds"]) for entry in complete.values()), 3
                ),
            }
        )
    return {"logs": reports}


def run_deeplll(source: Path, output: Path) -> None:
    app = Path(BLASTER_APP)
    if not app.is_file() or DEEPLLL_DEPTH < 1:
        raise ReproductionError("DeepLLL requires SVP_BLASTER_APP and positive depth")
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    command = [
        sys.executable, str(app), "--depth", str(DEEPLLL_DEPTH), "--delta", "0.99",
        "--lll_size", "64", "--cores", "32", "--input", str(source),
        "--output", str(tmp),
    ]
    print("[svp142] " + " ".join(command))
    proc = subprocess.run(command, check=False)
    if proc.returncode:
        raise ReproductionError(f"BLASter DeepLLL exited {proc.returncode}")
    validate_matrix(tmp)
    tmp.replace(output)


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
    if PREPROCESS_MODE not in {
        "lll-bkz", "lll-deeplll-bkz", "potlllbkz", "lll-potlllbkz", "potlll-bkz"
    }:
        raise ReproductionError(f"unsupported preprocessing mode: {PREPROCESS_MODE}")
    version = fplll_version()
    if not version.startswith("fplll "):
        raise ReproductionError(f"unexpected fplll version output: {version or 'unknown'}")
    fplll_commit = os.environ.get("BGJ_FPLLL_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", fplll_commit):
        raise ReproductionError("BGJ_FPLLL_COMMIT must identify the vendored fplll revision")
    if not strategy.is_file() or sha256(strategy) != STRATEGY_SHA256:
        raise ReproductionError(f"unexpected or missing fplll strategy: {strategy}")
    raw = download_raw(input_dir)
    _, lll, pre, manifest_path = input_paths(input_dir)

    blaster_commit = os.environ.get("BGJ_BLASTER_COMMIT", "")
    if PREPROCESS_MODE == "lll-deeplll-bkz" and not re.fullmatch(r"[0-9a-f]{40}", blaster_commit):
        raise ReproductionError("BGJ_BLASTER_COMMIT must identify the vendored BLASter revision")
    expected = {
        "dimension": DIMENSION,
        "lattice_seed": LATTICE_SEED,
        "raw_sha256": sha256(raw),
        "fplll_version": version,
        "fplll_commit": fplll_commit,
        "strategy_sha256": STRATEGY_SHA256,
        "pipeline": (
            f"LLL -> BLASter DeepLLL-{DEEPLLL_DEPTH} -> BKZ-{BKZ_BETA} "
            f"(pruned fplll strategy, max {BKZ_LOOPS} loops)"
            if PREPROCESS_MODE == "lll-deeplll-bkz"
            else (
                f"integrated PotLLLBKZ-{BKZ_BETA} "
                f"(pruned fplll strategy, max {BKZ_LOOPS} loops)"
                if PREPROCESS_MODE == "potlllbkz"
                else (
                    f"PotLLL (heuristic, MPFR 256-bit) -> BKZ-{BKZ_BETA} "
                    f"(pruned fplll strategy, max {BKZ_LOOPS} loops)"
                    if PREPROCESS_MODE == "potlll-bkz"
                    else (
                        f"LLL (heuristic, MPFR 256-bit) -> PotLLLBKZ-{BKZ_BETA} "
                        f"(pruned fplll strategy, max {BKZ_LOOPS} loops)"
                        if PREPROCESS_MODE == "lll-potlllbkz"
                        else f"LLL -> BKZ-{BKZ_BETA} "
                        f"(pruned fplll strategy, max {BKZ_LOOPS} loops)"
                    )
                )
            )
        ),
        "preprocess_mode": PREPROCESS_MODE,
        "blaster_commit": blaster_commit if PREPROCESS_MODE == "lll-deeplll-bkz" else None,
        "bkz_beta": BKZ_BETA,
        "bkz_max_loops": BKZ_LOOPS,
        "lll_method": "heuristic" if PREPROCESS_MODE in {"lll-potlllbkz", "potlll-bkz"} else None,
        "lll_float_type": "mpfr" if PREPROCESS_MODE in {"lll-potlllbkz", "potlll-bkz"} else None,
        "lll_precision": 256 if PREPROCESS_MODE in {"lll-potlllbkz", "potlll-bkz"} else None,
    }
    have_intermediate = PREPROCESS_MODE == "potlllbkz" or lll.is_file()
    if not force and manifest_path.is_file() and have_intermediate and pre.is_file():
        cached = json.loads(manifest_path.read_text())
        if all(cached.get(key) == value for key, value in expected.items()):
            expected_lll_sha = None if PREPROCESS_MODE == "potlllbkz" else sha256(lll)
            if cached.get("lll_sha256") == expected_lll_sha and cached.get("pre_sha256") == sha256(pre):
                if PREPROCESS_MODE != "potlllbkz":
                    validate_matrix(lll)
                validate_matrix(pre)
                print(f"[svp142] reusing verified preprocessing: {cached['pre_sha256']}")
                return pre

    pre_bkz = raw
    if PREPROCESS_MODE != "potlllbkz":
        lll_algorithm = "potlll" if PREPROCESS_MODE == "potlll-bkz" else "lll"
        lll_command = ["fplll", "-a", lll_algorithm]
        if PREPROCESS_MODE in {"lll-potlllbkz", "potlll-bkz"}:
            lll_command.extend(["-m", "heuristic", "-f", "mpfr", "-p", "256"])
        run_to_file(
            lll_command + [str(raw)], lll,
            phase="potlll" if PREPROCESS_MODE == "potlll-bkz" else "lll",
        )
        pre_bkz = lll
    if PREPROCESS_MODE == "lll-deeplll-bkz":
        deep = input_dir / f"L_{DIMENSION}_{LATTICE_SEED}.deeplll{DEEPLLL_DEPTH}"
        run_deeplll(lll, deep)
        pre_bkz = deep
    bkz_algorithm = (
        "potlllbkz"
        if PREPROCESS_MODE in {"potlllbkz", "lll-potlllbkz"}
        else "bkz"
    )
    rc = run_to_file(
        [
            "fplll", "-a", bkz_algorithm, "-b", str(BKZ_BETA), "-s", str(strategy),
            "-bkzmaxloops", str(BKZ_LOOPS), str(pre_bkz),
        ],
        pre,
        accept_nonzero=True,
        phase="bkz" if bkz_algorithm == "bkz" else "potlllbkz",
    )
    if rc:
        print(f"[svp142] fplll returned {rc} at the tour limit; complete output accepted")
    manifest = dict(expected)
    manifest.update(
        {
            "lll_sha256": None if PREPROCESS_MODE == "potlllbkz" else sha256(lll),
            "pre_sha256": sha256(pre),
            "known_vector": (
                verify_vector_text(KNOWN_VECTOR.read_text(), raw)
                if KNOWN_VECTOR.is_file() else None
            ),
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
    expected_raw = manifest.get("raw_sha256") if RAW_SHA256 == "GENERATE" else RAW_SHA256
    if manifest.get("raw_sha256") != expected_raw or sha256(raw) != expected_raw:
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
    continue_after_target: bool,
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
        "continue_after_target": continue_after_target,
        "target_norm2": TARGET_NORM2,
        "raw_sha256": sha256(raw),
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
                    if verification["ok"] and "t_solution" not in result:
                        result["verification"] = verification
                        result["t_solution"] = round(elapsed, 2)
                        if not continue_after_target:
                            result["status"] = "solved"
                            break
            if result["status"] == "running":
                returncode = proc.wait()
                if continue_after_target and returncode == 0 and "best_verification" in result:
                    best = result["best_verification"]
                    if best["norm2"] < TARGET_NORM2:
                        result["status"] = "shorter"
                    elif best["norm2"] == TARGET_NORM2:
                        result["status"] = "target-only"
                    else:
                        result["status"] = "completed-no-target"
                else:
                    result["status"] = f"exited_rc{returncode}"
    finally:
        stop_process(proc)
        proc.wait()
        selector.close()
    result["t_total"] = round(time.monotonic() - started, 2)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[svp142] {result['status']}; result: {result_path}")
    return 0 if result["status"] in ("solved", "shorter", "target-only") else 1


def show_plan(input_dir: Path, run_dir: Path) -> None:
    pipeline = (
        f"LLL -> DeepLLL-{DEEPLLL_DEPTH} -> pruned BKZ-{BKZ_BETA}"
        if PREPROCESS_MODE == "lll-deeplll-bkz"
        else (
            f"integrated pruned PotLLLBKZ-{BKZ_BETA}"
            if PREPROCESS_MODE == "potlllbkz"
            else (
                f"PotLLL (heuristic, MPFR 256-bit) -> pruned BKZ-{BKZ_BETA}"
                if PREPROCESS_MODE == "potlll-bkz"
                else (
                    f"LLL (heuristic, MPFR 256-bit) -> pruned PotLLLBKZ-{BKZ_BETA}"
                    if PREPROCESS_MODE == "lll-potlllbkz"
                    else f"LLL -> pruned BKZ-{BKZ_BETA}"
                )
            )
        )
    )
    print(f"""SVP-{DIMENSION} seed-{LATTICE_SEED} plan
1. Obtain and hash-pin the official {DIMENSION}x{DIMENSION} seed-{LATTICE_SEED} basis in {input_dir}.
2. Build the vendored dependencies, then run {pipeline}, maximum {BKZ_LOOPS} loops.
3. Build the four-A100 binary with the 240/96/24 GiB host-cache profile.
4. Sieve from MLD {DEFAULT_MLD} through TSD {DEFAULT_TSD} in {run_dir}.
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
    timing = sub.add_parser(
        "timing", help="extract timestamp-based PotLLL/BKZ durations from prep logs"
    )
    timing.add_argument("log", type=Path, nargs="+")
    run = sub.add_parser("run")
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=10500)
    run.add_argument("--tsd", type=int, default=DEFAULT_TSD)
    run.add_argument("--mld", type=int, default=DEFAULT_MLD)
    run.add_argument("--continue-after-target", action="store_true")
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
        if args.action == "timing":
            print(json.dumps(extract_timing([path.resolve() for path in args.log]), indent=2))
            return 0
        if args.action == "run":
            return run_sieve(
                args.input_dir.resolve(), args.run_dir.resolve(), args.binary.resolve(),
                args.timeout, args.tsd, args.mld, args.continue_after_target,
            )
    except (OSError, ReproductionError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"[svp142] error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
