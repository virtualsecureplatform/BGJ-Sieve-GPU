#!/usr/bin/env python3
"""Reproduce the SVP-138, lattice-seed-2, sieve-seed-0 result.

The expensive actions are explicit subcommands.  In particular, importing this
module or invoking ``plan`` never starts fplll or the GPU solver.
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


DIMENSION = 138
LATTICE_SEED = 2
SIEVE_SEED = "0"
TSD = 128
MLD = 114
BKZ_BETA = 60
BKZ_LOOPS = 8
TARGET_NORM2 = 8_496_181
STALE_SECONDS = 2700
TIMEOUT_SECONDS = 10800

GENERATOR_URL = "https://www.latticechallenge.org/svp-challenge/generator.php"
RAW_SHA256 = "afe6bda79e60ce5be463c40420fc89bf93dc7f22047324e005acf915972ba9e2"
LLL_SHA256 = "db21b2735e0cf7e8c747ad60693d7c9ae6b64d1481380ffa31996fbfd5604b7c"
PRE_SHA256 = "f902fd5b8b72171b084f8d62202c31a05ef18b9547fe99b23eb39446d355faeb"
STRATEGY_SHA256 = "f516b0a6f0c580cff72e1e2c3562c44dc6f17e8f99613e9e4020e35481b27a18"
SUCCESS_BINARY_SHA256 = "5d76fb2a2176dc6c1b8f4f8563f475016533cdeeb43022c117ee8be65b81ddfd"
SUCCESS_COMMIT = "c586b64"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KNOWN_VECTOR = HERE / "known-vector.txt"
DEFAULT_WORKDIR = HERE / "work"
DEFAULT_STRATEGY = Path("/usr/share/libfplll9/strategies/default.json")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SOL_RE = re.compile(
    r"\[pos (\d+)\] length = ([0-9.]+)\(([0-9.]+) gh, [0-9.eE+-]+ old\), "
    r"vec = \[([^\]]*)\]"
)


class ReproductionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected: str, label: str) -> bool:
    if not path.exists():
        return False
    actual = sha256(path)
    if actual != expected:
        raise ReproductionError(
            f"{label} has SHA-256 {actual}, expected {expected}: {path}"
        )
    print(f"[reproduce] {label}: cached and verified ({expected[:12]})")
    return True


def paths(workdir: Path) -> tuple[Path, Path, Path]:
    return (
        workdir / "L_138_2.raw",
        workdir / "L_138_2.lll",
        workdir / "L_138_2.p60l8.pre",
    )


def download_raw(workdir: Path) -> Path:
    raw, _, _ = paths(workdir)
    if checked_file(raw, RAW_SHA256, "official raw lattice"):
        return raw

    workdir.mkdir(parents=True, exist_ok=True)
    tmp = raw.with_name(f"{raw.name}.tmp.{os.getpid()}")
    body = urllib.parse.urlencode(
        {"dimension": DIMENSION, "seed": LATTICE_SEED, "sent": "true"}
    ).encode("ascii")
    print(f"[reproduce] downloading dimension {DIMENSION}, seed {LATTICE_SEED}")
    request = urllib.request.Request(
        GENERATOR_URL,
        data=body,
        headers={"User-Agent": "BGJ-Sieve-GPU reproducibility recipe"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as dst:
            shutil.copyfileobj(response, dst)
        actual = sha256(tmp)
        if actual != RAW_SHA256:
            raise ReproductionError(
                f"downloaded lattice has SHA-256 {actual}, expected {RAW_SHA256}; "
                f"untrusted response retained at {tmp}"
            )
        tmp.replace(raw)
    except Exception:
        # Retain a response with a wrong hash for diagnosis.  Empty/partial
        # network failures are safe to remove on the next invocation.
        if tmp.exists() and tmp.stat().st_size == 0:
            tmp.unlink()
        raise
    print(f"[reproduce] official raw lattice verified ({RAW_SHA256[:12]})")
    return raw


def fplll_version() -> str:
    binary = shutil.which("fplll")
    if not binary:
        raise ReproductionError("fplll is not installed")
    proc = subprocess.run(
        [binary, "--version"], text=True, capture_output=True, check=False
    )
    first = (proc.stdout or proc.stderr).splitlines()
    return first[0].strip() if first else ""


def run_to_file(command: list[str], output: Path, accept_nonzero: bool = False) -> int:
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    print("[reproduce] " + " ".join(command) + f" > {output}")
    with tmp.open("wb") as dst:
        proc = subprocess.run(command, stdout=dst, check=False)
    if proc.returncode and not accept_nonzero:
        raise ReproductionError(
            f"command exited {proc.returncode}; partial output retained at {tmp}"
        )
    return proc.returncode


def finish_generated(tmp: Path, output: Path, expected: str, label: str) -> None:
    actual = sha256(tmp)
    if actual != expected:
        raise ReproductionError(
            f"{label} has SHA-256 {actual}, expected {expected}; output retained at {tmp}"
        )
    tmp.replace(output)
    print(f"[reproduce] {label} verified ({expected[:12]})")


def prepare(workdir: Path, strategy: Path) -> Path:
    raw = download_raw(workdir)
    _, lll, pre = paths(workdir)

    version = fplll_version()
    if version != "fplll 5.5.0":
        raise ReproductionError(
            f"this recipe requires fplll 5.5.0, found {version or 'unknown version'}"
        )
    if not strategy.is_file():
        raise ReproductionError(f"BKZ strategy file not found: {strategy}")
    strategy_hash = sha256(strategy)
    if strategy_hash != STRATEGY_SHA256:
        raise ReproductionError(
            f"BKZ strategy has SHA-256 {strategy_hash}, expected {STRATEGY_SHA256}"
        )

    if not checked_file(lll, LLL_SHA256, "LLL basis"):
        run_to_file(["fplll", "-a", "lll", str(raw)], lll)
        finish_generated(
            lll.with_name(f"{lll.name}.tmp.{os.getpid()}"), lll, LLL_SHA256, "LLL basis"
        )

    if checked_file(pre, PRE_SHA256, "BKZ-60 basis"):
        return pre

    tmp = pre.with_name(f"{pre.name}.tmp.{os.getpid()}")
    rc = run_to_file(
        [
            "fplll",
            "-a",
            "bkz",
            "-b",
            str(BKZ_BETA),
            "-s",
            str(strategy),
            "-bkzmaxloops",
            str(BKZ_LOOPS),
            str(lll),
        ],
        pre,
        accept_nonzero=True,
    )
    rows = sum(1 for line in tmp.open(errors="replace") if "[" in line)
    if rows < DIMENSION:
        raise ReproductionError(
            f"BKZ output contains only {rows}/{DIMENSION} rows and was retained at {tmp}"
        )
    if rc:
        print(f"[reproduce] fplll returned {rc} at its tour limit; complete output will be checked")
    finish_generated(tmp, pre, PRE_SHA256, "BKZ-60 basis")
    return pre


def build(workdir: Path) -> Path:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidates = [
            Path("/usr/local/cuda/bin/nvcc"),
            *sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True),
        ]
        nvcc = next(
            (str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)),
            None,
        )
    if not nvcc:
        raise ReproductionError("nvcc was not found on PATH or under /usr/local/cuda*")
    print(f"[reproduce] nvcc: {nvcc}")
    workdir.mkdir(parents=True, exist_ok=True)
    define = "-DHD_SVP140_CACHE_PROFILE=1"
    commands = [
        ["make", "-C", str(REPO / "src"), "clean", f"NVCC={nvcc}"],
        ["make", "-C", str(REPO / "src"), f"INC_DIR={define}", f"NVCC={nvcc}"],
        [
            "make",
            "-C",
            str(REPO / "app"),
            "hd_sieve",
            f"INC_DIR={define}",
            f"NVCC={nvcc}",
        ],
    ]
    for command in commands:
        print("[reproduce] " + " ".join(command))
        subprocess.run(command, check=True)
    output = workdir / "hd_sieve_svp140"
    shutil.copy2(REPO / "app" / "hd_sieve", output)
    print(f"[reproduce] SVP-140 cache-profile binary: {output}")
    return output


def read_matrix(path: Path) -> list[list[int]]:
    rows = re.findall(r"\[([^\[\]]*)\]", path.read_text())
    matrix = [[int(value) for value in row.split()] for row in rows if row.strip()]
    if len(matrix) != DIMENSION or any(len(row) != DIMENSION for row in matrix):
        raise ReproductionError(f"expected a {DIMENSION}x{DIMENSION} basis in {path}")
    return matrix


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


def git_state() -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or "unknown"


def preflight(binary: Path, workdir: Path) -> None:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ReproductionError(f"solver binary is missing or not executable: {binary}")
    binary_hash = sha256(binary)
    if binary_hash == SUCCESS_BINARY_SHA256:
        print(f"[reproduce] exact successful binary verified ({binary_hash[:12]})")
    else:
        print(
            f"[reproduce] warning: binary SHA-256 is {binary_hash}; the successful binary was "
            f"{SUCCESS_BINARY_SHA256}"
        )

    smi = shutil.which("nvidia-smi")
    if not smi:
        raise ReproductionError("nvidia-smi was not found; two CUDA GPUs are required")
    proc = subprocess.run(
        [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    gpus = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(gpus) < 2:
        raise ReproductionError(f"the binary is configured for two GPUs; found {len(gpus)}")
    print("[reproduce] GPUs: " + "; ".join(gpus[:2]))

    free_gib = shutil.disk_usage(workdir).free / (1 << 30)
    if free_gib < 150:
        print(f"[reproduce] warning: only {free_gib:.1f} GiB is free in {workdir}")


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_sieve(workdir: Path, binary: Path, timeout: float, stale: float) -> int:
    raw, _, pre = paths(workdir)
    if not checked_file(raw, RAW_SHA256, "official raw lattice"):
        raise ReproductionError(f"raw lattice is missing; run the prepare stage first: {raw}")
    if not checked_file(pre, PRE_SHA256, "BKZ-60 basis"):
        raise ReproductionError(f"BKZ basis is missing; run the prepare stage first: {pre}")
    preflight(binary, workdir)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = workdir / "runs" / f"{timestamp}_svp138-seed2-r0-p60l8"
    for subdir in (".pool/0", ".pool/1", ".bucket/0", ".bucket/1", ".sol/0", ".sol/1", ".uid/0", ".uid/1"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    command = [
        "stdbuf",
        "-oL",
        "-eL",
        str(binary.resolve()),
        "--task",
        "sieve",
        "--input",
        str(pre.resolve()),
        "--TSD",
        str(TSD),
        "--MLD",
        str(MLD),
    ]
    env = dict(os.environ)
    env.update(
        {
            "HD_SIEVE_SEED": SIEVE_SEED,
            # These restore the behavior of successful commit c586b64 when
            # running the newer compatibility commit 6a89ef9.
            "HD_CUDA_BLOCKING_SYNC": "0",
            "HD_BGJ2_REDUCER_THREADS": "32",
        }
    )
    display = (
        "HD_SIEVE_SEED=0 HD_CUDA_BLOCKING_SYNC=0 HD_BGJ2_REDUCER_THREADS=32 "
        + " ".join(command)
    )
    print(f"[reproduce] run directory: {run_dir}")
    print(f"[reproduce] {display}")

    result: dict[str, object] = {
        "status": "running",
        "command": display,
        "dimension": DIMENSION,
        "lattice_seed": LATTICE_SEED,
        "sieve_seed": int(SIEVE_SEED),
        "tsd": TSD,
        "mld": MLD,
        "target_norm2": TARGET_NORM2,
        "repo_commit": git_state(),
        "successful_commit": SUCCESS_COMMIT,
        "raw_sha256": RAW_SHA256,
        "pre_sha256": PRE_SHA256,
        "binary_sha256": sha256(binary),
        "progression": [],
    }
    started = time.monotonic()
    best_length: float | None = None
    best_time: float | None = None
    proc = subprocess.Popen(
        command,
        cwd=run_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    log_path = run_dir / "run.log"
    try:
        with log_path.open("w") as log:
            while result["status"] == "running":
                elapsed = time.monotonic() - started
                running = proc.poll() is None
                if running and elapsed >= timeout:
                    result["status"] = "timeout"
                    break
                if running and best_time is not None and stale and elapsed - best_time >= stale:
                    result["status"] = "stale"
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
                    if not match:
                        continue
                    pos = int(match.group(1))
                    length = float(match.group(2))
                    result["progression"].append(
                        {"t": round(elapsed, 2), "pos": pos, "length": length}
                    )
                    if pos == 0 and (best_length is None or length < best_length - 1e-6):
                        best_length, best_time = length, elapsed
                    # The solver's displayed length is a GPU-side estimate.  It
                    # can differ from sqrt(sum(x_i^2)) by more than its two
                    # printed decimal places (the reproduced target appeared as
                    # 2914.85 although its exact norm is 2914.8209...).  Check
                    # every leading candidate using integer arithmetic instead
                    # of using that estimate as a correctness prefilter.
                    if pos == 0:
                        verification = verify_vector_text(match.group(4), raw)
                        if verification["ok"]:
                            result.update(
                                status="solved",
                                t_solution=round(elapsed, 2),
                                verification=verification,
                            )
                            break
        if result["status"] == "running":
            result["status"] = f"exited_rc{proc.returncode}"
    finally:
        stop_process(proc)
        proc.wait()
        selector.close()

    result["t_total"] = round(time.monotonic() - started, 2)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[reproduce] {result['status']}; result: {result_path}")
    print("[reproduce] run data was preserved; remove the run directory manually when finished")
    return 0 if result["status"] == "solved" else 1


def show_plan(workdir: Path, strategy: Path) -> None:
    raw, lll, pre = paths(workdir)
    print(
        f"""SVP-138 seed-2 reproduction plan
1. POST dimension=138, seed=2 to:
   {GENERATOR_URL}
   -> {raw} (SHA-256 {RAW_SHA256})
2. fplll 5.5.0 -a lll {raw}
   -> {lll} (SHA-256 {LLL_SHA256})
3. fplll 5.5.0 -a bkz -b 60 -s {strategy} -bkzmaxloops 8 {lll}
   -> {pre} (SHA-256 {PRE_SHA256})
4. Build with HD_SVP140_CACHE_PROFILE=1 (47/32/8 GiB caches).
5. Run with lattice seed 2, HD_SIEVE_SEED=0, TSD=128, MLD=114.
6. Verify exact lattice membership and squared norm <= {TARGET_NORM2}.

No command in this plan was executed."""
    )


def common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    plan_parser = subparsers.add_parser("plan", help="print steps without executing them")
    common_options(plan_parser)

    prepare_parser = subparsers.add_parser("prepare", help="download, LLL, and BKZ the lattice")
    common_options(prepare_parser)

    build_parser = subparsers.add_parser("build", help="build the SVP-140 cache-profile binary")
    build_parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)

    verify_parser = subparsers.add_parser("verify-known", help="verify the recorded vector only")
    common_options(verify_parser)

    run_parser = subparsers.add_parser("run", help="run only the GPU sieve on prepared data")
    common_options(run_parser)
    run_parser.add_argument("--binary", type=Path)
    run_parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    run_parser.add_argument("--stale", type=float, default=STALE_SECONDS)

    all_parser = subparsers.add_parser("all", help="download, prepare, build, and run")
    common_options(all_parser)
    all_parser.add_argument("--binary", type=Path, help="use this binary instead of building")
    all_parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    all_parser.add_argument("--stale", type=float, default=STALE_SECONDS)

    args = parser.parse_args()
    workdir = args.workdir.resolve()
    strategy = getattr(args, "strategy", DEFAULT_STRATEGY).resolve()
    try:
        if args.action == "plan":
            show_plan(workdir, strategy)
            return 0
        if args.action == "prepare":
            prepare(workdir, strategy)
            return 0
        if args.action == "build":
            build(workdir)
            return 0
        if args.action == "verify-known":
            raw = download_raw(workdir)
            result = verify_vector_text(KNOWN_VECTOR.read_text(), raw)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.action == "all":
            prepare(workdir, strategy)
            binary = args.binary.resolve() if args.binary else build(workdir)
            return run_sieve(workdir, binary, args.timeout, args.stale)
        if args.action == "run":
            binary = args.binary
            if binary is None:
                built = workdir / "hd_sieve_svp140"
                binary = built if built.exists() else REPO / "app" / "hd_sieve_140P"
            return run_sieve(workdir, binary.resolve(), args.timeout, args.stale)
    except (OSError, ReproductionError, subprocess.CalledProcessError) as exc:
        print(f"[reproduce] error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
