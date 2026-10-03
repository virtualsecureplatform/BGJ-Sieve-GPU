#!/usr/bin/env python3
"""Supervise an isolated compute-node replay and capture idle CSD143 stacks."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


def tail(path, size=65536):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode(errors="replace")


def solver_pids(binary):
    result = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            if item.stat().st_uid == os.getuid() and (item / "exe").resolve() == binary:
                result.append(int(item.name))
        except (OSError, RuntimeError):
            pass
    return result


def capture(root, binary, number):
    path = root / ("snapshot-%02d.txt" % number)
    with path.open("w") as output:
        output.write("time=%s\n" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        output.flush()
        commands = [
            ["nvidia-smi"],
            ["ps", "-u", str(os.getuid()), "-L", "-o",
             "pid,tid,pcpu,stat,wchan:32,comm", "--sort=-pcpu"],
            ["free", "-h"],
        ]
        for pid in solver_pids(binary):
            commands.append(["gdb", "-batch", "-ex", "set pagination off",
                             "-ex", "set sysroot /proc/%d/root" % pid,
                             "-ex", "thread apply all bt 12",
                             str(binary), "-p", str(pid)])
        for command in commands:
            output.write("\ncommand=%r\n" % command)
            output.flush()
            try:
                subprocess.run(command, stdout=output, stderr=output, timeout=90,
                               check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                output.write("diagnostic_error=%s\n" % error)
    print("captured %s" % path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--idle-minutes", type=int, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing replay command")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    log = root / "replay.log"
    idle, snapshot, reached143 = 0, 0, False
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=output,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                time.sleep(60)
                if process.poll() is not None:
                    break
                text = tail(log)
                reached143 = reached143 or "phase=reducer_buffers_ready csd=143" in text
                if not reached143:
                    continue
                try:
                    gpu = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"], timeout=15, text=True)
                    values = [int(row.strip()) for row in gpu.splitlines() if row.strip()]
                    idle = idle + 1 if values and all(value == 0 for value in values) else 0
                except (ValueError, OSError, subprocess.SubprocessError):
                    snapshot += 1
                    capture(root, binary, snapshot)
                    continue
                print("CSD143 idle_samples=%d/%d" % (idle, args.idle_minutes), flush=True)
                if idle in (2, 10, args.idle_minutes):
                    snapshot += 1
                    capture(root, binary, snapshot)
                if idle >= args.idle_minutes:
                    print("Sustained GPU-idle CSD143 replay; stopping after capture.", flush=True)
                    return 124
            return process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
