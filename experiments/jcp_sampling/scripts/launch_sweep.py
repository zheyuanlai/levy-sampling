from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_suite(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data.get("configs", [])
    return data


def extract_run_dir(stdout: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        m = re.search(r"DONE\s+(\S+)\s+status=", line)
        if m:
            return m.group(1)
        if line.startswith("results/jcp_sampling/"):
            return line
    return None


def persist_child_logs(run_dir: str | None, stdout: str, stderr: str) -> None:
    if not run_dir:
        return
    path = Path(run_dir)
    if not path.exists():
        return
    log_dir = path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_text(stdout)
    (log_dir / "stderr.log").write_text(stderr)


def main():
    ap = argparse.ArgumentParser(description="Launch JCP configs with bounded GPU concurrency.")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--output-root", default="results/jcp_sampling")
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.max_concurrent > 2:
        raise SystemExit("Refusing to exceed --max-concurrent 2 per AGENTS.md")
    if len(gpus) > 2:
        raise SystemExit("Refusing to use more than two GPUs per AGENTS.md")
    configs = load_suite(args.suite)
    manifest_dir = Path(args.output_root) / "launcher_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    running = []
    completed = []
    idx = 0
    while idx < len(configs) or running:
        while idx < len(configs) and len(running) < args.max_concurrent:
            if gpus:
                busy_gpus = {rec.get("gpu") for _, rec in running}
                free_gpus = [g for g in gpus if g not in busy_gpus]
                if not free_gpus:
                    break
                gpu = free_gpus[0]
            else:
                gpu = ""
            cfg = configs[idx]
            env = os.environ.copy()
            if gpu:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            cmd = [sys.executable, "-m", "experiments.jcp_sampling.scripts.run_experiment", "--config", cfg, "--output-root", args.output_root]
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            rec = {"config": cfg, "gpu": gpu, "pid": proc.pid, "cmd": cmd, "start_time": time.time()}
            running.append((proc, rec)); idx += 1
            with manifest_path.open("a") as f:
                f.write(json.dumps({**rec, "event": "start"}) + "\n")
        time.sleep(1)
        still = []
        for proc, rec in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, rec)); continue
            out, err = proc.communicate()
            run_dir = extract_run_dir(out)
            persist_child_logs(run_dir, out, err)
            done = {**rec, "event": "finish", "returncode": ret, "end_time": time.time(),
                    "run_dir": run_dir or "", "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
            completed.append(done)
            with manifest_path.open("a") as f:
                f.write(json.dumps(done) + "\n")
            print(out, end="")
            if err:
                print(err, file=sys.stderr, end="")
        running = still
    nfail = sum(1 for r in completed if r["returncode"] != 0)
    print(f"launcher manifest: {manifest_path}")
    if nfail:
        raise SystemExit(f"{nfail} launched jobs failed")


if __name__ == "__main__":
    main()
