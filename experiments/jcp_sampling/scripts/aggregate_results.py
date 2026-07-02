from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def as_float(v):
    try: x = float(v)
    except Exception: return None
    return x if np.isfinite(x) else None


def extract_run_dir(stdout: str) -> str | None:
    for line in reversed(str(stdout).splitlines()):
        line = line.strip()
        m = re.search(r"DONE\s+(\S+)\s+status=", line)
        if m:
            return m.group(1)
        if line.startswith("results/jcp_sampling/"):
            return line
    return None


def manifest_run_dirs(path: Path) -> list[Path]:
    dirs: list[Path] = []
    seen = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("event") != "finish" or int(rec.get("returncode", 1)) != 0:
                continue
            rd = rec.get("run_dir") or extract_run_dir(rec.get("stdout_tail", ""))
            if not rd or rd in seen:
                continue
            seen.add(rd); dirs.append(Path(rd))
    return dirs


def raw_metric_paths(root: Path, manifest: Path | None):
    if manifest:
        for d in manifest_run_dirs(manifest):
            p = d / "raw_metrics.csv"
            if p.exists():
                yield p
        return
    yield from root.rglob("raw_metrics.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--manifest", default=None, help="launcher manifest JSONL; if supplied, aggregate only completed run dirs from it")
    ap.add_argument("--include-smoke", action="store_true", help="include experiments whose names end with _smoke")
    args = ap.parse_args()
    root = Path(args.results_root)
    manifest = Path(args.manifest) if args.manifest else None
    raws = []
    used_paths = []
    for p in raw_metric_paths(root, manifest):
        used_paths.append(str(p))
        for r in read_csv(p):
            exp_name = str(r.get("experiment_name", ""))
            if (not args.include_smoke) and exp_name.endswith("_smoke"):
                continue
            r["run_dir"] = str(p.parent); raws.append(r)
    out_dir = root / "aggregate"; out_dir.mkdir(parents=True, exist_ok=True)
    if not raws: raise SystemExit(f"no raw_metrics.csv under {root} matching selection")
    write_csv(out_dir / "all_raw_metrics.csv", raws)
    group = ["experiment_name", "target_name", "method", "bank_name", "bank_scale", "bank_intensity"]
    groups = {}
    for r in raws:
        groups.setdefault(tuple(r.get(c, "") for c in group), []).append(r)
    rows = []
    for key, sub in sorted(groups.items()):
        rec = dict(zip(group, key)); rec["n_rows"] = len(sub); rec["n_failed"] = sum(1 for r in sub if r.get("status") != "ok")
        num_keys = sorted({k for r in sub for k, v in r.items() if as_float(v) is not None and k != "seed"})
        for m in num_keys:
            vals = np.array([as_float(r.get(m)) for r in sub if as_float(r.get(m)) is not None], dtype=float)
            if vals.size:
                rec[f"{m}_mean"] = float(vals.mean())
                rec[f"{m}_se"] = float(vals.std(ddof=1)/np.sqrt(vals.size)) if vals.size > 1 else 0.0
        rows.append(rec)
    write_csv(out_dir / "all_summary.csv", rows)
    (out_dir / "aggregate_manifest.json").write_text(json.dumps({
        "results_root": str(root),
        "launcher_manifest": str(manifest) if manifest else None,
        "raw_metric_paths": used_paths,
        "n_raw_rows": len(raws),
        "n_summary_rows": len(rows),
        "include_smoke": bool(args.include_smoke),
    }, indent=2))
    print(out_dir)

if __name__ == "__main__": main()
