"""Re-run an experiment's production with the multi-atom estimator (LSC-CP-MA),
overwriting the single-R-RA CSVs. Inherited dt + default q_theta (refinement is
diligence, not correctness). PT ladder tuned. Usage:

    JCP_EXTRA_GPUS=0 JCP_GPU=0 python scripts/rerun_production_ma.py <experiment>
    <experiment> in {mb3well_10d, coupled_phi4}
"""
import os, sys, time
_JCP = "/home/zheyuanlai/levy-sampling/JCP_experiments"
sys.path.insert(0, _JCP)
from src.gpu_guard import select_gpu
select_gpu(os.environ.get("JCP_GPU", "0"))
import torch; torch.set_default_dtype(torch.float64)
import numpy as np
from src.experiments import (build_e3, build_e4, make_batched_factory, make_metrics)
from src.samplers import tune_ladder
from src.runner import (run_experiment_batched, write_timeseries_csv,
                        write_summary_csv, checkpoint_schedule)
from src.config import N_CHECKPOINTS

_BUILD = {"mb3well_10d": build_e3, "coupled_phi4": build_e4}
DEV = "cuda"
name = sys.argv[1]
RES = os.path.join(_JCP, "results", name)
exp = _BUILD[name](device=DEV, basin_cache=os.path.join(RES, "basin_map.npz"))
cfg = exp.cfg
print(f"{name} MA re-run: N={cfg.n_particles} T={cfg.T} dt={cfg.dt} beta={cfg.beta}")
metrics_fn, floors, aux = make_metrics(exp, cfg.n_particles, device=DEV)

g = torch.Generator(device=DEV); g.manual_seed(0)
x0p = exp.init_fn(min(512, cfg.n_particles), g)
pt_betas, ladder = tune_ladder(exp.pot, x0p, cfg.dt, exp.box, cfg.beta,
                               exp.pt_beta_min, pilot_steps=20_000)
print(f"PT ladder K={ladder['K']} swap_acc={ladder['swap_acceptance']:.3f}")

methods = ["ULA", "MALA", "FLA", "BAOAB", "PT", "CP", "LSC-CP-MA"]
n_steps = cfg.n_steps
ck = checkpoint_schedule(n_steps)
bf = make_batched_factory(exp, cfg.dt, pt_betas, cfg.seeds, n_particles=cfg.n_particles)
exp.pot.reset_counters()
t0 = time.time()
rows, info = run_experiment_batched(methods, list(cfg.seeds), bf, n_steps,
                                    max(1, n_steps // N_CHECKPOINTS), cfg.dt,
                                    metrics_fn, exp.pot, cfg.n_particles,
                                    checkpoint_steps=ck)
print(f"production {time.time()-t0:.0f}s")
assert max(r["nonfinite_frac"] for r in rows) == 0.0
extra = ["W2_10d"] if cfg.d == 10 else []
write_timeseries_csv(rows, os.path.join(RES, "metrics_timeseries.csv"))
write_summary_csv(rows, methods, list(cfg.seeds),
                  ["W2", "TV", "MMD", "EMC"] + extra + ["basin_rel_max", "basin_L1", "nonfinite_frac"],
                  info, floors, os.path.join(RES, "summary.csv"))
last = [r for r in rows if r["step"] == max(x["step"] for x in rows)]
print(f"terminal basin_rel_max (p*={exp.p_star.cpu().numpy().round(3)}):")
for m in ("PT", "CP", "LSC-CP-MA"):
    brm = np.mean([r["basin_rel_max"] for r in last if r["method"] == m])
    print(f"  {m:11s} basin_rel_max={brm:.3f}")
print(f"{name} MA re-run DONE")
