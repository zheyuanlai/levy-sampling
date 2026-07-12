"""Generate the four experiment notebooks (01-04) with nbformat.

Run from JCP_experiments/notebooks:  python build_notebooks.py
Markdown is kept minimal by design: target introduction + jump law / Levy
score only; hyperparameters and protocol details live in the code cells.
"""
from __future__ import annotations

import nbformat as nbf


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


# ======================================================================
# shared code cells
# ======================================================================
def cell_setup(exp_name: str, builder: str, extra: str = "") -> str:
    return f'''EXPERIMENT = "{exp_name}"
import os, sys, math, time, json
sys.path.insert(0, os.path.abspath(".."))
from src.gpu_guard import select_gpu
select_gpu(int(os.environ.get("JCP_GPU", "4")))
import torch
assert torch.cuda.device_count() == 1
torch.set_default_dtype(torch.float64)
import numpy as np
import pandas as pd

from src import config as C
from src.experiments import ({builder}, make_sampler_factory,
                             make_batched_factory, make_metrics)
from src.runner import (run_experiment_batched, run_one, refine_dt,
                        quadrature_refinement, write_timeseries_csv,
                        write_summary_csv, write_manifest,
                        ula_first_passage, hardware_manifest)
from src.samplers import tune_ladder
from src.certificate import make_phi_family, certificate_grid, certificate_importance
from src.plotting import metric_grid

DEV = "cuda"
RESULTS = os.path.abspath(os.path.join("..", "results", EXPERIMENT))
FIGURES = os.path.abspath(os.path.join("..", "figures", EXPERIMENT))
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
{extra}
cfg = exp.cfg
print(f"experiment={{cfg.name}}  d={{cfg.d}}  N={{cfg.n_particles}}  T={{cfg.T}}  dt0={{cfg.dt}}")
print(f"beta={{cfg.beta}}  eps={{cfg.eps}}  lambda={{cfg.lam}}  seeds={{cfg.seeds}}")
print(hardware_manifest())'''


CELL_LADDER = '''# PT: geometric ladder beta_k = beta * r^(k-1); K tuned so the post-burn-in
# swap acceptance lands in [0.2, 0.4]
gen = torch.Generator(device=DEV); gen.manual_seed(0)
x0_pilot = exp.init_fn(min(512, cfg.n_particles), gen)
pt_betas, ladder_info = tune_ladder(exp.pot, x0_pilot, cfg.dt, exp.box,
                                    C.BETA, exp.pt_beta_min, pilot_steps=20_000)
print(f"PT ladder: K={ladder_info['K']}  r={ladder_info['r']:.4f}  "
      f"beta_K={pt_betas[-1].item():.4f}  swap acceptance={ladder_info['swap_acceptance']:.3f}"
      f"  band_attained={ladder_info['band_attained']}")'''


CELL_REFERENCE = '''# frozen reference sample (size N), frozen sliced-W2 projections, frozen MMD
# bandwidth (median heuristic on the reference); bias floors from 20
# independent reference pairs. EMC convention: exp(H(p_hat))/K for uniform
# p*, 1 - EJS(p_hat, p*) otherwise -- near 1 is better in both cases.
metrics_fn, floors, aux = make_metrics(exp, cfg.n_particles)
emc_target = exp.emc_target
print("p_star:", np.round(exp.p_star.cpu().numpy(), 6),
      " uniform:", exp.uniform_target)
print("MMD bandwidth:", round(aux["bandwidth"], 4))
for k, v in floors.items():
    print(f"  floor {k:>12s}: {v['mean']:.5f} +- {v['std']:.5f}")'''


def cell_dt_production(main_metrics: str) -> str:
    return f'''# dt rule: largest dyadic dt at which every PI-TARGETING method's terminal
# metrics agree with dt/2 (5% / floor-band / 4-sigma noise guards); FLA and
# raw CP have invariant laws != pi and are recorded but do not gate.
# Production: all 5 seeds batched into one (5N)-particle ensemble per method.
MAIN_METRICS = {main_metrics}

def run_terminal_all(dt_):
    n_ = int(round(cfg.T / dt_))
    factory = make_sampler_factory(exp, dt_, pt_betas, score_kwargs=CHOSEN_QUAD)
    out = {{}}
    for m in C.METHODS:
        rows_, _ = run_one(m, 0, factory, n_, n_, dt_, metrics_fn, exp.pot, quiet=True)
        out[m] = {{k: rows_[-1][k] for k in MAIN_METRICS}}
    print(f"  refine_dt: finished pass at dt={{dt_}}", flush=True)
    return out

dt_final, dt_table = refine_dt(run_terminal_all, cfg.dt, floors,
                               exclude=("FLA", "CP"))
print("chosen dt:", dt_final)
for row in dt_table:
    print(row)

n_steps = int(round(cfg.T / dt_final))
steps_per_ck = max(1, n_steps // C.N_CHECKPOINTS)
# dense-early checkpoint schedule: the nonlocal transient lives in the
# first ~5% of the run; 40 dense + 48 sparse points, identical across
# methods (measurement cadence only -- no protocol change)
from src.runner import checkpoint_schedule
ck_steps = checkpoint_schedule(n_steps)
bfactory = make_batched_factory(exp, dt_final, pt_betas, cfg.seeds,
                                score_kwargs=CHOSEN_QUAD)
t0 = time.time()
rows, method_info = run_experiment_batched(C.METHODS, cfg.seeds, bfactory,
                                           n_steps, steps_per_ck, dt_final,
                                           metrics_fn, exp.pot,
                                           cfg.n_particles,
                                           checkpoint_steps=ck_steps)
print(f"production total: {{time.time()-t0:.0f}}s")
assert max(r["nonfinite_frac"] for r in rows) == 0.0
print("nonfinite fraction: identically zero")'''


CELL_FIGURES = '''fig = metric_grid(rows, os.path.join(FIGURES, EXPERIMENT + "_metrics"),
                  metrics=("W2", "MMD", "EMC"), floors=floors,
                  emc_target=emc_target)
print("saved:", os.path.join(FIGURES, EXPERIMENT + "_metrics") + ".{png,pdf}")'''


def cell_csv(extra_manifest: str = "") -> str:
    return f'''ts_path = os.path.join(RESULTS, "metrics_timeseries.csv")
write_timeseries_csv(rows, ts_path)
summary_metrics = MAIN_METRICS + ["nonfinite_frac"]
summary = write_summary_csv(rows, C.METHODS, cfg.seeds, summary_metrics,
                            method_info, floors, os.path.join(RESULTS, "summary.csv"))

manifest = dict(
    experiment=EXPERIMENT,
    config=dict(d=cfg.d, N=cfg.n_particles, T=cfg.T, dt0=cfg.dt, dt=dt_final,
                beta=cfg.beta, eps=cfg.eps, lam=cfg.lam, seeds=list(cfg.seeds),
                n_checkpoints=C.N_CHECKPOINTS, warmup_steps=C.N_WARMUP_STEPS,
                batched_seeds=True),
    quadrature=dict(chosen=CHOSEN_QUAD, table=quad_table),
    dt_refinement=[{{k: (str(v) if isinstance(v, tuple) else v) for k, v in row.items()}}
                   for row in dt_table],
    pt_ladder={{k: v for k, v in ladder_info.items()}},
    certificate=cert_report,
    bias_floors=floors,
    barrier_verification=barrier_report,
    method_info={{m: {{k: v for k, v in mi.items() if isinstance(v, (int, float))}}
                 for m, mi in method_info.items()}},
    hardware=hardware_manifest(),
    {extra_manifest}
)
write_manifest(os.path.join(RESULTS, "manifest.json"), **manifest)
print("wrote", ts_path)
from IPython.display import display
display(pd.read_csv(os.path.join(RESULTS, "summary.csv")).round(5))'''


# ======================================================================
# E1 double well
# ======================================================================
def build_e1_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E1 — 1D double well

**Target.** $\pi(x)\propto e^{-\beta V(x)}$ at $\beta=8$ ($\varepsilon=1/\beta$) with $V(x)=(x^2-1)^2$: minima $\pm1$, saddle $0$, $\beta\Delta V=8$, Kramers time $\tau=\tfrac{2\pi}{\sqrt{32}}e^{8}\approx3.3\times10^3$ — local samplers started in the left well essentially never equilibrate within $T=100$. Seven methods (ULA, MALA, FLA, BAOAB, PT, Raw-CP, LSC-CP) share one tamed drift map, one $\Delta t$, one metric cadence and per-seed initial conditions $x_0\sim\mathcal N(-1,0.05^2)$."""),
        code(cell_setup("double_well", "build_e1", "exp = build_e1(device=DEV)")),
        code('''# model asserts + barrier verification (committed arrival in the right-well
# core x > 0.7; censored-exponential MLE vs the Kramers estimate)
V = lambda x: (x**2 - 1.0)**2
assert V(1.0) == 0.0 and V(-1.0) == 0.0 and V(0.0) == 1.0
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt),
                                   C.EPS, g)
barrier_report["kramers_tau"] = exp.kramers_tau
print(f"ULA committed MFPT {barrier_report['mfpt_estimate']:.0f} vs Kramers "
      f"{exp.kramers_tau:.0f} ({barrier_report['n_exits']} exits of "
      f"{barrier_report['n_particles']})")'''),
        md(r"""## Jump law and Lévy score

Two-atom symmetric shell: $r = \pm2 + \rho\,u$, $\rho\sim\mathrm{Unif}(-h,h)$, $h=0.2$, $w=(\tfrac12,\tfrac12)$, $\lambda=1$ — a $\pm2$ jump maps minimum to minimum. The stationary correction
$$S_{\nu,\beta}(x) = -\lambda\!\int\!\nu(dr)\,r\!\int_0^1\! e^{-\beta[V(x-\theta r)-V(x)]}d\theta$$
makes $\pi$ invariant for the jump diffusion *exactly at generator level, for any $\nu$*. It is computed with Gauss–Legendre probability weights on both inner integrals (the quadrature measure equals the sampling $\nu$) and **log-space accumulation**: the per-direction integrals span hundreds of orders of magnitude at $\beta=8$, so we assemble $\log I$ by log-sum-exp, extract the max exponent $M(x)$, form the $O(1)$ direction vector $v(x)$, and return $S = -\lambda e^{\min(M,600)}v$ — the drift is tamed, so only the direction matters when $\|S\|$ is astronomical. The weak stationarity residual $\mathcal R(\varphi)$ (drift term assembled in log space; domain one jump length beyond the support) certifies the correction; a deliberately tight box fails it."""),
        code('''DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(1, [0.0], 1.0, DEV)

def cert_e1(q_theta, q_rho, lo=-5.2, hi=5.2):
    score = exp.make_score(q_theta=q_theta, q_rho=q_rho)
    shifts, logw = exp.law.quadrature_shifts(64)   # fine continuous-nu J side
    return certificate_grid(exp.pot, score, shifts, logw, cfg.lam, cfg.beta,
                            phis, [lo], [hi], n_panels=120, nodes_per_panel=8)

cert_report = cert_e1(**DEFAULT_QUAD)
print(f"max R = {cert_report['max_residual']:.3e}")
assert cert_report["max_residual"] < 1e-6
tight = cert_e1(**DEFAULT_QUAD, lo=-1.3, hi=1.3)
print(f"deliberately TIGHT box: max R = {tight['max_residual']:.3e}")'''),
        code(CELL_LADDER),
        code(CELL_REFERENCE),
        code('''# quadrature refinement: smallest (Q_theta, Q_rho) with R < 1e-6 and
# terminal LSC-CP metrics converged against the finest setting
def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "TV_density", "MMD", "EMC")}

settings = [dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e1(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
display(pd.DataFrame(quad_table).round(6))'''),
        code(cell_dt_production('["W2", "TV", "TV_density", "MMD", "EMC"]')),
        code(CELL_FIGURES),
        code('''# terminal-sample CDF of every method vs the true CDF (single plot;
# all 5 seed blocks pooled -> 20k points per method)
from src.plotting import cdf_comparison
ref = exp.extras["ref"]
samples = {m: method_info[m]["final_positions_all"].reshape(-1).cpu().numpy()
           for m in C.METHODS}
cdf_fig = cdf_comparison(samples, ref.x.cpu().numpy(), ref.cdf.cpu().numpy(),
                         os.path.join(FIGURES, EXPERIMENT + "_cdf"))
print("saved:", os.path.join(FIGURES, EXPERIMENT + "_cdf") + ".{png,pdf}")'''),
        code(cell_csv()),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E2 MoG40
# ======================================================================
def build_e2_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E2 — MoG40 (2D)

**Target.** $V(x) = -\tfrac1\beta\log\sum_{k=1}^{40} e^{-\|x-\mu_k\|^2/2}$, so $\pi\propto e^{-\beta V}$ is an equal-weight mixture of $\mathcal N(\mu_k, I_2)$ (the $1/\beta$ prefactor is what makes the barriers right: $\beta\Delta V = d^2/8-\log 2$ between modes at distance $d$). Modes $\mu_k\sim\mathrm{Unif}([-40,40]^2)$, frozen from `default_rng(0)`. All particles start at $\mu_0+0.5\,\xi$; partition = nearest-mode Voronoi, $p^\star_k = 1/40$."""),
        code(cell_setup("mog40", "build_e2", "exp = build_e2(device=DEV)")),
        code('''np.savetxt(os.path.join(RESULTS, "modes.csv"), exp.pot.mu.cpu().numpy(),
           delimiter=",", header="mu_x,mu_y", comments="")
dists = torch.cdist(exp.pot.mu, exp.pot.mu); dists.fill_diagonal_(float("inf"))
nn = dists.min(dim=1).values.cpu().numpy()
print(f"NN distances: min {nn.min():.2f} median {np.median(nn):.2f} max {nn.max():.2f}"
      "  -> jump radii Unif[4, 15] chosen from this histogram alone")
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), C.EPS, g)
barrier_report["kramers_tau_mode0"] = exp.kramers_tau
print(f"ULA committed MFPT {barrier_report['mfpt_estimate']:.0f} vs Kramers "
      f"{exp.kramers_tau:.0f}")'''),
        md(r"""## Jump law and Lévy score (closed form)

Deliberately generic annulus law — $r=\rho u_\phi$, $\rho\sim\mathrm{Unif}[4,15]$, $\phi\sim\mathrm{Unif}[0,2\pi)$ — **neither PT nor LSC-CP receives mode locations**. For the Gaussian mixture the $\theta$ and $\rho$ integrals of the score are analytic: with $m_{k\ell}=u_\ell\cdot(x-\mu_k)$,
$$S(x) = -\frac{\lambda}{M_\phi(b-a)}\sum_\ell u_\ell \sum_k e^{\log\omega_k + m_{k\ell}^2/2}\,\sqrt{\tfrac\pi2}\,\mathcal B(m_{k\ell}),$$
with $\mathcal B(m) = F(b\!-\!m)-F(a\!-\!m)+(b\!-\!a)\,\mathrm{erf}(m/\sqrt2)>0$ and $F(z)=z\,\mathrm{erf}(z/\sqrt2)+\sqrt{2/\pi}e^{-z^2/2}$, evaluated by outer-regime branches in which the $O(m)$ parts cancel *analytically* (the naive form has 100% error at $m=30$; the branched form is validated against 3000-digit mpmath and a brute-force 3-D quadrature at $10^{-8}$). Only the periodic $\phi$-trapezoid ($M_\phi$ directions) is numerical — **zero potential evaluations**."""),
        code('''DEFAULT_QUAD = dict(m_phi=C.M_PHI)
phis = make_phi_family(2, [0.0, 0.0], 30.0, DEV)

def cert_e2(m_phi):
    score = exp.make_score(m_phi=m_phi)
    shifts, logw = exp.law.quadrature_shifts(16, 64)
    return certificate_grid(exp.pot, score, shifts, logw, cfg.lam, cfg.beta,
                            phis, [-60.0, -60.0], [60.0, 60.0],
                            n_panels=120, nodes_per_panel=6, chunk=8192)

cert_report = cert_e2(**DEFAULT_QUAD)
print(f"max R = {cert_report['max_residual']:.3e}")
assert cert_report["max_residual"] < 1e-6'''),
        code(CELL_LADDER),
        code(CELL_REFERENCE + '''
assert aux["bandwidth"] > 3.0   # bandwidth reflects mode spacing, not width'''),
        code('''def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "MMD", "EMC")}

settings = [dict(m_phi=m) for m in (16, 32, 64)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e2(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
display(pd.DataFrame(quad_table).round(6))'''),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC"]')),
        code(CELL_FIGURES),
        code('''# terminal exact-W2 spot check (Hungarian, 500-point subsample, 2D only)
gen_h = torch.Generator(device=DEV); gen_h.manual_seed(202)
ref_sub = exp.ref_sample(2500, gen_h)
from src.metrics import hungarian_w2
hungarian = {m: hungarian_w2(method_info[m]["final_positions_seed0"], ref_sub, m=500)
             for m in C.METHODS}
print("Hungarian W2:", {k: round(v, 3) for k, v in hungarian.items()})
''' + cell_csv("hungarian_w2_terminal=hungarian,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E3 4-well modified Mueller-Brown 10D
# ======================================================================
CELL_E3_ASSERTS = '''from src.potentials import MB4_CRITICAL, mb4_2d, mb4_2d_grad, newton_refine
for key, (z_tab, V_tab) in MB4_CRITICAL.items():
    z = newton_refine(mb4_2d_grad, torch.tensor(z_tab, device=DEV))
    Vv = mb4_2d(z.unsqueeze(0))[0].item()
    assert abs(z[0].item()-z_tab[0]) < 5e-5 and abs(z[1].item()-z_tab[1]) < 5e-5, key
    assert abs(Vv - V_tab) < 5e-4, (key, Vv)
    print(f"{key}: ({z[0].item():+.4f}, {z[1].item():+.4f})  V = {Vv:.4f}")
print("p_star (W1..W4):", np.round(exp.p_star.cpu().numpy(), 4),
      " | S12 barrier beta*b = %.1f | island walls beta*b ~ %.0f" %
      (8*(MB4_CRITICAL["S12"][1] - MB4_CRITICAL["W1"][1]),
       8*exp.extras["barrier_W3"]))

# committed first passage from island W3 (expect ZERO local exits)
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), C.EPS, g)
barrier_report["kramers_tau_plateau"] = exp.kramers_tau
print(f"ULA committed exits from W3: {barrier_report['n_exits']} of "
      f"{barrier_report['n_particles']} in T={cfg.T} "
      f"(plateau-wall estimate: beta*b = {8*exp.extras['barrier_W3']:.0f})")'''

CELL_E3_CERT = '''from src.potentials import MB4Latent2D
from src.jumps import ShellJumpLaw
from src.score import ShellScore
potr = MB4Latent2D()
dz = exp.extras["atoms_z"][:, :2]
h_z = exp.extras["h"] * dz.norm(dim=1) / exp.law.atoms.norm(dim=1)
law_r = ShellJumpLaw(dz, exp.law.weights.clone(), h_z)
print("atoms (latent):", np.round(dz.cpu().numpy(), 3).tolist())
print("h =", round(exp.extras["h"], 4), " drift cap (CP pair) =",
      round(exp.cp_drift_cap, 4))
DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(2, [0.1, 0.5], 0.9, DEV)

def cert_e3(q_theta, q_rho):
    score = ShellScore(potr, law_r, cfg.lam, cfg.beta, q_theta, q_rho)
    shifts, logw = law_r.quadrature_shifts(64)
    return certificate_grid(potr, score, shifts, logw, cfg.lam, cfg.beta, phis,
                            [-4.2, -3.9], [4.4, 4.9],
                            n_panels=170, nodes_per_panel=8, chunk=4096)

cert_report = cert_e3(**DEFAULT_QUAD)
print(f"max R = {cert_report['max_residual']:.3e}")
assert cert_report["max_residual"] < 1e-6'''

CELL_E3_QUAD = '''def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "MMD", "EMC", "W2_10d")}

settings = [dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e3(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
display(pd.DataFrame(quad_table).round(6))'''

CELL_E3_PISTART = '''# pi-start hold test: initialise at the reference and measure any stationary
# drift of the discretised LSC-CP chain (mid-asymmetry honesty check)
g_h = torch.Generator(device=DEV); g_h.manual_seed(777)
x_pi = exp.ref_sample(4000, g_h)
from src.metrics import occupancy as _occ
bf = make_batched_factory(exp, dt_final, pt_betas, (0,), n_particles=4000,
                          score_kwargs=CHOSEN_QUAD)
s_h = bf("LSC-CP")
s_h.x = x_pi.clone()
p0 = _occ(exp.labels_fn(s_h.positions()), 4)
for _i in range(int(round(100.0 / dt_final))):
    s_h.step()
p1 = _occ(exp.labels_fn(s_h.positions()), 4)
hold_tv = 0.5 * float((p1 - exp.p_star).abs().sum())
pi_start_hold = {"init": [round(float(v), 4) for v in p0],
                 "after_T100": [round(float(v), 4) for v in p1],
                 "TV_vs_pstar": round(hold_tv, 4)}
print("pi-start hold:", pi_start_hold)'''


def build_e3_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E3 — 4-well modified Müller–Brown (10D)

**Target.** The equal-amplitude 4-well Müller–Brown variant (all four Gaussian amplitudes $-200\times0.05$; the true MB's repulsive hump replaced by a fourth well at $(-0.8,-0.5)$ — `archive/mueller.py` precedent), embedded in 10D exactly as before: $U(z) = V_4(z_1,z_2) + \|z_{3:10}\|^2/(2\cdot0.4^2)$, $x = zB^\top$. Unlike the true MB — whose depth gap equals its barrier scale, so *no* temperature is simultaneously multimodal and metastable — this target at $\beta=8$ has masses $p^\star \approx (0.617,\ 0.338,\ 0.027,\ 0.018)$ with a $\beta b = 16.5$ saddle between the two major wells and **plateau-level walls ($\beta b \approx 80$) around the two minor island wells** — locals provably never move, PT needs a deep ladder, and only nonlocal jumps can populate the islands at the right mass. Runs start on island W3 (2.7% mass). Metrics in latent 2D $z_{1:2}$ (full-10D sliced $W_2$ in the CSV)."""),
        code(cell_setup("mb4well_10d", "build_e3",
                        'exp = build_e3(device=DEV, basin_cache=os.path.join(RESULTS, "basin_map.npz"))')),
        code(CELL_E3_ASSERTS),
        md(r"""## Jump law and Lévy score

Complete graph over the four latent minima (12 directed atoms $r_a = (\Delta z, 0_8)B^\top$, $w_a = 1/12$, shell $h = 0.1\min\|r_a\|$, $\lambda = 1$), with the CP pair's drift step capped at $2h$ — all three measured E3 design rules apply: every well here carries $\geq 1.8\%$ mass (no negligible relay targets), weights stay $O(1)$ (mass-ratio skew measurably backfires), and the step cap keeps returned landers inside the score tube. Score: generic shell with log-space accumulation; certificate on the exact latent-2D reduction. *Known regime note:* inter-well asymmetries $\beta\Delta V \in [0.6, 3.3]$ put this landscape partly in the measured mid-asymmetry zone where fixed-step tamed integration of the detailed-balance return flux is imperfect; the $\pi$-start hold test below quantifies the resulting stationary offset honestly."""),
        code(CELL_E3_CERT),
        code(CELL_LADDER),
        code(CELL_REFERENCE),
        code(CELL_E3_QUAD),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC", "W2_10d"]')),
        code(CELL_E3_PISTART),
        code(CELL_FIGURES),
        code(cell_csv("pi_start_hold=pi_start_hold,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E4 coupled phi4
# ======================================================================
def build_e4_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E4 — Coupled $\phi^4$ chain (24D)

**Target.** $q_i\in\mathbb R^2$, $N_s=12$ periodic sites, $\delta=1/N_s$, $\kappa=2.5$:
$$V(q) = \frac{\kappa}{2\delta}\sum_i\|q_{i+1}-q_i\|^2 + \delta\sum_i W(q_i),\qquad W(x,y) = (x^2-1)^2+(y^2-1)^2-0.0125\,xy+0.0075\,x+0.015\,y.$$
Four coherent phases $(\pm1,\pm1)$; for homogeneous fields $V(\mathbf 1\otimes v)=W(v)$, so the coherent barrier equals the barrier of $W$ ($\beta\cdot\min$ barrier $=7.8$), and the kink-pair cost $5.96\gg1$ makes **the coherent flip the minimum-energy path**. The tilt terms are chosen so the inter-phase asymmetry is $\beta\Delta W = 0.44 \le 0.5$: inside the regime where the tamed fixed-step integrator realises the correction's detailed-balance return flux (at $\beta\Delta W\approx1.8$ a measured, $\Delta t$-independent occupancy offset of order $10\%$ appears — a deliberate benchmark-design choice, recorded here). Phase masses remain distinguishably non-uniform, $p^\star \approx (0.323, 0.212, 0.238, 0.227)$. Init at the $--$ phase; partition = basin of $W$ at $\bar q$. **Reference: exact $\pi$ samples** by SNIS resampling from the harmonic (Laplace) mixture proposal; a long PT chain cross-checks the phase masses."""),
        code(cell_setup("coupled_phi4", "build_e4",
                        'exp = build_e4(device=DEV, basin_cache=os.path.join(RESULTS, "basin_map.npz"))')),
        code('''from src.potentials import (PHI4_MINIMA, PHI4_ESCAPE_BARRIERS,
                            PHI4_LAPLACE_MASSES, phi4_W, phi4_W_grad, newton_refine)
V2 = exp.extras["minima_2d"]
for i, ph in enumerate(exp.extras["phases"]):
    v_tab, W_tab = PHI4_MINIMA[ph]
    W = phi4_W(V2[i].unsqueeze(0))[0].item()
    assert abs(V2[i][0].item()-v_tab[0]) < 5e-5 and abs(W - W_tab) < 5e-4
    # p_star is the EXACT (SNIS) mass; the Laplace table should agree to
    # the anharmonic correction scale (~1e-2)
    assert abs(exp.p_star[i].item() - PHI4_LAPLACE_MASSES[ph]) < 2e-2
print("minima / Laplace masses verified; kink pair cost",
      round(2*exp.pot.kink_energy(), 2), ">> 1.0 coherent barrier")
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), C.EPS, g)
barrier_report["kramers_tau_langer"] = exp.kramers_tau
print(f"ULA committed MFPT {barrier_report['mfpt_estimate']:.0f} "
      f"({barrier_report['n_exits']} exits) vs 24D Langer {exp.kramers_tau:.0f}")'''),
        md(r"""## Jump law and Lévy score (moment-exact)

Homogeneous phase shifts on the complete graph over the 4 minima (12 directed atoms $r_a=\mathbf 1_{N_s}\otimes(v_j-v_i)$, $w_a=1/12$, shell $h=0.1\min\|r_a\|$). The gradient energy is exactly invariant under homogeneous shifts, so $V(q-r)-V(q)=\delta\sum_i[W(q_i-d)-W(q_i)]$ is a fixed polynomial in $d$ whose coefficients are the per-particle moments $\sum x_i, \sum x_i^2, \sum x_i^3$ (and $y$ analogues): moments once per step in $O(N_s)$, then every quadrature energy delta is $O(1)$ — **no lattice sweeps** (validated to $10^{-13}$). In 24D the certificate uses the shifted-form identity with importance sampling from the Laplace mixture (equivalent to the deployed score; the $M$ cap never fires on the sampled region)."""),
        code('''from src.jumps import gauss_legendre_01
from src.certificate import TanhRidgeProduct
DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(24, exp.extras["means24"][0].tolist(), 1.5, DEV, n_phi=4)
# jump-ALIGNED test functions: in 24D random ridge directions have
# a.r_hat ~ 1/sqrt(24) and are blind to variation along the coherent path,
# which is exactly where the theta-quadrature acts; add ridges along the
# first atoms at three sharpness scales.
for a_idx, sc in ((0, 0.5), (0, 1.0), (2, 0.5)):
    r0 = exp.law.atoms[a_idx]
    rhat = (r0 / r0.norm()).unsqueeze(0)
    mid = exp.extras["means24"][0] + 0.5 * r0
    phis.append(TanhRidgeProduct(rhat, (rhat @ mid.unsqueeze(1)).reshape(1),
                                 torch.tensor([sc], device=DEV)))

def cert_e4(q_theta, q_rho):
    theta, w_theta = gauss_legendre_01(q_theta, DEV)
    shifts, logw = exp.law.quadrature_shifts(q_rho)
    shifts_j, logw_j = exp.law.quadrature_shifts(64)
    return certificate_importance(exp.pot, shifts, logw, theta, w_theta,
                                  cfg.lam, cfg.beta, phis, exp.extras["laplace"],
                                  n_samples=200_000,
                                  nu_shifts_jump=shifts_j, nu_logw_jump=logw_j)

cert_report = cert_e4(**DEFAULT_QUAD)
print(f"max R at default orders = {cert_report['max_residual']:.3e}")
print("NOTE: the jump-aligned phis expose the theta-quadrature defect along "
      "the coherent path at Q_theta=16; the refinement gate below selects "
      "the smallest orders with R < 1e-6 (and re-asserts).")
gm = torch.Generator(device=DEV); gm.manual_seed(11)
Mv, _ = exp.make_score(**DEFAULT_QUAD).log_parts(exp.extras["laplace"].sample(100_000, gm))
print(f"max log score magnitude on support: {Mv.max().item():.1f} << 600")
cert_report["max_log_magnitude_on_support"] = float(Mv.max().item())'''),
        code(CELL_LADDER),
        code(CELL_REFERENCE + '''

# SNIS proposal quality + PT cross-check of the exact phase masses
g_ess = torch.Generator(device=DEV); g_ess.manual_seed(555)
ess = exp.extras["laplace"].snis_ess_fraction(exp.pot, C.BETA, g_ess)
print(f"SNIS proposal ESS fraction: {ess:.3f}")
gen_x = torch.Generator(device=DEV); gen_x.manual_seed(4242)
from src.samplers import ParallelTempering
from src.metrics import occupancy
pt_x = ParallelTempering(exp.pot, exp.init_fn(1000, gen_x), cfg.dt, pt_betas,
                         gen_x, exp.box)
for _ in range(int(round(300.0 / cfg.dt))):
    pt_x.step()
p_pt = occupancy(exp.labels_fn(pt_x.positions()), 4).cpu().numpy()
print("long-PT phase masses:", np.round(p_pt, 3),
      " vs exact p* (SNIS):", np.round(exp.p_star.cpu().numpy(), 3))
pt_crosscheck = p_pt.tolist()'''),
        code('''def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "MMD", "EMC")}

settings = [dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e4(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
if CHOSEN_QUAD != DEFAULT_QUAD:
    cert_report = cert_e4(**CHOSEN_QUAD)
    print(f"certificate at chosen orders: max R = {cert_report['max_residual']:.3e}")
    assert cert_report["max_residual"] < 1e-6
display(pd.DataFrame(quad_table).round(6))'''),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC"]')),
        code(CELL_FIGURES),
        code(cell_csv("pt_phase_mass_crosscheck=pt_crosscheck,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for name, builder in [("01_double_well", build_e1_nb),
                          ("02_mog40", build_e2_nb),
                          ("03_mb4well_10d", build_e3_nb),
                          ("04_coupled_phi4", build_e4_nb)]:
        nb = builder()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        path = os.path.join(here, f"{name}.ipynb")
        with open(path, "w") as f:
            nbf.write(nb, f)
        print("wrote", path)
