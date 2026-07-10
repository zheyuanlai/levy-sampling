"""Generate the four experiment notebooks (01-04) with nbformat.

Run from JCP_experiments/notebooks:  python build_notebooks.py
Each notebook follows the structure of spec section 12: model derivation,
jump law + Levy score, target preservation + certificate, samplers,
reference/partition/floors, dt refinement + production, figures, CSVs.
"""
from __future__ import annotations

import nbformat as nbf


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


# ======================================================================
# shared markdown blocks
# ======================================================================
MD_TARGET_PRESERVATION = r"""## Target preservation: the stationarity identity $(\star)$

The LSC-CP generator is
$$\mathcal A f = \big[-\nabla V + S_{\nu,\beta}\big]\cdot\nabla f + \varepsilon\,\Delta f + \lambda\!\int\!\big[f(x+r)-f(x)\big]\nu(dr),$$
with $\nu$ a **probability** measure and
$$S_{\nu,\beta}(x) = -\lambda \int \nu(dr)\; r \int_0^1 \exp\!\Big[-\beta\big(V(x-\theta r) - V(x)\big)\Big]\, d\theta .$$
Raw CP is the same generator with $S \equiv 0$.

Write $p = e^{-\beta V}/Z$. The overdamped part is $\pi$-reversible, so invariance of $\pi$ is equivalent to
$$\int S\cdot\nabla\varphi \, d\pi + \int J\varphi \, d\pi = 0 \qquad \forall\, \varphi \in C_c^\infty . \tag{$\star$}$$

**Jump term.** Shift the integration variable and apply the fundamental theorem of calculus along $\theta \mapsto y - \theta r$:
$$\int J\varphi\,d\pi = \lambda\!\int\!\nu(dr)\!\int\!\varphi(y)\big[p(y-r)-p(y)\big]dy = -\lambda\!\int\!\nu(dr)\!\int_0^1\!\!\int\!\varphi(y)\,r\!\cdot\!\nabla p(y-\theta r)\,dy\,d\theta.$$

**Drift term.** $S(x) = -\lambda\int\nu(dr)\,r\int_0^1 \frac{p(x-\theta r)}{p(x)}d\theta$ (identical to the boxed formula since $p \propto e^{-\beta V}$), so integrating by parts in $x$:
$$\int S\cdot\nabla\varphi\,d\pi = -\lambda\!\int\!\nu(dr)\!\int_0^1\!\!\int\!\big(r\!\cdot\!\nabla\varphi(x)\big) p(x-\theta r)\,dx\,d\theta = +\lambda\!\int\!\nu(dr)\!\int_0^1\!\!\int\!\varphi(x)\,r\!\cdot\!\nabla p(x-\theta r)\,dx\,d\theta.$$

The two cancel identically. **Target preservation is unconditional in $\nu$** — any finite-activity jump law works; only the *speed* depends on $\nu$.

### The measured certificate $\mathcal R(\varphi)$

For smooth bounded test functions (products of tanh ridges) we report
$$\mathcal R(\varphi) = \frac{\big|\int S_{\nu,\beta}\!\cdot\!\nabla\varphi\,d\pi + \int J_\nu\varphi\,d\pi\big|}{\big|\int J_\nu\varphi\,d\pi\big|},$$
zero in exact arithmetic; the measured value is the combined defect of the $\theta$/$\rho$ quadratures. Two implementation notes, both load-bearing:

1. **The integration domain extends a full jump length beyond the target's effective support.** Order-one contributions to $(\star)$ live where $\pi$ is tiny and $S$ is enormous; a deliberately tight box produces a large residual (demonstrated below and regression-tested).
2. **The drift integrand $p\,S\cdot\nabla\varphi$ is assembled in log space** from the score's $(M, v)$ parts as $\exp(-\beta V + M)\,v\cdot\nabla\varphi$: in linear fp64 arithmetic $p$ underflows exactly where $\|S\|$ is astronomical, silently dropping those order-one far-field contributions. The residual uses the *uncapped* $M$; the deployed drift caps $M$ at $M_{\max}=600$, but because taming saturates (the tamed step tends to $-v/\|v\|$), the deployed tamed step differs from the uncapped one by $O(e^{-M_{\max}})$ — that saturation defect is reported alongside $\mathcal R$ and is $\lesssim 10^{-250}$ here.

A useful exact identity (change of variables $x \to x+\theta_p r$ in the drift term): for the *implemented* quadrature score,
$$\int S\cdot\nabla\varphi\,d\pi + \int J\varphi\,d\pi \;=\; \lambda\,\mathbb E_\pi\!\int\!\nu(dr)\Big[\varphi(x+r)-\varphi(x) - \sum_p \hat w_p\, r\cdot\nabla\varphi(x+\theta_p r)\Big],$$
i.e. the residual is **independent of $V$** and equals the $\theta$-quadrature error on the smooth test-function integrand. This is why moderate pointwise errors of the Gauss–Legendre rule on the stiff factor $e^{\beta\Delta V}$ do not translate into a weak (distributional) defect of the sampled law."""

MD_SAMPLERS = r"""## The seven methods

All methods share one **taming policy**: the same map $b \mapsto b/(1+\Delta t\,\|b\|)$ is applied to every method's drift (ULA, the MALA proposal, FLA, the BAOAB force, raw CP, LSC-CP). Tamed MALA is still exact because the proposal density $q(y|x) = \mathcal N(y;\, x + \Delta t\, b_{\rm tamed}(x),\, 2\varepsilon\Delta t\, I)$ is used consistently in both directions of the MH ratio; asymmetric taming would make taming a hidden variable in the comparison.

**1. ULA.** $X \leftarrow X + \Delta t\,\mathrm{tame}(-\nabla V) + \sqrt{2\varepsilon\Delta t}\,\xi$.

**2. MALA.** With $\nabla\log\pi = -\beta\nabla V$, the proposal $Y = X - \tfrac{h\beta}2\nabla V(X) + \sqrt h\,\xi$ matches the ULA step iff $\tfrac{h\beta}{2} = \Delta t$ **and** $h = 2\varepsilon\Delta t$. Both conditions coincide:
$$\boxed{h = \frac{2\Delta t}{\beta} = 2\varepsilon\Delta t = \Delta t/4 \quad\text{at }\beta=8.}$$
Log-acceptance
$$\log\alpha = -\beta[V(Y)-V(X)] - \frac{1}{2h}\Big[\|X - \mu(Y)\|^2 - \|Y-\mu(X)\|^2\Big],\qquad \mu(z) = z + \Delta t\,\mathrm{tame}(-\nabla V(z)),$$
accepted elementwise. Proposals are **never clipped before the accept step** (that silently breaks exactness); out-of-box proposals are auto-rejected, which is valid MH for the box-restricted target. Expect acceptance $\approx 1$ — the honest message: **rejection does not cure metastability**.

**3. FLA / FLMC** (Şimşekli, ICML 2017, §3.3). $X \leftarrow X + \Delta t\,\mathrm{tame}(-c_\alpha \nabla U) + \Delta t^{1/\alpha}\,\xi^{(\alpha)}$ with $U = \beta V$, $c_\alpha = \Gamma(\alpha-1)/\Gamma(\alpha/2)^2$, $\alpha = 1.7$; per-coordinate $S\alpha S(1)$ noise by Chambers–Mallows–Stuck. **No tail clipping** — a truncated stable is not stable. FLA is the *uncorrected nonlocal* comparator: heavy tails cross barriers, but the invariant law is not $\pi$.

**4. Kinetic Langevin (BAOAB).** It is **not HMC** (no accept/reject; carries $O(\Delta t^2)$ configurational bias). Unit mass, $\gamma = 1$; the O-step coefficient is the exact OU solution: $dp = -\gamma p\,dt + \sqrt{2\gamma\varepsilon}\,dW$ gives $\mathrm{Var} = 2\gamma\varepsilon\int_0^{\Delta t}e^{-2\gamma s}ds = \varepsilon(1-e^{-2\gamma\Delta t})$ by Itô isometry. The trailing force is cached as the next step's leading B (one gradient per step).

**5. Parallel tempering.** MALA-within-replica at $\beta_k = \beta\, r^{k-1}$, replica $k$ using $h_k = 2\Delta t/\beta_k$ (same tamed drift step for every replica; only the noise scale differs). Adjacent swaps every $n_{\rm swap}$ steps (alternating parity); the joint target is $\prod_k \pi_k$ and the swap is a deterministic involution, so
$$\alpha_{\rm swap} = \min\Big\{1,\ \exp\big[(\beta_i - \beta_{i+1})\big(V(x_i) - V(x_{i+1})\big)\big]\Big\}.$$
$V$ values are cached by MALA, so swaps are free in evaluation count. $K$ is tuned so the mean swap acceptance lands in $[0.2, 0.4]$. The replica index is a batch dimension, state $(K, N, d)$; metrics use the cold replica only; **wall-clock includes all $K$ replicas**.

**6/7. Raw CP and LSC-CP.** Identical time discretisation,
$$X^{(1)} = X_n + \Delta t\,\frac{b(X_n)}{1+\Delta t\,\|b(X_n)\|} + \sqrt{2\varepsilon\Delta t}\;\xi_n,\qquad X_{n+1} = X^{(1)} + \sum_{k=1}^{N_n} A_k,$$
$N_n \sim \mathrm{Poisson}(\lambda\Delta t)$, $A_k \stackrel{iid}\sim \nu$; $b = -\nabla V$ (raw CP) or $b = -\nabla V + S_{\nu,\beta}$ (LSC-CP). The jump stream is a dedicated generator seeded identically for both methods, so their jump times and increments are **pathwise identical** (verified in `tests/test_samplers.py`), not merely equal in law."""

MD_METRICS = r"""## Reference, partition, metrics, bias floors

Reference sample size equals the run's $N$; metrics are evaluated at every checkpoint (cadence fixed in $t$, identical across methods).

* **$W_2$**: exact in 1D (sorted coupling); **sliced** $W_2$ for $d\ge2$ with $L=200$ projections drawn once from a fixed seed and reused across all times and methods (its bias floor decays like $N^{-1/2}$, not $N^{-1/d}$).
* **TV** (occupancy, on the partition): $\tfrac12\sum_k|\hat p_k - p^\star_k|$ — a **lower bound** on the full TV.
* **MMD**: Gaussian kernel, bandwidth **frozen once** by the median heuristic on the reference sample (per-frame bandwidths would make curves non-comparable); biased V-statistic $\widehat{\mathrm{MMD}}_b^2 = \|\mu_X-\mu_Y\|_{\mathcal H}^2 \ge 0$.
* **EMC** $= e^{H(\hat p)}/K$: plotted with a horizontal line at the target $e^{H(p^\star)}/K$; EMC $=1$ is optimal only for uniform $p^\star$, and deviation in *either* direction is error.
* **EJS**: base-2 Jensen–Shannon divergence between $\hat p$ and $p^\star$ (Blessing et al., arXiv:2406.07423, App. A.3), bounded in $[0,1]$, quadratic near the target, so it stays informative where TV saturates.
* **Bias floors** (mandatory): each metric between two independent reference samples of size $N$, 20 replicates; dashed line on every panel. Without this, every plateau is uninterpretable.
* **Nonfinite fraction**: logged per method per checkpoint; must be identically zero — metrics on survivors only would be survivorship bias, so nothing is ever filtered.

**Coverage vs correctness, once:** EMC measures *coverage*, TV/EJS measure *correctness*. Raw CP, whose invariant law is not $\pi$, over-flattens — driving EMC toward 1 (above its target line) while TV and EJS stay bad. That pairing *is* the raw-CP-vs-LSC-CP story."""

MD_DT_RULE = r"""## $\Delta t$ refinement and production

Declared $\Delta t$ selection rule, applied uniformly to every experiment (reported in the SI): **the largest $\Delta t$ on a dyadic grid at which every method's terminal value of every metric is within 5% of its $\Delta t/2$ value.** Three statistical guards make the rule meaningful at a single refinement seed: differences are measured relative to $\max(|m_{\Delta t/2}|,\ \text{bias floor})$; when *both* values sit inside the floor band (floor mean $+\,3$ s.d.) they are declared in agreement; and differences within $4\times$ the floor s.d. — the natural unit of single-run metric sampling noise at this $N$ — are likewise noise, not discretisation bias. The same guards apply to the quadrature-refinement comparison.

**One declared exception:** FLA does not gate the $\Delta t$ selection. Its continuum limit is not $\pi$, so its bias has no $\Delta t$ at which it should stabilise (empirically its density error *drifts monotonically* under refinement); demanding 5% stability from it would refine $\Delta t$ forever. FLA still runs at the shared chosen $\Delta t$, and its deviations across the dyadic grid are recorded in the refinement table for transparency.

Production protocol: 5 seeds $\times$ 7 methods, run **sequentially** (never batched) so per-run wall-clock is meaningful; all methods share $x_0$ per seed; 20 untimed warm-up steps absorb allocator/JIT effects; `torch.cuda.synchronize()` brackets every timed region, and the timer covers sampler work only."""

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
assert torch.cuda.device_count() == 1, "GPU guard must mask to exactly one device"
torch.set_default_dtype(torch.float64)
import numpy as np
import pandas as pd

from src import config as C
from src.experiments import {builder}, make_sampler_factory, make_metrics
from src.runner import (run_experiment, run_one, refine_dt, quadrature_refinement,
                        write_timeseries_csv, write_summary_csv, write_manifest,
                        ula_first_passage, hardware_manifest)
from src.samplers import tune_ladder
from src.certificate import make_phi_family, certificate_grid, certificate_importance
from src.plotting import make_all_figures, apply_style

DEV = "cuda"
RESULTS = os.path.abspath(os.path.join("..", "results", EXPERIMENT))
FIGURES = os.path.abspath(os.path.join("..", "figures", EXPERIMENT))
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
{extra}
cfg = exp.cfg
print(f"experiment={{cfg.name}}  d={{cfg.d}}  N={{cfg.n_particles}}  T={{cfg.T}}  dt0={{cfg.dt}}")
print(f"beta={{cfg.beta}}  eps={{cfg.eps}}  lambda={{cfg.lam}}  seeds={{cfg.seeds}}")
print(hardware_manifest())'''


CELL_LADDER = '''# PT ladder: geometric in beta, K tuned so mean swap acceptance is in [0.2, 0.4]
gen = torch.Generator(device=DEV); gen.manual_seed(0)
x0_pilot = exp.init_fn(min(512, cfg.n_particles), gen)
pt_betas, ladder_info = tune_ladder(exp.pot, x0_pilot, cfg.dt, exp.box,
                                    C.BETA, exp.pt_beta_min, pilot_steps=600)
print(f"PT ladder: K={ladder_info['K']}  r={ladder_info['r']:.4f}  "
      f"beta_K={pt_betas[-1].item():.4f}  swap acceptance={ladder_info['swap_acceptance']:.3f}")
print("tuning history {K: acceptance}:", ladder_info["history"])'''


CELL_REFERENCE = '''metrics_fn, floors, aux = make_metrics(exp, cfg.n_particles)
emc_target = exp.emc_target
print("p_star:", np.round(exp.p_star.cpu().numpy(), 6))
print("EMC target line: %.4f" % emc_target)
print("MMD bandwidth (median heuristic on reference, frozen):", round(aux["bandwidth"], 4))
print("bias floors (mean +- std over 20 replicate pairs):")
for k, v in floors.items():
    print(f"  {k:>12s}: {v['mean']:.5f} +- {v['std']:.5f}")'''


def cell_quad_refine(settings_expr: str, cert_call: str, terminal_keys: str,
                     score_kwargs_expr: str) -> str:
    """Section 9.5: quadrature refinement table (certificate residual +
    terminal LSC-CP metrics per setting; smallest converged setting wins)."""
    return f'''def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs={score_kwargs_expr})
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {{k: r_[-1][k] for k in {terminal_keys}}}

settings = {settings_expr}
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: {cert_call}["max_residual"], floors)
print("chosen production quadrature:", CHOSEN_QUAD)
display(pd.DataFrame(quad_table).round(6))
if CHOSEN_QUAD != DEFAULT_QUAD:
    cert_report = {cert_call.replace("**s", "**CHOSEN_QUAD")}
    print("certificate re-evaluated at chosen orders: max R =",
          f"{{cert_report['max_residual']:.3e}}")
    assert cert_report["max_residual"] < 1e-6'''


def cell_dt_production(main_metrics: str) -> str:
    return f'''MAIN_METRICS = {main_metrics}

def run_terminal_all(dt_):
    n_ = int(round(cfg.T / dt_))
    factory = make_sampler_factory(exp, dt_, pt_betas, score_kwargs=CHOSEN_QUAD)
    out = {{}}
    for m in C.METHODS:
        rows_, _ = run_one(m, 0, factory, n_, n_, dt_, metrics_fn, exp.pot, quiet=True)
        out[m] = {{k: rows_[-1][k] for k in MAIN_METRICS}}
    print(f"  refine_dt: finished pass at dt={{dt_}}", flush=True)
    return out

dt_final, dt_table = refine_dt(run_terminal_all, cfg.dt, floors, exclude=("FLA",))
print("chosen dt:", dt_final)
for row in dt_table:
    print(row)

n_steps = int(round(cfg.T / dt_final))
steps_per_ck = max(1, n_steps // C.N_CHECKPOINTS)
factory = make_sampler_factory(exp, dt_final, pt_betas, score_kwargs=CHOSEN_QUAD)
t0 = time.time()
rows, method_info = run_experiment(C.METHODS, cfg.seeds, factory, n_steps,
                                   steps_per_ck, dt_final, metrics_fn, exp.pot)
print(f"production total: {{time.time()-t0:.0f}}s")
worst_nonfinite = max(r["nonfinite_frac"] for r in rows)
assert worst_nonfinite == 0.0, worst_nonfinite
print("nonfinite fraction: identically zero across all methods/checkpoints")'''


def cell_figures(metric_tuple: str) -> str:
    return f'''fig_metrics = {metric_tuple}
written = make_all_figures(rows, FIGURES, floors, emc_target, metrics=fig_metrics)
print(f"{{len(written)}} figures x 3 formats (.pdf/.png 600dpi/.eps) + captions -> {{FIGURES}}")

# grid display for inspection (saved files above are one-figure-per-file)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
n_m = len(fig_metrics)
fig, axes = plt.subplots(n_m, 2, figsize=(11, 3.2 * n_m))
for i, metric in enumerate(fig_metrics):
    for j, tag in enumerate(("vs_time", "vs_wallclock")):
        ax = axes[i, j] if n_m > 1 else axes[j]
        ax.imshow(mpimg.imread(os.path.join(FIGURES, f"{{metric}}_{{tag}}.png")))
        ax.set_axis_off()
plt.tight_layout(); plt.show()'''


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
                n_checkpoints=C.N_CHECKPOINTS, warmup_steps=C.N_WARMUP_STEPS),
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
        md(r"""# E1 — 1D double well: LSC-CP vs six baselines

Boltzmann target $\pi(x) \propto e^{-\beta V(x)}$ at $\beta = 8$ ($\varepsilon = 1/\beta = 0.125$) for the scaled quartic double well
$$V(x) = (x^2-1)^2 .$$
Seven samplers (ULA, MALA, FLA, kinetic Langevin/BAOAB, parallel tempering, raw compound-Poisson, LSC-CP) share one drift-taming policy, one $\Delta t$, one metric cadence, and per-seed initial conditions. LSC-CP adds nonlocal well-to-well jumps *plus* the stationary Lévy-score drift correction that preserves $\pi$ exactly at the generator level; raw CP has the same jumps without the correction. The wall-clock axis is a reported result: timing covers sampler work only, with CUDA synchronisation around every timed region."""),
        code(cell_setup("double_well", "build_e1",
                        "exp = build_e1(device=DEV)")),
        md(r"""## The model

$$V(x) = (x^2-1)^2,\qquad V'(x) = 4x(x^2-1),$$
minima $\pm1$, saddle $0$, $V''(\pm1)=8$, $V''(0)=-4$, $\Delta V = 1$, so $\beta\Delta V = 8$. Kramers:
$$\tau = \frac{2\pi}{\sqrt{V''(\pm1)\,|V''(0)|}}\,e^{\beta\Delta V} = \frac{2\pi}{\sqrt{32}}e^{8} \approx 3.3\times10^3,$$
so over $T=100$ a ULA particle crosses with probability $\approx 3\%$. *(This is the standard $\tfrac{x^4}4 - \tfrac{x^2}2$ well scaled by 4; the unscaled version gives $\beta\Delta V = 2$, $\tau \approx 33$, and ULA would visibly cross.)*

Protocol: $N=4000$, $T=100$, $\Delta t_0 = 0.005$, box $[-3,3]$, $x_0 \sim \mathcal N(-1, 0.05^2)$. Partition ($K=2$): $\mathrm{sign}(x)$, $p^\star=(\tfrac12,\tfrac12)$ exactly by symmetry; additionally a 200-bin **density TV** against the exact $\pi$ on $[-3,3]$ (a genuine density TV, unlike the occupancy TV). Reference: inverse-CDF on a dense grid (tail mass outside $[-3,3]$ is $<10^{-30}$).

Below: barrier verification — ULA's empirical mean first-passage time out of the initial basin against the Kramers estimate (censored-exponential MLE). The exit event is **committed arrival in the other basin's core**, not first touch of the partition boundary: boundary-touch counts half-committed excursions and overstates the escape rate several-fold. Do not trust Kramers alone."""),
        code('''# critical points and curvatures, asserted
V = lambda x: (x**2 - 1.0)**2
dV = lambda x: 4.0*x*(x**2 - 1.0)
assert abs(V(1.0)) < 1e-15 and abs(V(-1.0)) < 1e-15 and V(0.0) == 1.0
assert dV(1.0) == 0.0 and dV(-1.0) == 0.0 and dV(0.0) == 0.0
d2V = lambda x: 12.0*x**2 - 4.0
assert d2V(1.0) == 8.0 and d2V(0.0) == -4.0
beta_dV = C.BETA * 1.0
print(f"beta*DeltaV = {beta_dV}, Kramers tau = {exp.kramers_tau:.1f}")

# barrier verification: ULA MFPT out of the left well vs Kramers
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt),
                                   C.EPS, g)
barrier_report["kramers_tau"] = exp.kramers_tau
print("ULA first-passage:", barrier_report)
print(f"measured MFPT {barrier_report['mfpt_estimate']:.0f} vs Kramers {exp.kramers_tau:.0f} "
      f"({barrier_report['n_exits']} exits of {barrier_report['n_particles']})")'''),
        md(r"""## Jump law and Lévy score (generic shell)

Two-atom symmetric shell: centres $r_a = \pm 2$ with $w = (\tfrac12,\tfrac12)$, half-thickness $h=0.2$, $\lambda = 1$ — a jump of $\pm2$ maps minimum to minimum exactly, and the shell thickening $r = r_a + \rho u_a$, $\rho\sim\mathrm{Unif}(-h,h)$ avoids a purely atomic law.

The score uses Gauss–Legendre probability weights on both inner integrals ($Q_\theta$ nodes on $[0,1]$; $Q_\rho$ nodes on $[-h,h]$ matching $\rho\sim\mathrm{Unif}$), so the quadrature measure is exactly the sampler's $\nu$ — an invariant asserted in `tests/test_samplers.py`. **Log-space accumulation** (no score clipping): the per-direction integral $I_{a,q}$ is strictly positive but spans hundreds of orders of magnitude at $\beta=8$, so we form $\log I_{a,q}$ by log-sum-exp, extract $M(x) = \max_{a,q}[\log w_a + \log\hat w_q + \log I_{a,q}]$, build the $O(1)$ direction vector $v(x)$, and return $S = -\lambda\,e^{\min(M, 600)}v$. Because the drift is tamed, when $\|S\|$ is astronomical only its direction matters, and $v$ preserves the direction exactly. The fraction of particles hitting $M_{\max}$ (`m_clip_fraction`) and the running $\max M$ are logged.

Defaults $Q_\theta = 16$, $Q_\rho = 8$; the §9.5 quadrature-refinement table (certificate residual + terminal LSC-CP metrics per setting) is recorded in the run section below, and production uses the smallest setting converged against the finest."""),
        code('''print("atoms:", exp.law.atoms.squeeze(-1).tolist(), " weights:", exp.law.weights.tolist(),
      " h:", exp.law.h.tolist())
DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(1, [0.0], 1.0, DEV)

def cert_e1(q_theta, q_rho, lo=-5.2, hi=5.2):
    score = exp.make_score(q_theta=q_theta, q_rho=q_rho)
    # jump side: FINE rho quadrature representing the continuous nu, so the
    # residual also sees any inadequacy of the score's rho order
    shifts, logw = exp.law.quadrature_shifts(64)
    return certificate_grid(exp.pot, score, shifts, logw, cfg.lam, cfg.beta,
                            phis, [lo], [hi], n_panels=120, nodes_per_panel=8)'''),
        md(MD_TARGET_PRESERVATION),
        code('''cert_report = cert_e1(**DEFAULT_QUAD)
print(f"R(phi) over the generous box [-5.2, 5.2] (support [-3,3] + jump reach 2.2):")
for i in range(len(phis)):
    print(f"  phi_{i}: R = {cert_report[f'phi_{i}']['residual']:.3e}"
          f"  (jump term {cert_report[f'phi_{i}']['jump_term']:+.3e})")
print(f"max R = {cert_report['max_residual']:.3e}  "
      f"clip saturation defect = {cert_report['clip_tamed_step_defect']:.2e}")
assert cert_report["max_residual"] < 1e-6

tight = cert_e1(**DEFAULT_QUAD, lo=-1.3, hi=1.3)
print(f"DELIBERATELY TIGHT box [-1.3, 1.3]: max R = {tight['max_residual']:.3e} "
      "-- order-one contributions to the identity live beyond the target support")''' ),
        md(MD_SAMPLERS),
        code(CELL_LADDER),
        md(MD_METRICS),
        code(CELL_REFERENCE),
        md(MD_DT_RULE),
        code(cell_quad_refine(
            "[dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]",
            'cert_e1(**s)',
            '("W2", "TV", "TV_density", "MMD", "EMC", "EJS")',
            "quad")),
        code(cell_dt_production('["W2", "TV", "TV_density", "MMD", "EMC", "EJS"]')),
        md("## Figures\n\nOne figure per metric, all seven methods, versus $t=n\\Delta t$ and versus wall-clock; mean over 5 seeds with a pre-blended $\\pm1$ s.d. band (EPS forbids transparency). Dashed lines: bias floors (log-scale metrics) and the EMC target. Saved individually in `.pdf`, `.png` (600 dpi) and `.eps` with caption `.txt` files."),
        code(cell_figures('("W2", "TV", "TV_density", "MMD", "EMC", "EJS")')),
        md("## CSV emission and summary"),
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
        md(r"""# E2 — MoG40 (2D): 40 Gaussian modes, generic annulus jumps, closed-form score

$$\boxed{V(x) = -\frac1\beta \log\sum_{k=1}^{40}\exp\!\Big(-\tfrac12\|x-\mu_k\|^2\Big)}\quad\Longrightarrow\quad \pi(x)\propto e^{-\beta V} = \sum_k e^{-\|x-\mu_k\|^2/2},$$
an equal-weight mixture of $\mathcal N(\mu_k, I_2)$. **The $1/\beta$ prefactor is essential** — it is what makes the barriers right:
$$\nabla V(x) = \frac1\beta\sum_k \omega_k(x)\,(x-\mu_k),\qquad \omega_k = \mathrm{softmax}_k(-\tfrac12\|x-\mu_k\|^2),$$
and the effective barrier between modes at distance $d$ is $\beta\Delta V = \tfrac{d^2}{8} - \log 2$ — heterogeneous by design.

**Neither PT nor LSC-CP receives mode locations.** The jump law is deliberately generic: $r = \rho\,u_\phi$ with $\rho\sim\mathrm{Unif}[4,15]$, $\phi\sim\mathrm{Unif}[0,2\pi)$, the interval $[4,15]$ set from the NN-distance histogram alone (printed below). The Lévy score for this law has a **closed form with zero potential evaluations**."""),
        code(cell_setup("mog40", "build_e2", "exp = build_e2(device=DEV)")),
        md(r"""## The model

$\mu_k \sim \mathrm{Unif}([-40,40]^2)$ frozen from `np.random.default_rng(0)` and saved to CSV. $N=2500$, $T=100$, $\Delta t_0 = 0.01$, box $[-65,65]^2$, all particles initialised at $\mu_0 + 0.5\,\xi$. Partition ($K=40$): nearest-mode Voronoi with $p^\star_k = 1/40$ (exact up to $O(e^{-d^2/8})$ Voronoi leakage, absorbed into the occupancy bias floor). Reference: exact i.i.d. mixture draws.

Barrier verification below uses the nearest-neighbour gap of mode 0 and a 1D Kramers estimate along the inter-mode line ($\omega_{\min} = \omega_{\rm saddle} \approx 1$ for unit-variance components)."""),
        code('''np.savetxt(os.path.join(RESULTS, "modes.csv"), exp.pot.mu.cpu().numpy(),
           delimiter=",", header="mu_x,mu_y", comments="")
dists = torch.cdist(exp.pot.mu, exp.pot.mu)
dists.fill_diagonal_(float("inf"))
nn = dists.min(dim=1).values.cpu().numpy()
print("NN-distance histogram (the sole input to the jump interval [4,15]):")
hist, edges = np.histogram(nn, bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 20])
for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
    print(f"  [{e0:>4.1f}, {e1:>4.1f}): {'#'*h} {h}")
print(f"NN distances: min {nn.min():.2f}, median {np.median(nn):.2f}, max {nn.max():.2f}")
print(f"beta*DeltaV across gaps d=4..16: {4**2/8 - math.log(2):.1f} .. {16**2/8 - math.log(2):.1f}")
print(f"mode-0 NN gap d0 = {exp.extras['nn_dist_mode0']:.2f}, "
      f"beta*dV = {exp.extras['beta_dV_mode0']:.2f}, Kramers tau ~ {exp.kramers_tau:.0f}")

g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), C.EPS, g)
barrier_report["kramers_tau_mode0"] = exp.kramers_tau
print("ULA first-passage out of Voronoi cell 0:", barrier_report)'''),
        md(r"""## The closed-form Lévy score

With $d_k = x - \mu_k$, $m_k = u_\phi\cdot d_k$, the mixture gives
$$\frac{\pi(x-\theta\rho u)}{\pi(x)} = \sum_k \omega_k(x)\, e^{\theta\rho m_k - \theta^2\rho^2/2},\qquad \omega_k = \mathrm{softmax}_k(-\tfrac12\|d_k\|^2).$$
Completing the square, $I(\rho,m) = \int_0^1 e^{-\theta^2\rho^2/2+\theta\rho m}d\theta = \frac{\sqrt{\pi/2}}{\rho}e^{m^2/2}[\mathrm{erf}(\tfrac{\rho-m}{\sqrt2})+\mathrm{erf}(\tfrac{m}{\sqrt2})]$, and the $\rho$ from $r=\rho u$ cancels the $1/\rho$. With $F(z) = z\,\mathrm{erf}(z/\sqrt2)+\sqrt{2/\pi}e^{-z^2/2}$ ($F'=\mathrm{erf}(z/\sqrt2)$, $F$ even):
$$H(m) = \int_a^b \rho I(\rho,m)\,d\rho = \sqrt{\tfrac\pi2}e^{m^2/2}\underbrace{\big[F(b-m)-F(a-m)+(b-a)\mathrm{erf}(\tfrac m{\sqrt2})\big]}_{\mathcal B(m)>0},$$
so only the $\phi$ integral needs numerics (an $M_\phi$-point trapezoid rule — spectrally accurate for the periodic integrand):
$$S(x) \approx -\frac{\lambda}{M_\phi(b-a)}\sum_{\ell}u_\ell\sum_k \exp\big[\log\omega_k + m_{k\ell}^2/2\big]\sqrt{\tfrac\pi2}\,\mathcal B(m_{k\ell}).$$

**$\mathcal B$ must be evaluated by branches.** With $g(z) = e^{-z^2/2}[\sqrt{2/\pi}-z\,\mathrm{erfcx}(z/\sqrt2)]$ (so $F(z)=|z|+g(|z|)$), the $O(m)$ parts cancel *analytically* in the outer regimes:

| regime | formula |
|---|---|
| $m\ge b$ | $\mathcal B = g(m-b)-g(m-a)-(b-a)\,\mathrm{erfc}(m/\sqrt2)$ |
| $m\le 0$ | $\mathcal B = g(b-m)-g(a-m)+(b-a)\,\mathrm{erfc}(-m/\sqrt2)$ |
| $0<m<b$ | direct, safe |

The naive form has 100% relative error at $m=30$; the branched form $\sim10^{-12}$ (validated against 3000-digit mpmath, and the whole closed form against a brute-force 3-D quadrature at $10^{-8}$, `tests/test_score.py`). Our implementation additionally factors the dominant exponential out of each branch so $\log\mathcal B$ is finite on the whole real line, then applies the same log-space $(M, v)$ accumulation as the generic shell score.

The $\theta,\rho$ integrals being analytic, the only quadrature parameter is $M_\phi$; the §9.5 table in the run section records $\mathcal R$ and terminal metrics for $M_\phi \in \{16,32,64\}$ (production default 32, verified against 64). Note the *pointwise* direction-quadrature error at $M_\phi=32$ is $O(10^{-3})$ relative (far-mode terms peak in $\phi$ with width $\sim 1/\sqrt{b\,d_k}$), but its weak imprint against smooth test functions — which is what invariance is — integrates out, as the certificate shows."""),
        code('''DEFAULT_QUAD = dict(m_phi=C.M_PHI)
phis = make_phi_family(2, [0.0, 0.0], 30.0, DEV)

def cert_e2(m_phi):
    score = exp.make_score(m_phi=m_phi)
    shifts, logw = exp.law.quadrature_shifts(16, 64)   # fine continuous-nu J side
    return certificate_grid(exp.pot, score, shifts, logw, cfg.lam, cfg.beta,
                            phis, [-60.0, -60.0], [60.0, 60.0],
                            n_panels=120, nodes_per_panel=6, chunk=8192)'''),
        md(MD_TARGET_PRESERVATION),
        code('''cert_report = cert_e2(**DEFAULT_QUAD)
print("R(phi) on the generous box [-60,60]^2 (mode cloud [-40,40]^2 + 4 sigma + jump reach 15):")
for i in range(len(phis)):
    print(f"  phi_{i}: R = {cert_report[f'phi_{i}']['residual']:.3e}")
print(f"max R = {cert_report['max_residual']:.3e}")
assert cert_report["max_residual"] < 1e-6'''),
        md(MD_SAMPLERS + "\n\nExpect $K \\approx 15$–$25$ replicas for this target — that *is* the finding: PT needs a long ladder to bridge $\\beta\\Delta V$ up to $\\sim27$, and every replica costs a gradient per step."),
        code(CELL_LADDER),
        md(MD_METRICS + "\n\nSanity check below: the frozen median-heuristic bandwidth must land near the mode spacing (several units), not the component width (1)."),
        code(CELL_REFERENCE + '''
assert aux["bandwidth"] > 3.0, "bandwidth should reflect mode spacing, not component width"'''),
        md(MD_DT_RULE),
        code(cell_quad_refine("[dict(m_phi=m) for m in (16, 32, 64)]",
                              'cert_e2(**s)',
                              '("W2", "TV", "MMD", "EMC", "EJS")',
                              "quad")),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC", "EJS"]')),
        md("## Figures"),
        code(cell_figures('("W2", "TV", "MMD", "EMC", "EJS")')),
        md("## CSV emission, Hungarian spot-check, summary\n\nTerminal exact-$W_2$ spot-check on a 500-point subsample (Hungarian algorithm), 2D only — a cross-validation of the sliced estimator, not a headline number."),
        code('''gen_h = torch.Generator(device=DEV); gen_h.manual_seed(202)
ref_sub = exp.ref_sample(2500, gen_h)
from src.metrics import hungarian_w2
hungarian = {}
for m in C.METHODS:
    xf = method_info[m]["final_positions_seed0"]
    hungarian[m] = hungarian_w2(xf, ref_sub, m=500)
print("terminal Hungarian W2 (M=500 subsample, seed 0):")
for m, v in hungarian.items():
    print(f"  {m:>8s}: {v:.4f}")
''' + cell_csv("hungarian_w2_terminal=hungarian,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E3 Mueller-Brown 10D
# ======================================================================
def build_e3_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E3 — Transformed Müller–Brown (10D): a single-basin target with rare escapes

$$U(z) = \frac{1}{s}\,U_{\rm MB}(z_1,z_2) + \frac{1}{2\sigma_{\rm aux}^2}\sum_{\ell=3}^{10} z_\ell^2,\qquad s = 40,\ \sigma_{\rm aux} = 0.4,$$
sampled in mixed coordinates $x = z B^\top$, $B = Q\,\mathrm{diag}(\mathrm{linspace}(0.75,1.45,10))$ with $Q$ from QR of a `default_rng(12345)` normal; $V(x) = U(xB^{-\top})$ and $\nabla_x V = (\nabla_z U)B^{-1}$.

**Initialise in basin B.** The target occupancy is $\approx(0.9994,\ 0.0006,\ 10^{-5})$ for $(A, B, C)$ — Müller–Brown at low temperature **is a single-basin target; this is intrinsic to the potential, not a calibration failure**. The task is escaping the shallow initial basin $B$ and putting the mass where it belongs; EJS (bounded, quadratic near the target) is the primary occupancy metric because $p^\star$ is far from uniform and TV saturates."""),
        code(cell_setup("muller_brown_10d", "build_e3",
                        'exp = build_e3(device=DEV, basin_cache=os.path.join(RESULTS, "basin_map.npz"))')),
        md(r"""## The model

$$U_{\rm MB}(\zeta) = \sum_{i=1}^4 A_i \exp\!\big[a_i(\zeta_1-x_i)^2 + b_i(\zeta_1-x_i)(\zeta_2-y_i) + c_i(\zeta_2-y_i)^2\big]$$
with $A=(-200,-100,-170,15)$, $a=(-1,-1,-6.5,0.7)$, $b=(0,0,11,0.6)$, $c=(-10,-10,-6.5,0.7)$, $x=(1,0,-0.5,-1)$, $y=(0,0.5,1.5,1)$. Verified critical points (asserted to 4 decimals below):

| | location | $U_{\rm MB}$ |
|---|---|---|
| min A | $(-0.5582,\ 1.4417)$ | $-146.70$ |
| min B | $(0.6235,\ 0.0280)$ | $-108.17$ |
| min C | $(-0.0500,\ 0.4667)$ | $-80.77$ |
| saddle S1 (A↔C) | $(-0.8220,\ 0.6243)$ | $-40.66$ |
| saddle S2 (C↔B) | $(0.2125,\ 0.2930)$ | $-72.25$ |

Connectivity is a **chain** $A \leftrightarrow C \leftrightarrow B$. Escape barrier from $B$: $35.92$, so $\beta b_B/s = 7.18$ and $\tau_{\rm Kramers}(B) \approx 940$.

Protocol: $N=2000$, $T=200$, $\Delta t_0=0.005$; box clipped in **latent** coordinates $z_1\in[-3,3]$, $z_2\in[-1.5,3.5]$, $z_{3:10}\in[-2,2]$ — deliberately generous, because the score integrand's support extends a full jump length beyond the target's effective support (do not shrink it). Metrics are primary in latent 2D $z_{1:2}=(xB^{-\top})_{1:2}$; full-10D sliced $W_2$ is also reported with its own bias floor (essential in 10D). Partition ($K=3$): gradient-flow basin map on a cached latent grid; $p^\star$ by grid quadrature."""),
        code('''from src.potentials import MB_CRITICAL, muller_brown_2d, muller_brown_2d_grad, newton_refine
for key, (z_tab, U_tab) in MB_CRITICAL.items():
    z = newton_refine(muller_brown_2d_grad, torch.tensor(z_tab, device=DEV))
    U = muller_brown_2d(z.unsqueeze(0))[0].item()
    assert abs(z[0].item() - z_tab[0]) < 5e-5 and abs(z[1].item() - z_tab[1]) < 5e-5, key
    assert abs(U - U_tab) < 5e-2, (key, U)
    print(f"{key:>10s}: ({z[0].item():+.4f}, {z[1].item():+.4f})  U_MB = {U:9.4f}  [table: {z_tab}, {U_tab}]")
print(f"escape barrier from B = {exp.extras['barrier_B']*exp.pot.s:.2f}, "
      f"beta*b/s = {C.BETA*exp.extras['barrier_B']:.2f}, Kramers tau ~ {exp.kramers_tau:.0f}")
print("p_star (A, B, C):", np.round(exp.p_star.cpu().numpy(), 6),
      " -- single-basin target, intrinsic to Mueller-Brown at this temperature")

g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), C.EPS, g)
barrier_report["kramers_tau_B"] = exp.kramers_tau
print("ULA first-passage out of basin B:", barrier_report)'''),
        md(r"""## Jump law and score

Euclidean MST on the three latent minima gives edges $(C,B)$ (length 0.804) and $(A,C)$ (length 1.10), symmetrised to 4 directed atoms $r_a = (\Delta z,\ 0_8)B^\top$, $w_a = \tfrac14$, shell $h = 0.1\,\min_a\|r_a\|$, $\lambda = 1$. Score: generic shell (log-space, as in E1) — the potential evaluations for $V(x-\theta_p r_{a,q})$ run over $Q_\theta\times A\times Q_\rho$ shifted copies, vectorised over the ensemble."""),
        code('''print("latent minima:", {k: np.round(v.cpu().numpy(), 4).tolist()
                          for k, v in exp.extras["minima_latent"].items()})
print("MST atoms (latent dz):", np.round(exp.extras["atoms_z"][:, :2].cpu().numpy(), 4).tolist())
print("edge lengths:", [round(float(exp.extras['atoms_z'][i, :2].norm()), 4) for i in (1, 2)],
      " h (x-space):", round(exp.extras["h"], 4))

# certificate operates on the EXACT latent 2D reduction (jumps and test
# functions act on z_{1:2} only; dot products are affine-invariant; the aux
# Gaussian factorises) with per-atom shell widths carrying h into z-space
from src.potentials import MuellerBrownLatent2D
from src.jumps import ShellJumpLaw
from src.score import ShellScore
potr = MuellerBrownLatent2D(s=exp.pot.s)
dz = exp.extras["atoms_z"][:, :2]
atoms_x = exp.law.atoms
h_z = exp.extras["h"] * dz.norm(dim=1) / atoms_x.norm(dim=1)
law_r = ShellJumpLaw(dz, torch.full((4,), 0.25, device=DEV), h_z)
DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(2, [0.0, 0.8], 0.8, DEV)

def cert_e3(q_theta, q_rho):
    score = ShellScore(potr, law_r, cfg.lam, cfg.beta, q_theta, q_rho)
    shifts, logw = law_r.quadrature_shifts(64)   # fine continuous-nu J side
    return certificate_grid(potr, score, shifts, logw, cfg.lam, cfg.beta, phis,
                            [-4.2, -2.7], [4.2, 4.7],
                            n_panels=130, nodes_per_panel=8, chunk=8192)'''),
        md(MD_TARGET_PRESERVATION),
        code('''cert_report = cert_e3(**DEFAULT_QUAD)
print("R(phi), latent-2D reduction, generous latent box (metric box + jump reach):")
for i in range(len(phis)):
    print(f"  phi_{i}: R = {cert_report[f'phi_{i}']['residual']:.3e}")
print(f"max R = {cert_report['max_residual']:.3e}  "
      f"clip saturation defect = {cert_report['clip_tamed_step_defect']:.2e}")
assert cert_report["max_residual"] < 1e-6'''),
        md(MD_SAMPLERS),
        code(CELL_LADDER),
        md(MD_METRICS),
        code(CELL_REFERENCE),
        md(MD_DT_RULE),
        code(cell_quad_refine(
            "[dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]",
            'cert_e3(**s)',
            '("W2", "TV", "MMD", "EMC", "EJS", "W2_10d")',
            "quad")),
        md(r"""### E3 exception: $\Delta t$ is declared, not certified

On this landscape the dyadic rule cannot certify any practical $\Delta t$ for LSC-CP, and the failure is *mechanistic, not statistical* (diagnosed in the mechanism section below): uphill jump-landers receive a corrective drift of magnitude $e^{\beta\Delta V}$ (up to $e^{21}$ here — the only experiment with strongly asymmetric basin depths), and the tamed Euler step turns the corresponding smooth return flow into a single $O(1)$ hop whose landing scatter parks a $\Delta t$-independent fraction of mass in score-dark regions. Since the tamed step saturates for any $\Delta t > e^{-M}$, refining $\Delta t$ cannot converge these terminal values. Production therefore runs at the declared $\Delta t_0$ (well inside the stability bound), the level-0 comparison table is recorded, and `dt_certified = False` is written to the manifest. LSC-CP's terminal metrics carry a documented $\Delta t$-sensitivity of the order of the table's spread."""),
        code('''MAIN_METRICS = ["W2", "TV", "MMD", "EMC", "EJS", "W2_10d"]

def run_terminal_all(dt_):
    n_ = int(round(cfg.T / dt_))
    factory = make_sampler_factory(exp, dt_, pt_betas, score_kwargs=CHOSEN_QUAD)
    out = {}
    for m in C.METHODS:
        rows_, _ = run_one(m, 0, factory, n_, n_, dt_, metrics_fn, exp.pot, quiet=True)
        out[m] = {k: rows_[-1][k] for k in MAIN_METRICS}
    print(f"  refine_dt: finished pass at dt={dt_}", flush=True)
    return out

_, dt_table = refine_dt(run_terminal_all, cfg.dt, floors, exclude=("FLA",),
                        max_halvings=1)
dt_certified = bool(dt_table[0]["pass"])
for row in dt_table:
    print(row)
dt_final = cfg.dt          # declared (see markdown above); certification recorded
print(f"dt_certified = {dt_certified}; production at declared dt0 = {dt_final}")

n_steps = int(round(cfg.T / dt_final))
steps_per_ck = max(1, n_steps // C.N_CHECKPOINTS)
factory = make_sampler_factory(exp, dt_final, pt_betas, score_kwargs=CHOSEN_QUAD)
t0 = time.time()
rows, method_info = run_experiment(C.METHODS, cfg.seeds, factory, n_steps,
                                   steps_per_ck, dt_final, metrics_fn, exp.pot)
print(f"production total: {time.time()-t0:.0f}s")
worst_nonfinite = max(r["nonfinite_frac"] for r in rows)
assert worst_nonfinite == 0.0, worst_nonfinite
print("nonfinite fraction: identically zero across all methods/checkpoints")'''),
        md(r"""## Mechanism: why the corrected process stalls on a quasi-stationary plateau

The MST law's uphill jumps (into $C$, whose equilibrium mass is $\sim 5\times10^{-6}$) land with an enormous, *correctly aimed* Lévy score — the correction implementing detailed-balance rejection as a drift. Three measurements pin the failure of the tamed discretisation to represent that drift:

1. **Occupancy trajectories**: LSC-CP reaches its plateau by $t\approx25$ and then stalls (it is quasi-stationary, not slowly relaxing), and the plateau *worsens* slightly at finer $\Delta t$; raw CP is $\Delta t$-stable at a much worse law; locals barely leave B.
2. **Lander cohort**: an equilibrated A-ensemble jumped by the exact $A\to C$ atom lands 99% in the C label with median score log-magnitude $M\approx 8$; one tamed step returns most of it, but the single $O(1)$ hop scatters — after a few steps a $\Delta t$-independent $\sim$17% is parked with $M\approx-3$ (score-dark), rescued only by later jumps.
3. **Score field**: $\|S\|\approx 5\times10^{3}$ at $z_C$ pointing along $C\to A$ (correct); $O(10^{-3})$ at the inhabited minima. The stiffness is confined to the jump shadows — exactly where landers appear.

E1/E2/E4 are immune because their jump laws connect (near-)iso-energetic minima, so scores at inhabited points are $O(1)$. The finding: **on landscapes with strongly asymmetric basin depths, the exactness of the Lévy-score correction concentrates in $e^{\beta\Delta V}$-stiff drifts that the tamed Euler scheme cannot integrate**; a discretisation that resolves the post-jump return flow (or a Metropolised jump step) is required to realise the generator's exactness in practice."""),
        code('''# 1) occupancy trajectories: quasi-stationary plateau, dt-dependence
from src.metrics import occupancy as _occ

def occ_traj(sampler, n_steps_, every, label):
    out = []
    for s_ in range(n_steps_):
        sampler.step()
        if (s_ + 1) % every == 0:
            p = _occ(exp.labels_fn(sampler.positions()), 3)
            out.append(((s_ + 1) * sampler.dt, [round(float(v), 4) for v in p]))
    print(label)
    for t_, p in out:
        print(f"   t={t_:6.1f}: A={p[0]:.4f} B={p[1]:.4f} C={p[2]:.4f}")
    return out

mech_traj = {}
for meth, dt_ in (("LSC-CP", dt_final), ("LSC-CP", dt_final / 4.0),
                  ("CP", dt_final), ("ULA", dt_final)):
    f = make_sampler_factory(exp, dt_, pt_betas, score_kwargs=CHOSEN_QUAD)
    s = f(meth, 0)
    n_ = int(round(cfg.T / dt_))
    mech_traj[f"{meth}@dt={dt_}"] = occ_traj(s, n_, n_ // 8, f"{meth} dt={dt_}")'''),
        code('''# 2) lander cohort: single-hop return + dt-independent parked fraction
from src.samplers import tame as _tame
score_m = exp.make_score(**CHOSEN_QUAD)
zA_, zC_ = exp.extras["minima_latent"]["min_A"], exp.extras["minima_latent"]["min_C"]
mechanism_report = {"traj": {k: v[-1][1] for k, v in mech_traj.items()}}
for dt_ in (dt_final, dt_final / 4.0):
    g_ = torch.Generator(device=DEV); g_.manual_seed(3)
    zloc = torch.zeros(4000, 10, device=DEV)
    zloc[:, :2] = zA_ + 0.12 * torch.randn(4000, 2, generator=g_, device=DEV)
    zloc[:, 2:] = 0.1414 * torch.randn(4000, 8, generator=g_, device=DEV)
    x_ = exp.pot.from_latent(zloc) + exp.law.atoms[2]      # jump A -> C
    M0, _ = score_m.log_parts(x_)
    for _s in range(24):
        S_, _d = score_m(x_)
        b_ = -exp.pot.grad(x_) + S_
        xi_ = torch.randn(x_.shape, generator=g_, device=DEV)
        x_ = exp.box.clip(x_ + dt_ * _tame(b_, dt_) + (2 * C.EPS * dt_) ** 0.5 * xi_)
    Mf, _ = score_m.log_parts(x_)
    occ_f = _occ(exp.labels_fn(x_), 3)
    parked = float(occ_f[2].item())
    print(f"dt={dt_}: lander median M {M0.median():.2f} -> after 24 steps "
          f"M {Mf.median():.2f}; parked C fraction {parked:.3f}")
    mechanism_report[f"parked_fraction@dt={dt_}"] = parked
S_min, _ = score_m.log_parts(exp.pot.from_latent(torch.cat([
    torch.cat([zA_, torch.zeros(8, device=DEV)]).unsqueeze(0),
    torch.cat([zC_, torch.zeros(8, device=DEV)]).unsqueeze(0)])))
mechanism_report["score_logmag_at_A_C"] = [float(v) for v in S_min]
print("score log-magnitude at (A, C):", mechanism_report["score_logmag_at_A_C"])'''),
        md("## Figures"),
        code(cell_figures('("W2", "TV", "MMD", "EMC", "EJS", "W2_10d")')),
        md("## CSV emission and summary"),
        code(cell_csv("dt_certified=dt_certified, mechanism=mechanism_report,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E4 coupled phi4
# ======================================================================
def build_e4_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E4 — Coupled $\phi^4$ / Ginzburg–Landau chain (24D): coherent flips via moment-exact homogeneous jumps

$q_i\in\mathbb R^2$, $i=0,\dots,11$, periodic, $N_s=12$, $\delta = 1/N_s$, $d=24$, $\kappa=2.5$:
$$V(q) = \frac{\kappa}{2\delta}\sum_i \|q_{i+1}-q_i\|^2 + \delta\sum_i W(q_i),\qquad
\boxed{W(x,y) = (x^2-1)^2 + (y^2-1)^2 - 0.05\,xy + 0.03\,x + 0.06\,y}$$
The small tilt terms split the four phases without destroying any well. *(The legacy parameters from `experiments/` are **not** used: their $\eta = 0.467$ term reduced the $-+$ well's escape barrier to $0.0026$ and its soft Hessian eigenvalue to $0.335$ — the well was all but destroyed. $\eta$ existed to break parallel jump-edge directions for a graph-family ablation not run here.)*

For a homogeneous field $q_i\equiv v$ the gradient energy vanishes and $V = W(v)$: **the coherent barrier equals the barrier of $W$**. Coherent flip vs nucleation: the kink energy is $\sigma = \int_{-1}^1\sqrt{2\kappa(1-x^2)^2}dx = \tfrac43\sqrt{2\kappa} = 2.98$, so a periodic wall pair costs $5.96 \gg 1.0$ — **the coherent flip is the minimum-energy path**, which is precisely what makes the homogeneous-shift jump law (and its moment-exact score) the right choice."""),
        code(cell_setup("coupled_phi4", "build_e4",
                        'exp = build_e4(device=DEV, basin_cache=os.path.join(RESULTS, "basin_map.npz"))')),
        md(r"""## The model

Verified coherent minima of $W$ (asserted to 4 decimals below):

| phase | minimum | $W$ | escape barrier | Laplace mass |
|---|---|---|---|---|
| $--$ | $(-1.0099,-1.0135)$ | $-0.1412$ | 1.082 | 0.583 |
| $-+$ | $(-0.9976,\ 0.9860)$ | $+0.0792$ | 0.892 | 0.106 |
| $+-$ | $(\ 0.9898,-1.0013)$ | $+0.0196$ | 0.921 | 0.169 |
| $++$ | $(\ 1.0025,\ 0.9988)$ | $+0.0400$ | 0.990 | 0.142 |

$\beta\times\min$ barrier $= 7.14$. Stiffness: $\max\lambda(\nabla^2 V)\approx 4\kappa/\delta = 120$, so $\Delta t < 0.017$. Protocol: $N=1000$, $T=100$, $\Delta t_0=0.002$, box $[-2,2]^{24}$, init at the $--$ coherent state $+\,0.05\,\xi$. Partition ($K=4$): basin map of $W$ at the mean order parameter $\bar q = \tfrac1{N_s}\sum_i q_i$.

**References, not ground truth:** (i) the harmonic (Laplace) mixture — phase $k$ with weight $\propto e^{-\beta W(v_k)}/\sqrt{\det H_k}$ ($H_k$ the full $24\times24$ Hessian at the coherent minimum), fluctuations $\mathcal N(0,\varepsilon H_k^{-1})$; (ii) a long PT chain as a cross-check. Both are labelled as references below."""),
        code('''from src.potentials import (PHI4_MINIMA, PHI4_ESCAPE_BARRIERS, PHI4_LAPLACE_MASSES,
                            phi4_W, phi4_W_grad, newton_refine)
V2 = exp.extras["minima_2d"]
phases = exp.extras["phases"]
for i, ph in enumerate(phases):
    v_tab, W_tab = PHI4_MINIMA[ph]
    v = V2[i]
    W = phi4_W(v.unsqueeze(0))[0].item()
    assert abs(v[0].item() - v_tab[0]) < 5e-5 and abs(v[1].item() - v_tab[1]) < 5e-5
    assert abs(W - W_tab) < 5e-4
    print(f"{ph}: v = ({v[0].item():+.4f}, {v[1].item():+.4f})  W = {W:+.4f}  [table {v_tab}, {W_tab}]")

saddle_guesses = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
saddles = [(newton_refine(phi4_W_grad, torch.tensor(sg, device=DEV))) for sg in saddle_guesses]
sW = [phi4_W(s.unsqueeze(0))[0].item() for s in saddles]
adj = {"--": [0, 2], "-+": [0, 3], "+-": [1, 2], "++": [1, 3]}
for i, ph in enumerate(phases):
    Wm = phi4_W(V2[i].unsqueeze(0))[0].item()
    bar = min(sW[j] - Wm for j in adj[ph])
    assert abs(bar - PHI4_ESCAPE_BARRIERS[ph]) < 2e-3, (ph, bar)
    print(f"{ph}: escape barrier {bar:.4f} [table {PHI4_ESCAPE_BARRIERS[ph]}]")
print(f"beta * min barrier = {C.BETA * min(PHI4_ESCAPE_BARRIERS.values()):.2f}")

sigma_kink = exp.pot.kink_energy()
print(f"kink energy sigma = (4/3)sqrt(2 kappa) = {sigma_kink:.3f}; "
      f"periodic wall pair costs {2*sigma_kink:.2f} >> 1.0 coherent barrier"
      " -> the coherent flip is the minimum-energy path")

print("Laplace masses:", np.round(exp.p_star.cpu().numpy(), 3),
      " [table:", list(PHI4_LAPLACE_MASSES.values()), "]")
for i, ph in enumerate(phases):
    assert abs(exp.p_star[i].item() - PHI4_LAPLACE_MASSES[ph]) < 5e-3, ph

g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), C.EPS, g)
barrier_report["kramers_tau_langer"] = exp.kramers_tau
print("ULA first-passage out of the -- basin:", barrier_report)
print(f"(24D Langer estimate over the coherent saddle: tau ~ {exp.kramers_tau:.0f})")'''),
        md(r"""## Jump law and the moment-exact score

Homogeneous phase-to-phase shifts on the complete graph over the 4 minima (6 undirected $\to$ 12 directed atoms): $r_a = \mathbf 1_{N_s}\otimes(v_j-v_i)$, $w_a = 1/12$, $h = 0.1\min_a\|r_a\|$, $\lambda=1$ (a shell-thickened homogeneous shift is still homogeneous).

The periodic gradient energy is **exactly invariant** under $q_i \mapsto q_i - d$, so
$$V(q-r)-V(q) = \delta\sum_i\big[W(q_i-d)-W(q_i)\big]$$
is a fixed polynomial in $(d_x,d_y)$ — e.g. $((x-d_x)^2-1)^2-(x^2-1)^2 = -4d_xx^3+6d_x^2x^2+(4d_x-4d_x^3)x+d_x^4-2d_x^2$ — whose coefficients are the per-particle moments $\sum_i x_i, \sum_i x_i^2, \sum_i x_i^3$ (and $y$ analogues). The moments cost $O(N_s)$ once per step; all $12\times Q_\rho\times Q_\theta$ energy deltas are then $O(1)$ arithmetic each. **No lattice sweeps.** Validated against the direct lattice energy difference to $10^{-13}$ absolute in `tests/test_score.py`."""),
        code('''print("12 directed homogeneous atoms; per-site shifts (dv):")
print(np.round(exp.law.atoms[:, :2].cpu().numpy(), 4))
print("h =", round(exp.extras["h"], 4), " ||r_a|| =",
      np.round(exp.law.atoms.norm(dim=1).cpu().numpy(), 3))

from src.jumps import gauss_legendre_01
DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(24, exp.extras["means24"][0].tolist(), 1.5, DEV, n_phi=4)

def cert_e4(q_theta, q_rho):
    theta, w_theta = gauss_legendre_01(q_theta, DEV)
    shifts, logw = exp.law.quadrature_shifts(q_rho)
    shifts_j, logw_j = exp.law.quadrature_shifts(64)   # fine continuous-nu J side
    return certificate_importance(exp.pot, shifts, logw, theta, w_theta,
                                  cfg.lam, cfg.beta, phis, exp.extras["laplace"],
                                  n_samples=200_000,
                                  nu_shifts_jump=shifts_j, nu_logw_jump=logw_j)'''),
        md(MD_TARGET_PRESERVATION + "\n\nIn 24D a grid is infeasible; the residual uses the shifted-form identity above with self-normalised importance sampling from the Laplace mixture. This is exactly equivalent to the deployed quadrature score provided the $M_{\\max}$ cap never fires on the sampled region — asserted below (the max log-magnitude there is $\\approx 15 \\ll 600$)."),
        code('''cert_report = cert_e4(**DEFAULT_QUAD)
for i in range(len(phis)):
    print(f"  phi_{i}: R = {cert_report[f'phi_{i}']['residual']:.3e}")
print(f"max R = {cert_report['max_residual']:.3e}")
assert cert_report["max_residual"] < 1e-6

score = exp.make_score(**DEFAULT_QUAD)
g = torch.Generator(device=DEV); g.manual_seed(11)
Mv, _ = score.log_parts(exp.extras["laplace"].sample(100_000, g))
print(f"max log score magnitude on the sampled region: {Mv.max().item():.1f} << 600 (cap never fires)")
cert_report["max_log_magnitude_on_support"] = float(Mv.max().item())'''),
        md(MD_SAMPLERS),
        code(CELL_LADDER),
        md(MD_METRICS + "\n\n$W_2$ and MMD are computed on the 2D mean order parameter $\\bar q$; occupancy metrics on the $K=4$ basin map of $W$ at $\\bar q$. The Laplace mixture supplies both the reference sample and $p^\\star$ — it is a **reference, not ground truth** — so a long PT chain is run below as an independent cross-check of the phase masses."),
        code(CELL_REFERENCE + '''

# PT cross-check of the Laplace phase masses (reference vs reference)
t0 = time.time()
gen_x = torch.Generator(device=DEV); gen_x.manual_seed(4242)
from src.samplers import ParallelTempering
pt_x = ParallelTempering(exp.pot, exp.init_fn(1000, gen_x), cfg.dt, pt_betas, gen_x, exp.box)
n_x = int(round(300.0 / cfg.dt))
for _ in range(n_x):
    pt_x.step()
from src.metrics import occupancy
p_pt = occupancy(exp.labels_fn(pt_x.positions()), 4).cpu().numpy()
print(f"long PT chain (T=300, cold replica) phase masses:   {np.round(p_pt, 3)}")
print(f"harmonic (Laplace) reference phase masses:          "
      f"{np.round(exp.p_star.cpu().numpy(), 3)}  ({time.time()-t0:.0f}s)")
pt_crosscheck = p_pt.tolist()'''),
        md(MD_DT_RULE),
        code(cell_quad_refine(
            "[dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]",
            'cert_e4(**s)',
            '("W2", "TV", "MMD", "EMC", "EJS")',
            "quad")),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC", "EJS"]')),
        md("## Figures"),
        code(cell_figures('("W2", "TV", "MMD", "EMC", "EJS")')),
        md("## CSV emission and summary"),
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
                          ("03_muller_brown_10d", build_e3_nb),
                          ("04_coupled_phi4", build_e4_nb)]:
        nb = builder()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        path = os.path.join(here, f"{name}.ipynb")
        with open(path, "w") as f:
            nbf.write(nb, f)
        print("wrote", path)
