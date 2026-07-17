"""NFE ledger: per-particle-per-step V / grad / V_delta counts match the
declared cost model, and the no_count context excludes enclosed evals.

Ledger (per particle per step):
  ULA / FLA / BAOAB / raw-CP / raw-CP-RA : 1 grad
  MALA                                   : 1 grad + 1 V
  exact LSC-CP                           : 1 grad + q_theta*A*q_rho V_delta
  RA LSC-CP                              : 1 grad + q_theta V_delta
  paired multi-atom LSC-CP                : 1 grad + A*q_theta V_delta
  PT                                     : K grad + K V (all replicas)
"""
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.config import Q_THETA, Q_RHO
from src.experiments import build_e1, make_sampler_factory

DEV = "cuda"


def _counts_one_step(exp, factory, method, N):
    s = factory(method, 0)
    exp.pot.reset_counters()          # baseline AFTER construction
    s.step()
    return (exp.pot.n_grad / N, exp.pot.n_V / N, exp.pot.n_Vdelta / N)


def test_nfe_ledger_per_step():
    exp = build_e1(device=DEV)
    N = 128
    pt_dummy = torch.tensor([exp.cfg.beta], device=DEV)
    f = make_sampler_factory(exp, exp.cfg.dt, pt_dummy, n_particles=N)
    A = exp.law.A
    expect = {
        "ULA":       (1, 0, 0),
        "FLA":       (1, 0, 0),
        "BAOAB":     (1, 0, 0),
        "MALA":      (1, 1, 0),
        "CP":        (1, 0, 0),
        "CP-RA":     (1, 0, 0),
        "LSC-CP":    (1, 0, Q_THETA * A * Q_RHO),
        "LSC-CP-RA": (1, 0, Q_THETA),
        "LSC-CP-MA": (1, 0, Q_THETA * A),
    }
    for method, (eg, ev, eq) in expect.items():
        g, v, q = _counts_one_step(exp, f, method, N)
        assert (g, v, q) == (eg, ev, eq), (method, (g, v, q), (eg, ev, eq))


def test_no_count_excludes_evals():
    exp = build_e1(device=DEV)
    x = torch.zeros(10, 1, dtype=torch.float64, device=DEV)
    exp.pot.reset_counters()
    exp.pot.V(x); exp.pot.grad(x)
    base = exp.pot.nfe()
    assert base == 20                       # 10 V + 10 grad
    with exp.pot.no_count():
        exp.pot.V(x); exp.pot.grad(x); exp.pot.V(x)
    assert exp.pot.nfe() == base            # enclosed evals excluded
