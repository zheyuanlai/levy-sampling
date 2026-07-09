"""Sampler validation (spec section 9.3)."""
import math

import torch

from src.config import Q_RHO
from src.potentials import DoubleWell1D, MoG40
from src.jumps import AnnulusJumpLaw, ShellJumpLaw, apply_poisson_jumps
from src.samplers import (BAOAB, MALA, CompoundPoisson, ParallelTempering,
                          RectBox)
from src.score import ShellScore

DEV = "cuda"


class _Quadratic:
    """V(x) = x^2/2 in 1D, for exactness tests."""
    d = 1

    def __init__(self):
        self.n_V = self.n_grad = self.n_Vdelta = 0

    def V(self, x):
        return 0.5 * (x[..., 0] ** 2)

    def grad(self, x):
        return x.clone()


def test_jump_law_mc_vs_quadrature():
    """The nu used by the score/certificate quadrature and the nu used to
    generate jumps must be the SAME measure: E_nu[phi(x+R) - phi(x)] by MC
    (4e5 draws) vs quadrature to < 1e-3; probability weights sum to 1."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)

    def phi(y):
        # gentle slope keeps Var(phi) small enough that 4e5 MC draws resolve
        # the 1e-3 tolerance (se ~ 1e-3 at slope 0.4 would drown the check)
        return torch.tanh(0.1 * y[:, 0] + 0.05 * y.sum(dim=1))

    # E1-style shell law
    law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], device=DEV),
                       torch.tensor([0.5, 0.5], device=DEV), 0.2)
    shifts, logw = law.quadrature_shifts(Q_RHO)
    assert abs(torch.exp(logw).sum().item() - 1.0) < 1e-12
    assert abs(law.weights.sum().item() - 1.0) < 1e-12
    x = torch.tensor([[0.3]], device=DEV)
    mc = phi(x + law.sample(400_000, gen)).mean().item()
    quad = (torch.exp(logw) * phi(x + shifts)).sum().item()
    assert abs(mc - quad) < 1e-3, (mc, quad)

    # E2 annulus law
    law2 = AnnulusJumpLaw(4.0, 15.0, DEV)
    shifts2, logw2 = law2.quadrature_shifts(16, 64)
    assert abs(torch.exp(logw2).sum().item() - 1.0) < 1e-12
    x2 = torch.tensor([[1.0, -2.0]], device=DEV)
    mc2 = phi(x2 + law2.sample(400_000, gen)).mean().item()
    quad2 = (torch.exp(logw2) * phi(x2 + shifts2)).sum().item()
    assert abs(mc2 - quad2) < 1e-3, (mc2, quad2)


def test_mala_acceptance_to_one():
    pot = DoubleWell1D()
    box = RectBox([-3.0], [3.0], DEV)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    x0 = -1.0 + 0.1 * torch.randn(4096, 1, generator=gen, device=DEV)
    rates = []
    for dt in (5e-3, 5e-4, 5e-5):
        g = torch.Generator(device=DEV)
        g.manual_seed(2)
        m = MALA(pot, x0, dt, 8.0, g, box)
        for _ in range(100):
            m.step()
        rates.append(m.pop_diagnostics()["mala_accept"])
    assert rates[0] < rates[1] < rates[2]
    assert rates[2] > 0.999, rates


def test_baoab_o_step_variance():
    """O-step: p <- e^{-gamma dt} p + sqrt(eps (1 - e^{-2 gamma dt})) xi is
    the exact OU solution; its coefficient and the stationarity of the
    momentum marginal N(0, eps) are both checked."""
    pot = _Quadratic()
    box = RectBox([-50.0], [50.0], DEV)
    dt, gamma, eps = 0.01, 1.0, 0.125
    gen = torch.Generator(device=DEV)
    gen.manual_seed(3)
    # start from the joint equilibrium (q, p) ~ N(0, eps) x N(0, eps) for
    # V = x^2/2, so the momentum marginal is a stationarity check
    x0 = math.sqrt(eps) * torch.randn(200_000, 1, generator=gen, device=DEV)
    b = BAOAB(pot, x0, dt, eps, gen, box, gamma=gamma)
    assert abs(b.c2 ** 2 - eps * (1.0 - math.exp(-2 * gamma * dt))) < 1e-15
    # one O-step from p = 0 has variance exactly c2^2
    p0 = torch.zeros(200_000, device=DEV)
    xi = torch.randn(200_000, generator=gen, device=DEV)
    var = (b.c1 * p0 + b.c2 * xi).var().item()
    assert abs(var / (eps * (1 - math.exp(-2 * gamma * dt))) - 1.0) < 2e-2
    # equilibrium marginals stay put over many steps (O(dt^2) bias + MC noise)
    for _ in range(500):
        b.step()
    assert abs(b.p.var().item() / eps - 1.0) < 2e-2
    assert abs(b.x.var().item() / eps - 1.0) < 2e-2


def test_pt_swap_preserves_product_target():
    """Swap-only PT chain on V = x^2/2: the swap is a deterministic
    involution with the exact MH ratio, so the product prod_k pi_k with
    pi_k ~ N(0, 1/beta_k) must be preserved."""
    pot = _Quadratic()
    box = RectBox([-50.0], [50.0], DEV)
    betas = torch.tensor([8.0, 2.0, 0.5], device=DEV)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(4)
    n = 200_000
    x0 = torch.zeros(n, 1, device=DEV)
    pt = ParallelTempering(pot, x0, dt=0.01, betas=betas, gen=gen, box=box)
    # overwrite replica states with exact samples from their targets
    for k in range(3):
        pt.x[k] = (torch.randn(n, 1, generator=gen, device=DEV)
                   / math.sqrt(betas[k].item()))
    pt.Vx = pot.V(pt.x)
    pt.gx = pot.grad(pt.x)
    for it in range(50):
        pt._swap_pass(it % 2)
    for k in range(3):
        v = pt.x[k].var().item()
        assert abs(v * betas[k].item() - 1.0) < 2e-2, (k, v)
    d = pt.pop_diagnostics()
    assert 0.0 < d["pt_swap_accept"] < 1.0


def test_cp_lsc_jump_streams_pathwise_identical():
    """Raw CP and LSC-CP consume the SAME dedicated jump generator in the
    same order every step, so their jump times and increments are pathwise
    identical (not merely equal in law)."""
    pot = DoubleWell1D()
    law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], device=DEV),
                       torch.tensor([0.5, 0.5], device=DEV), 0.2)
    box = RectBox([-3.0], [3.0], DEV)
    score = ShellScore(pot, law, 1.0, 8.0, 16, 8)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(5)
    x0 = -1.0 + 0.05 * torch.randn(512, 1, generator=gen, device=DEV)

    def mk(with_score, seed_jump=99):
        gd = torch.Generator(device=DEV)
        gd.manual_seed(6 if with_score else 7)   # different diffusion streams
        gj = torch.Generator(device=DEV)
        gj.manual_seed(seed_jump)                # SAME jump stream
        return CompoundPoisson(pot, x0, 0.005, 0.125, 1.0, law, gd, gj, box,
                               score=score if with_score else None)

    cp, lsc = mk(False), mk(True)
    for _ in range(60):
        cp.step()
        lsc.step()
        assert torch.equal(cp.gen_jump.get_state(), lsc.gen_jump.get_state())

    # and the jump application itself is deterministic given the generator
    g1 = torch.Generator(device=DEV); g1.manual_seed(8)
    g2 = torch.Generator(device=DEV); g2.manual_seed(8)
    z = torch.zeros(4096, 1, device=DEV)
    y1, c1 = apply_poisson_jumps(z, law, 1.0, 0.5, g1)
    y2, c2 = apply_poisson_jumps(z, law, 1.0, 0.5, g2)
    assert torch.equal(y1, y2) and torch.equal(c1, c2)
    assert c1.max().item() >= 1.0                # jumps actually fire
