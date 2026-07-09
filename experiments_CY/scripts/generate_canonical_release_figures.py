"""Generate canonical theory-facing figures and release metadata."""

from __future__ import annotations

import math
import re
import shutil
import zlib
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[4]
MANUSCRIPT = REPO / "manuscript_clean_active"
NUM_ROOT = MANUSCRIPT / "numerics" / "four_experiment_release"
COMMON = NUM_ROOT / "common"
if str(COMMON) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(COMMON))
from levy.plot_style import (  # noqa: E402
    METHOD_STYLES,
    PHASE_COLORS,
    REFERENCE_STYLES,
    apply_plot_style,
    clean_axes,
    method_color,
    method_label,
    method_marker,
    panel_label,
    style_registry_rows,
)
from levy.spectral_references import (  # noqa: E402
    add_spectral_reference_lines,
    write_reference_registry,
)
from levy.doublewell_scaling import generate_phase17l_figure  # noqa: E402
from levy.doublewell_definitive import generate_main_figure  # noqa: E402

METHOD_COLORS = {name: style.color for name, style in METHOD_STYLES.items()}

TABLES = NUM_ROOT / "tables"
FIG_ROOT = MANUSCRIPT / "figures" / "four_experiment_release"
MAIN = FIG_ROOT / "main_candidates"
APP = FIG_ROOT / "appendix_candidates"
DIAG = FIG_ROOT / "diagnostics"
VALIDATION = TABLES / "canonical_release"

PHASE_ORDER = ["--", "-+", "+-", "++"]
GL_ORDER = ["Langevin", "LSC-MST-shell", "LSC-cycle-shell", "LSC-5-shell", "LSC-complete-shell"]
GL_SHORT = {
    "Langevin": "Langevin",
    "LSC-MST-shell": "MST",
    "LSC-cycle-shell": "cycle",
    "LSC-5-shell": "5-edge",
    "LSC-complete-shell": "complete",
}


def clean(ax, grid: bool = True) -> None:
    clean_axes(ax, grid=grid)


def label(ax, text: str) -> None:
    panel_label(ax, text)


def save(fig, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def setup() -> None:
    for path in (MAIN, APP, DIAG, VALIDATION):
        path.mkdir(parents=True, exist_ok=True)
    apply_plot_style(plt)


def fixed_log_amplitude(eps: np.ndarray, y: np.ndarray, shape: np.ndarray) -> float:
    mask = np.isfinite(eps) & np.isfinite(y) & np.isfinite(shape) & (y > 0) & (shape > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.exp(np.mean(np.log(y[mask]) - np.log(shape[mask]))))


def r2_log(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    ly = np.log(y[mask])
    lh = np.log(yhat[mask])
    ss_res = float(np.sum((ly - lh) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def weights_for_grid(x: np.ndarray) -> np.ndarray:
    w = np.empty_like(x, dtype=float)
    dx = np.diff(x)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    return w


def normalize_pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y / float(np.sum(y * weights_for_grid(x)))


def central_mask_from_pdf(
    x: np.ndarray, p: np.ndarray, mass: float = 0.99
) -> tuple[np.ndarray, float, float]:
    cdf = np.cumsum(p * weights_for_grid(x))
    cdf /= cdf[-1]
    lo = float(np.interp((1.0 - mass) / 2.0, cdf, x))
    hi = float(np.interp(1.0 - (1.0 - mass) / 2.0, cdf, x))
    return (x >= lo) & (x <= hi), lo, hi


def doublewell_pdf(x: np.ndarray, eps: float = 0.125) -> np.ndarray:
    potential = 0.25 * x**4 - 0.5 * x**2
    return normalize_pdf(x, np.exp(-(potential - np.min(potential)) / eps))


def robust_limits(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.quantile(finite, [0.01, 0.99])
    if lo == hi:
        span = max(1.0, abs(float(lo)))
        return float(lo - span), float(hi + span)
    span = float(hi - lo)
    return float(lo - pad * span), float(hi + pad * span)


FIGURE_SYNC = {
    "doublewell": [
        ("doublewell_fig01_target_cdf_nu.pdf", MAIN),
        ("doublewell_fig03_mixing_with_form_and_abscissa_references.pdf", APP),
        ("doublewell_fig04d_rich_metrics_weak_errors.pdf", APP),
        ("doublewell_fig04_final_pdf_cdf_metrics_particles.pdf", APP),
        ("doublewell_fig07_jump_event_diagnostics.pdf", APP),
    ],
    "threewell": [
        ("threewell_fig01_target_cdf_nu.pdf", MAIN),
        ("threewell_fig03_mode_mixing_with_form_and_abscissa_references.pdf", APP),
        ("threewell_fig04d_rich_metrics_mode_coverage_weak_errors.pdf", APP),
        ("threewell_fig01b_closed_form_levy_score_comparison.pdf", APP),
        ("threewell_fig07_jump_event_diagnostics.pdf", APP),
    ],
    "muller10d": [
        ("muller10d_fig00_muller10d_target_transform_jump_geometry.pdf", MAIN),
        ("muller10d_fig01_muller10d_latent_particles_occupancy.pdf", MAIN),
        ("muller10d_fig02_muller10d_basin_communication.pdf", MAIN),
        ("muller10d_fig01b_muller10d_latent_transition_matrices.pdf", APP),
        ("muller10d_fig03_muller10d_target_compatibility.pdf", APP),
        ("muller10d_fig04b_muller10d_auxiliary_distribution_checks.pdf", APP),
        ("muller10d_fig05_muller10d_weak_observable_errors.pdf", APP),
    ],
    "vector_gl": [
        ("vector_gl_fig04a_coupled_phi4_graph_families.pdf", MAIN),
        ("vector_gl_fig02_coupled_phi4_phase_communication.pdf", APP),
        ("vector_gl_fig05_vector_landau_final_distributions.pdf", APP),
        ("vector_gl_fig06_vector_landau_recorded_phase_transition_matrices.pdf", APP),
        ("vector_gl_fig07_vector_landau_sampler_diagnostics.pdf", APP),
        ("vector_gl_fig08_vector_landau_final_profile_heatmaps.pdf", APP),
        ("vector_gl_fig09_coupled_phi4_physical_diagnostics.pdf", APP),
    ],
}


def sync_experiment_figures(experiment: str) -> None:
    for name, destination in FIGURE_SYNC[experiment]:
        source = DIAG / name
        if not source.exists():
            raise FileNotFoundError(source)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / name)


def fit_row(fits: pd.DataFrame, method: str, quantity: str, family: str) -> Optional[pd.Series]:
    sub = fits[(fits.method == method) & (fits.quantity == quantity) & (fits.fit_family == family)]
    return None if sub.empty else sub.iloc[0]


def model_curve(eps: np.ndarray, row: Optional[pd.Series], quantity: str) -> Optional[np.ndarray]:
    if row is None:
        return None
    pref = float(row.get("prefactor", np.nan))
    if not np.isfinite(pref):
        return None
    if row["fit_family"] == "arrhenius":
        barrier = float(row.get("barrier", np.nan))
        if not np.isfinite(barrier):
            return None
        if quantity == "form_gap":
            return pref * np.exp(-barrier / eps)
        return pref * np.exp(barrier / eps)
    exponent = float(row.get("exponent", np.nan))
    if not np.isfinite(exponent):
        return None
    if quantity == "form_gap":
        return pref * eps**exponent
    return pref * eps ** (-exponent)


def generate_doublewell_sweep() -> None:
    phase17m_summary = VALIDATION / "doublewell_phase17m_mixing_times_summary.csv"
    phase17l_summary = VALIDATION / "doublewell_phase17l_mixing_times_summary.csv"
    if phase17m_summary.exists():
        estimator = pd.read_csv(
            VALIDATION / "doublewell_phase17m_estimator_selection.csv"
        )
        selected_metric = estimator.loc[
            estimator["selected_for_main"].astype(bool), "metric"
        ].iloc[0]
        generate_main_figure(VALIDATION, MAIN, selected_metric)
        return
    elif phase17l_summary.exists():
        generate_phase17l_figure(VALIDATION, MAIN)
        return
    summary = pd.read_csv(TABLES / "01_double_well" / "doublewell_epsilon_sweep_summary.csv")
    fits = pd.read_csv(TABLES / "01_double_well" / "doublewell_epsilon_sweep_model_fits.csv")
    eps_grid = np.linspace(summary.eps.min(), summary.eps.max(), 240)
    inv_grid = 1.0 / eps_grid
    methods = ["Langevin", "LSC-CP"]
    delta_v = 0.25
    reference_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.7), constrained_layout=True)
    ax = axes.ravel()

    for method in methods:
        sub = summary[summary.method == method].sort_values("eps")
        color = method_color(method)
        inv = 1.0 / sub.eps.to_numpy()

        ax[0].semilogy(
            inv,
            sub.form_gap,
            marker=method_marker(method),
            color=color,
            label=method_label(method),
        )
        gap_candidates = fits[(fits.method == method) & (fits.quantity == "form_gap")]
        fit = gap_candidates.sort_values("r2_log", ascending=False).iloc[0] if not gap_candidates.empty else None
        curve = model_curve(eps_grid, fit, "form_gap")
        if curve is not None:
            ax[0].semilogy(inv_grid, curve, color=color, **REFERENCE_STYLES["empirical_fit"])

        ax[2].semilogy(
            inv,
            sub.mixing_time,
            marker=method_marker(method),
            color=color,
            label=method_label(method),
        )
        mix_candidates = fits[(fits.method == method) & (fits.quantity == "mixing_time")]
        fit = mix_candidates.sort_values("r2_log", ascending=False).iloc[0] if not mix_candidates.empty else None
        curve = model_curve(eps_grid, fit, "mixing_time")
        if curve is not None:
            ax[2].semilogy(inv_grid, curve, color=color, **REFERENCE_STYLES["empirical_fit"])

    lsub = summary[(summary.method == "Langevin") & summary.chi2_mixing_time_reached].sort_values("eps")
    ssub = summary[(summary.method == "LSC-CP") & summary.chi2_mixing_time_reached].sort_values("eps")

    reached = {"Langevin": lsub, "LSC-CP": ssub}
    theory_specs = {
        "Langevin": ("fixed_arrhenius", "exp(DeltaV / eps)", delta_v),
        "LSC-CP": ("fixed_theorem_inverse_channel", "eps^(-1/2)", np.nan),
    }
    for method in methods:
        sub = reached[method]
        color = method_color(method)
        ax[1].semilogy(
            1.0 / sub.eps,
            sub.chi2_mixing_time,
            marker=method_marker(method),
            color=color,
            label=f"{method_label(method)} data",
            **REFERENCE_STYLES["data"],
        )
        series_rows.append(
            {
                "panel": "b",
                "method": method,
                "series_role": "data",
                "family": "observed reached chi-square mixing times",
                "color": color,
                "linestyle": "none",
                "marker": method_marker(method),
            }
        )

        if method == "Langevin":
            theory_shape_data = np.exp(delta_v / sub.eps.to_numpy())
            theory_shape_grid = np.exp(delta_v / eps_grid)
            theory_label = r"Langevin theory $C_L e^{\Delta V/\varepsilon}$"
        else:
            theory_shape_data = sub.eps.to_numpy() ** (-0.5)
            theory_shape_grid = eps_grid ** (-0.5)
            theory_label = r"LSC-CP theory $C_S\varepsilon^{-1/2}$"
        amplitude = fixed_log_amplitude(
            sub.eps.to_numpy(), sub.chi2_mixing_time.to_numpy(), theory_shape_data
        )
        ax[1].semilogy(
            inv_grid,
            amplitude * theory_shape_grid,
            color=color,
            label=theory_label,
            **REFERENCE_STYLES["theory"],
        )
        family, shape, barrier = theory_specs[method]
        reference_rows.append(
            {
                "method": method,
                "quantity": "chi2_mixing_time",
                "reference_family": family,
                "shape": shape,
                "DeltaV": barrier,
                "amplitude_rule": "least-squares log amplitude over reached chi-square mixing times",
                "amplitude": amplitude,
                "r2_log": r2_log(
                    sub.chi2_mixing_time.to_numpy(), amplitude * theory_shape_data
                ),
                "n": len(sub),
            }
        )
        series_rows.append(
            {
                "panel": "b",
                "method": method,
                "series_role": "theory_reference",
                "family": family,
                "color": color,
                "linestyle": REFERENCE_STYLES["theory"]["linestyle"],
                "marker": "",
            }
        )

        candidates = fits[(fits.method == method) & (fits.quantity == "chi2_mixing_time")].copy()
        candidates = candidates.sort_values("r2_log", ascending=False)
        selected_family = str(candidates.iloc[0]["fit_family"])
        for _, row in candidates.iterrows():
            fit_rows.append(
                {
                    **row.to_dict(),
                    "selected_for_display": str(row["fit_family"]) == selected_family,
                    "selection_rule": "maximum log-R2 among Arrhenius and polynomial candidates",
                }
            )
        selected = candidates.iloc[0]
        empirical_curve = model_curve(eps_grid, selected, "chi2_mixing_time")
        if empirical_curve is not None:
            if selected_family == "arrhenius":
                parameter = rf"$\widehat\Delta={float(selected['barrier']):.3f}$"
            else:
                parameter = rf"$\widehat\alpha={float(selected['exponent']):.3f}$"
            ax[1].semilogy(
                inv_grid,
                empirical_curve,
                color=color,
                label=f"{method_label(method)} empirical {selected_family} ({parameter})",
                **REFERENCE_STYLES["empirical_fit"],
            )
        series_rows.append(
            {
                "panel": "b",
                "method": method,
                "series_role": "empirical_fit",
                "family": selected_family,
                "color": color,
                "linestyle": REFERENCE_STYLES["empirical_fit"]["linestyle"],
                "marker": "",
            }
        )

    ax[0].set_title("diagonal-form gap")
    ax[0].set_ylabel("rate")
    ax[1].set_title("chi-square mixing time: data, theory, and empirical fits")
    ax[1].set_ylabel("time")
    ax[2].set_title("well-TV communication time")
    ax[2].set_ylabel("time")
    fit_frame = pd.DataFrame(fit_rows)
    xloc = np.arange(len(methods))
    width = 0.34
    arr = fit_frame[fit_frame.fit_family == "arrhenius"].set_index("method").reindex(methods)
    poly = fit_frame[fit_frame.fit_family == "polynomial"].set_index("method").reindex(methods)
    ax[3].bar(xloc - width / 2, arr.r2_log, width=width, color="#718096", label="Arrhenius")
    ax[3].bar(xloc + width / 2, poly.r2_log, width=width, color="#B794F4", label="polynomial")
    ax[3].set_xticks(xloc)
    ax[3].set_xticklabels(methods)
    ax[3].set_ylim(0, 1.05)
    ax[3].set_title("empirical candidate fit quality")
    ax[3].set_ylabel(r"log-$R^2$")
    ax[3].legend(frameon=False)
    for j, axi in enumerate(ax):
        if j < 3:
            axi.set_xlabel(r"$1/\varepsilon$")
        clean(axi)
        label(axi, "abcd"[j])
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].legend(frameon=False, fontsize=6.6, ncol=2)
    save(fig, MAIN, "doublewell_theory_epsilon_chi2_sweep")
    pd.DataFrame(reference_rows).to_csv(
        VALIDATION / "doublewell_chi2_theory_references.csv", index=False
    )
    fit_frame.to_csv(VALIDATION / "doublewell_chi2_empirical_fit_candidates.csv", index=False)
    pd.DataFrame(series_rows).to_csv(
        VALIDATION / "doublewell_chi2_six_series_design.csv", index=False
    )


def triplewell_pdf(x: np.ndarray) -> np.ndarray:
    weights = np.array([5.0 / 21.0, 3.0 / 7.0, 1.0 / 3.0])
    means = np.array([-3.0, 0.0, 3.0])
    sigmas = np.array([0.5, 0.75, 0.5])
    y = np.zeros_like(x, dtype=float)
    for wt, mu, sig in zip(weights, means, sigmas):
        y += wt * np.exp(-0.5 * ((x - mu) / sig) ** 2) / (math.sqrt(2.0 * math.pi) * sig)
    return y / np.trapz(y, x)


def generate_one_d_drift_figure() -> None:
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.8), constrained_layout=True)
    specs = [
        ("double-well", TABLES / "01_double_well" / "doublewell_drift_grid.csv", doublewell_pdf),
        ("triple-well", TABLES / "02_triple_well" / "threewell_drift_grid.csv", triplewell_pdf),
    ]
    for row_index, (experiment, path, pdf_fn) in enumerate(specs):
        frame = pd.read_csv(path)
        unique_x = np.sort(frame["x"].unique().astype(float))
        density = pdf_fn(unique_x)
        central, left, right = central_mask_from_pdf(unique_x, density, 0.99)
        central_x = unique_x[central]
        weights = weights_for_grid(central_x)
        central_density = pdf_fn(central_x)
        central_density /= np.sum(central_density * weights)
        if "method" not in frame:
            frame["method"] = "LSC-CP"
        display_frame = frame[(frame["x"] >= left) & (frame["x"] <= right)].copy()

        local_ax, score_ax, total_ax = axes[row_index]
        first_method = str(display_frame["method"].iloc[0])
        for method, sub in display_frame.groupby("method"):
            sub = sub.sort_values("x")
            color = method_color(method)
            if str(method) == first_method:
                local_ax.plot(
                    sub["x"],
                    sub["local_Langevin_drift"],
                    color=method_color("Langevin"),
                    label="local Langevin",
                )
            score_ax.plot(sub["x"], sub["Levy_score_correction"], color=color, label=method_label(method))
            total_ax.plot(sub["x"], sub["total_LSC_drift"], color=color, label=method_label(method))
            score = np.interp(
                central_x,
                sub["x"].to_numpy(dtype=float),
                sub["Levy_score_correction"].to_numpy(dtype=float),
            )
            finite = score[np.isfinite(score)]
            full = frame[frame["method"].eq(method)]["Levy_score_correction"].to_numpy(dtype=float)
            full = full[np.isfinite(full)]
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "central_mass": 0.99,
                    "central_x_left": left,
                    "central_x_right": right,
                    "central_score_max_abs": float(np.max(np.abs(finite))),
                    "target_weighted_score_rms": float(
                        math.sqrt(np.sum(score**2 * central_density * weights))
                    ),
                    "full_grid_score_max_abs": float(np.max(np.abs(full))),
                }
            )

        for axis, values in [
            (local_ax, display_frame["local_Langevin_drift"]),
            (score_ax, display_frame["Levy_score_correction"]),
            (total_ax, display_frame["total_LSC_drift"]),
        ]:
            axis.set_xlim(left, right)
            axis.set_ylim(*robust_limits(values.to_numpy(dtype=float)))
            axis.axhline(0.0, color="0.15", linewidth=0.7)
            clean(axis)
            axis.set_xlabel("x")
        local_ax.set_title(f"{experiment}: local drift")
        score_ax.set_title(f"{experiment}: score correction")
        total_ax.set_title(f"{experiment}: total corrected drift")
        local_ax.set_ylabel("signed drift")
        score_ax.legend(frameon=False)
        total_ax.legend(frameon=False)
    for index, axis in enumerate(axes.ravel()):
        label(axis, "abcdef"[index])
    save(fig, MAIN, "one_d_target_relevant_drift_fields")
    pd.DataFrame(rows).to_csv(VALIDATION / "one_d_drift_score_scale_audit.csv", index=False)


def generate_triplewell_graph_figure() -> None:
    metrics = pd.read_csv(TABLES / "02_triple_well" / "threewell_metrics_timeseries.csv")
    density = pd.read_csv(TABLES / "02_triple_well" / "threewell_density_convergence_summary.csv")
    terminal = pd.read_csv(TABLES / "02_triple_well" / "threewell_terminal_metrics.csv")
    rates = pd.read_csv(TABLES / "02_triple_well" / "threewell_rate_comparison.csv")
    gen = pd.read_csv(TABLES / "02_triple_well" / "threewell_generator_main_diagnostics.csv")
    chosen = ["Langevin", "LSC-adjacent", "LSC-overlong", "CP-overlong"]
    form_rates = rates.set_index("method")["form_gap"].to_dict()
    abscissa_rates = gen.set_index("method")["generator_abscissa"].to_dict()
    reference_records: list[dict[str, object]] = []

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.5), constrained_layout=True)
    ax = axes.ravel()
    x = np.linspace(-5.2, 5.2, 800)
    p = triplewell_pdf(x)
    ax[0].plot(x, p, color="0.18", lw=2.0, label="target density")
    for boundary in [-1.5, 1.5]:
        ax[0].axvline(boundary, color="0.15", lw=0.9, linestyle="--")
    ax[0].annotate("adjacent", xy=(-3.0, 0.12), xytext=(0.0, 0.12),
                   arrowprops=dict(arrowstyle="<->", color=METHOD_COLORS["LSC-adjacent"]),
                   color=METHOD_COLORS["LSC-adjacent"], ha="center")
    ax[0].annotate("overlong", xy=(-3.0, 0.07), xytext=(3.0, 0.07),
                   arrowprops=dict(arrowstyle="<->", color=METHOD_COLORS["LSC-overlong"]),
                   color=METHOD_COLORS["LSC-overlong"], ha="center")
    ax[0].set_title("target modes and jump support")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("density")

    gap_rows = rates[rates["form_gap"].notna()].copy()
    gap_rows = gap_rows[gap_rows["method"].isin(["Langevin", "LSC-adjacent", "LSC-overlong"])]
    ax[1].bar(gap_rows["method"], gap_rows["form_gap"], color=[METHOD_COLORS[m] for m in gap_rows["method"]])
    ax[1].set_yscale("log")
    ax[1].set_title("certified diagonal-form gap")
    ax[1].set_ylabel("form gap")
    ax[1].tick_params(axis="x", rotation=20)

    for method in chosen:
        sub = metrics[metrics.method == method].groupby("time")[["mode_TV", "middle_error"]].mean().reset_index()
        if sub.empty:
            continue
        ax[2].semilogy(sub.time, np.maximum(sub.mode_TV, 1e-12), color=METHOD_COLORS.get(method), lw=1.7, label=method)
        ax[3].semilogy(sub.time, np.maximum(np.abs(sub.middle_error), 1e-12), color=METHOD_COLORS.get(method), lw=1.7, label=method)
    for metric, axis_index in [("mode_TV", 2), ("middle_error", 3)]:
        metric_frame = metrics[["method", "time", metric]].copy()
        metric_frame[metric] = np.abs(metric_frame[metric])
        for method in ["Langevin", "LSC-adjacent", "LSC-overlong"]:
            add_spectral_reference_lines(
                ax[axis_index],
                metric_frame,
                metric,
                method,
                form_rate=form_rates.get(method),
                abscissa_rate=abscissa_rates.get(method),
                records=reference_records,
                experiment="triple-well",
                figure="threewell_theory_graph_weighted_density.pdf",
            )
    for method in chosen:
        sub = density[density.method == method]
        if sub.empty:
            continue
        ax[4].semilogy(sub.time, np.maximum(sub.truncated_KDE_chi2_mean, 1e-12),
                       color=METHOD_COLORS.get(method), lw=1.7, label=method)
    density_refs = density[["method", "time", "truncated_KDE_chi2_mean"]].rename(
        columns={"truncated_KDE_chi2_mean": "truncated_KDE_chi2"}
    )
    for method in ["Langevin", "LSC-adjacent", "LSC-overlong"]:
        add_spectral_reference_lines(
            ax[4],
            density_refs,
            "truncated_KDE_chi2",
            method,
            form_rate=form_rates.get(method),
            abscissa_rate=abscissa_rates.get(method),
            records=reference_records,
            experiment="triple-well",
            figure="threewell_theory_graph_weighted_density.pdf",
        )

    term = terminal[terminal.method.isin(chosen)].set_index("method").loc[chosen].reset_index()
    xloc = np.arange(len(term))
    ax[5].bar(xloc - 0.18, term["mode_TV_mean"], width=0.36, color="#4C78A8", label="mode-TV")
    ax[5].bar(xloc + 0.18, np.abs(term["middle_error_mean"]), width=0.36, color="#F58518", label="middle error")
    ax[5].set_xticks(xloc)
    ax[5].set_xticklabels([m.replace("LSC-", "") for m in term.method], rotation=20, ha="right")
    ax[5].set_title("terminal coarse-mass errors")
    ax[5].set_ylabel("absolute error")
    ax[5].legend(frameon=False, fontsize=8)

    ax[2].set_title("mode-TV communication")
    ax[2].set_ylabel("mode TV")
    ax[3].set_title("middle-mode mass error")
    ax[3].set_ylabel("absolute error")
    ax[4].set_title("truncated KDE chi-square")
    ax[4].set_ylabel(r"$\widehat{\chi}^2_{\rm trunc}$")
    for axi in ax[2:5]:
        axi.set_xlabel("time")
        axi.legend(frameon=False, fontsize=6.8)
    for j, axi in enumerate(ax):
        clean(axi)
        label(axi, "abcdef"[j])
    save(fig, MAIN, "threewell_theory_graph_weighted_density")

    cert = gap_rows.merge(gen, on="method", how="left", suffixes=("", "_generator"))
    cert["abscissa_main_text_decision"] = "shown as a labeled finite-dimensional rate reference; not used as a theorem bound"
    cert["form_gap_decision"] = "used as certified diagonal-form quantity"
    cert.to_csv(VALIDATION / "triplewell_form_and_abscissa_certificate.csv", index=False)
    write_reference_registry(
        reference_records, VALIDATION / "triplewell_main_spectral_reference_lines.csv"
    )


def generate_threewell_split_endpoint() -> None:
    terminal = pd.read_csv(TABLES / "02_triple_well" / "threewell_terminal_metrics.csv")
    methods = ["Langevin", "Kinetic-Langevin", "LSC-adjacent", "LSC-overlong", "CP-overlong"]
    terminal = terminal.set_index("method").loc[methods].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), constrained_layout=True)
    dist_metrics = [("W1_mean", "W1"), ("CDF_sup_mean", "CDF sup"), ("L1_pdf_mean", "density L1")]
    coarse_metrics = [("mode_TV_mean", "mode-TV"), ("middle_error_mean", "middle error")]
    width = 0.22
    x = np.arange(len(methods))
    for j, (col, lab) in enumerate(dist_metrics):
        axes[0].bar(x + (j - 1) * width, terminal[col], width=width, label=lab)
    for j, (col, lab) in enumerate(coarse_metrics):
        axes[1].bar(x + (j - 0.5) * 0.28, np.abs(terminal[col]), width=0.28, label=lab)
    for axi, title in [(axes[0], "distributional endpoint errors"), (axes[1], "coarse-mass endpoint errors")]:
        axi.set_xticks(x)
        axi.set_xticklabels([m.replace("LSC-", "") for m in methods], rotation=25, ha="right")
        axi.set_title(title)
        axi.set_ylabel("error")
        axi.legend(frameon=False, fontsize=8)
        clean(axi)
    label(axes[0], "a")
    label(axes[1], "b")
    save(fig, APP, "threewell_endpoint_split_terminal_errors")


def gl_params(cache: np.lib.npyio.NpzFile) -> dict[str, float]:
    return {str(k): float(v) for k, v in zip(cache["param_names"].astype(str), cache["param_values"].astype(float))}


def local_potential(q: np.ndarray, params: dict[str, float]) -> np.ndarray:
    x = q[..., 0]
    y = q[..., 1]
    return (
        params["ax"] / 4.0 * (x * x - 1.0) ** 2
        + params["ay"] / 4.0 * (y * y - 1.0) ** 2
        + params["c"] * x * y
        + params["hx"] * x
        + params["hy"] * y
        + 0.5 * params["eta"] * x * x * y
    )


def generate_gl_geometry() -> None:
    cache = np.load(TABLES / "04_coupled_phi4_gl" / "coupled_phi4_basin_map_cache.npz", allow_pickle=True)
    params = gl_params(cache)
    gx, gy, labels = cache["gx"], cache["gy"], cache["basin_labels"]
    minima = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "coupled_phi4_local_minima.csv").set_index("phase").loc[PHASE_ORDER]
    masses = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "coupled_phi4_local_2d_basin_masses.csv").set_index("phase").loc[PHASE_ORDER]
    q1 = np.linspace(-1.55, 1.55, 260)
    q2 = np.linspace(-1.55, 1.55, 260)
    Q1, Q2 = np.meshgrid(q1, q2)
    W = local_potential(np.stack([Q1, Q2], axis=-1), params)

    fig, ax = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)
    axes = ax.ravel()
    cs = axes[0].contourf(Q1, Q2, W, levels=45, cmap="viridis")
    axes[0].scatter(minima.x, minima.y, c=[PHASE_COLORS[p] for p in PHASE_ORDER], edgecolor="black", s=64)
    for p in PHASE_ORDER:
        axes[0].text(minima.loc[p, "x"], minima.loc[p, "y"] + 0.08, p, ha="center", fontsize=8)
    axes[0].set_title("local double-well potential")
    axes[0].set_aspect("equal")
    fig.colorbar(cs, ax=axes[0], fraction=0.046)

    axes[1].contourf(gx, gy, labels, levels=np.arange(len(PHASE_ORDER) + 1) - 0.5, cmap="tab10", alpha=0.72)
    axes[1].contour(Q1, Q2, W, levels=16, colors="k", linewidths=0.35, alpha=0.45)
    axes[1].scatter(minima.x, minima.y, c="white", edgecolor="black", s=58)
    axes[1].set_title("gradient-flow basin map")
    axes[1].set_aspect("equal")

    axes[2].bar(PHASE_ORDER, masses["local_2d_basin_mass"], color=[PHASE_COLORS[p] for p in PHASE_ORDER], alpha=0.84)
    axes[2].set_title("single-site Gibbs probabilities")
    axes[2].set_ylabel("probability")

    axes[3].bar(PHASE_ORDER, minima["hessian_min_eig"], color=[PHASE_COLORS[p] for p in PHASE_ORDER], alpha=0.84)
    axes[3].set_title("local Hessian check")
    axes[3].set_ylabel("minimum eigenvalue")
    axes[3].axhline(0, color="0.2", lw=0.8)

    for j, axi in enumerate(axes):
        clean(axi, grid=j not in [0, 1])
        label(axi, "abcd"[j])
        if j in [0, 1]:
            axi.set_xlabel("q1")
            axi.set_ylabel("q2")
    save(fig, MAIN, "vector_gl_phase_geometry_gibbs_probabilities")


def mean_sem(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.groupby(["method", "time"])[metric].agg(["mean", "sem"]).reset_index().fillna(0.0)


def generate_gl_theory() -> None:
    metrics = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "vector_gl_metrics_timeseries.csv")
    final_seed = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "vector_gl_final_metrics_by_seed.csv")
    minima = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "coupled_phi4_local_minima.csv").set_index("phase").loc[PHASE_ORDER]
    graphs = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "vector_gl_graph_gaps.csv")
    mix = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "vector_gl_mixing_times.csv")
    final = final_seed.groupby("method").agg(
        coherent_phase_TV=("coherent_phase_TV", "mean"),
        coherent_field_fraction=("coherent_field_fraction", "mean"),
        noncoherent_field_probability=("noncoherent_field_probability", "mean"),
        mean_gradient_energy_density=("mean_gradient_energy_density", "mean"),
        domain_wall_density_mean=("domain_wall_density_mean", "mean"),
        structure_factor_0=("structure_factor_0", "mean"),
        structure_factor_1=("structure_factor_1", "mean"),
        vector_correlation_1=("vector_correlation_1", "mean"),
    ).reset_index()
    final.to_csv(VALIDATION / "gl_physical_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    ax = axes.ravel()
    graph_order = ["MST", "cycle", "5", "complete"]
    g = graphs.set_index("graph_family").loc[graph_order].reset_index()
    ax[0].plot(g["undirected_edges"], g["graph_gap"], marker="o", color="#4C78A8", lw=2.0)
    for _, row in g.iterrows():
        ax[0].text(row["undirected_edges"], row["graph_gap"], str(row["graph_family"]), fontsize=8)
    ax[0].set_title("weighted graph gap")
    ax[0].set_xlabel("undirected edges")
    ax[0].set_ylabel("coarse graph gap")

    mm = mix[mix.graph_family.notna()].copy()
    mm["short"] = mm["method"].map(GL_SHORT)
    x = np.arange(len(mm))
    ax[1].bar(x - 0.18, mm["terminal_phase_TV"], width=0.36, color="#F58518", label="phase-reference TV")
    nc = final.set_index("method").reindex(mm.method)["noncoherent_field_probability"].to_numpy()
    ax[1].bar(x + 0.18, nc, width=0.36, color="#E45756", label="noncoherent probability")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(mm["short"], rotation=20, ha="right")
    ax[1].set_title("phase accuracy and noncoherence")
    ax[1].set_ylabel("probability / TV")
    ax[1].legend(frameon=False, fontsize=8)

    for method in GL_ORDER:
        sub = mean_sem(metrics[metrics.method == method], "coherent_phase_TV")
        if sub.empty:
            continue
        ax[2].semilogy(sub.time, np.maximum(sub["mean"], 1e-12), color=METHOD_COLORS.get(method), lw=1.6,
                       label=GL_SHORT[method])
    ax[2].set_title("coherent phase communication")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("phase-reference TV")
    ax[2].legend(frameon=False, fontsize=7, ncol=2)

    for p in PHASE_ORDER:
        ax[3].scatter(minima.loc[p, "x"], minima.loc[p, "y"], s=95, color=PHASE_COLORS[p], edgecolor="black")
        ax[3].text(minima.loc[p, "x"], minima.loc[p, "y"] + 0.09, p, ha="center", fontsize=8)
    for method in GL_ORDER:
        sub = final_seed[final_seed.method == method]
        if sub.empty:
            continue
        ax[3].scatter(sub.mean_q1, sub.mean_q2, s=42, color=METHOD_COLORS.get(method), alpha=0.78,
                      label=GL_SHORT[method])
    ax[3].set_title("final mean-order scatter")
    ax[3].set_xlabel("mean q1")
    ax[3].set_ylabel("mean q2")
    ax[3].legend(frameon=False, fontsize=6.5, ncol=2)

    f = final.set_index("method").reindex(GL_ORDER).dropna(how="all").reset_index()
    ax[4].scatter(f["domain_wall_density_mean"], f["mean_gradient_energy_density"],
                  c=[METHOD_COLORS[m] for m in f.method], s=58)
    for _, row in f.iterrows():
        ax[4].text(row["domain_wall_density_mean"], row["mean_gradient_energy_density"], GL_SHORT[row["method"]], fontsize=7)
    ax[4].set_title("domain walls and gradient energy")
    ax[4].set_xlabel("domain-wall density")
    ax[4].set_ylabel("gradient energy density")

    x = np.arange(len(f))
    ax[5].bar(x - 0.18, f["structure_factor_0"], width=0.36, color="#4C78A8", label="S(0)")
    ax[5].bar(x + 0.18, f["structure_factor_1"], width=0.36, color="#54A24B", label="S(1)")
    ax[5].set_xticks(x)
    ax[5].set_xticklabels([GL_SHORT[m] for m in f.method], rotation=25, ha="right")
    ax[5].set_title("low-mode structure factors")
    ax[5].set_ylabel("structure factor")
    ax[5].legend(frameon=False, fontsize=8)

    for j, axi in enumerate(ax):
        clean(axi)
        label(axi, "abcdef"[j])
    save(fig, MAIN, "vector_gl_theory_graph_phase_physical")
    pd.DataFrame(
        [
            {"panel": "a", "quantity": "weighted graph gap", "physical_interpretation": "graph connectivity certificate"},
            {"panel": "b", "quantity": "phase-reference TV and noncoherent probability", "physical_interpretation": "coherent reference accuracy plus domain-rich fields"},
            {"panel": "c", "quantity": "coherent phase-reference TV over time", "physical_interpretation": "coherent phase communication"},
            {"panel": "d", "quantity": "final mean-order scatter", "physical_interpretation": "phase coverage in order-parameter space"},
            {"panel": "e", "quantity": "domain-wall density versus gradient energy", "physical_interpretation": "nonuniform field content"},
            {"panel": "f", "quantity": "low-mode structure factors", "physical_interpretation": "finite-lattice spatial organization"},
        ]
    ).to_csv(VALIDATION / "gl_physical_panel_map.csv", index=False)


def build_gl_appendix_communication_figure(metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.5), constrained_layout=True)
    specs = [
        ("phase_TV", "phase-reference TV", "TV to coherent reference", True),
        ("effective_phase_count", "effective phase count", "count", False),
        ("domain_wall_density_mean", "domain-wall density", "neighbor fraction", False),
    ]
    for ax, (metric, title, ylabel, logy) in zip(axes, specs):
        g = mean_sem(metrics, metric)
        for method in GL_ORDER:
            sub = g[g.method == method]
            if sub.empty:
                continue
            y = sub["mean"].to_numpy()
            if logy:
                ax.semilogy(sub.time, np.maximum(y, 1e-12), color=METHOD_COLORS.get(method), lw=1.6, label=GL_SHORT[method])
            else:
                ax.plot(sub.time, y, color=METHOD_COLORS.get(method), lw=1.6, label=GL_SHORT[method])
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        clean(ax)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for j, ax in enumerate(axes):
        label(ax, "abc"[j])
    return fig


def generate_gl_appendix_communication() -> None:
    metrics = pd.read_csv(TABLES / "04_coupled_phi4_gl" / "vector_gl_metrics_timeseries.csv")
    save(build_gl_appendix_communication_figure(metrics), APP, "vector_gl_fig02_coupled_phi4_phase_communication")
    save(build_gl_appendix_communication_figure(metrics), DIAG, "vector_gl_fig02_coupled_phi4_phase_communication")


def pdf_stream_text(path: Path) -> bytes:
    data = path.read_bytes()
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        blob = m.group(1).strip(b"\r\n")
        try:
            chunks.append(zlib.decompress(blob))
        except Exception:
            continue
    return b"\n".join(chunks)


def write_pdf_label_audit() -> None:
    files = [
        MAIN / "doublewell_theory_epsilon_chi2_sweep.pdf",
        MAIN / "threewell_theory_graph_weighted_density.pdf",
        MAIN / "vector_gl_phase_geometry_gibbs_probabilities.pdf",
        MAIN / "vector_gl_theory_graph_phase_physical.pdf",
        APP / "vector_gl_fig02_coupled_phi4_phase_communication.pdf",
        APP / "threewell_endpoint_split_terminal_errors.pdf",
    ]
    terms = [
        ("old_gl_local_basin_label", b"local 2D basin " + b"masses"),
        ("old_gl_phase_tv_label", b"phase-" + b"TV"),
        ("old_triple_endpoint_dashboard_title", b"terminal " + b"metrics"),
        ("old_doublewell_panel_d_ylabel", b"terminal diagnostic / " + b"threshold"),
    ]
    rows = []
    for path in files:
        text = pdf_stream_text(path) if path.exists() else b""
        for term_id, term in terms:
            rows.append(
                {
                    "file": str(path.relative_to(REPO)).replace("\\", "/"),
                    "term_id": term_id,
                    "found": term in text,
                }
            )
    pd.DataFrame(rows).to_csv(VALIDATION / "generated_pdf_label_audit.csv", index=False)


def write_main_figure_map() -> None:
    rows = [
        {"figure": "doublewell_fig01_target_cdf_nu.pdf", "experiment": "double-well", "theory_role": "target and graph-channel setup"},
        {"figure": "doublewell_theory_epsilon_chi2_sweep.pdf", "experiment": "double-well", "theory_role": "low-temperature form-gap, chi-square, Arrhenius, and graph-channel references"},
        {"figure": "one_d_target_relevant_drift_fields.pdf", "experiment": "1D shared", "theory_role": "implemented corrected drift and target-relevant score scale"},
        {"figure": "threewell_fig01_target_cdf_nu.pdf", "experiment": "triple-well", "theory_role": "normalized target and shell-law setup"},
        {"figure": "threewell_theory_graph_weighted_density.pdf", "experiment": "triple-well", "theory_role": "jump-support graph design, form-gap certificate, and weighted-L2 density convergence"},
        {"figure": "muller10d_fig00_muller10d_target_transform_jump_geometry.pdf", "experiment": "Muller-Brown 10D", "theory_role": "latent CV geometry and lifted jump directions"},
        {"figure": "muller10d_fig01_muller10d_latent_particles_occupancy.pdf", "experiment": "Muller-Brown 10D", "theory_role": "target basin occupancy in latent coordinates"},
        {"figure": "muller10d_fig02_muller10d_basin_communication.pdf", "experiment": "Muller-Brown 10D", "theory_role": "basin communication in the transformed sampler"},
        {"figure": "vector_gl_phase_geometry_gibbs_probabilities.pdf", "experiment": "phi4 GL", "theory_role": "local phase geometry and single-site Gibbs probabilities"},
        {"figure": "vector_gl_fig04a_coupled_phi4_graph_families.pdf", "experiment": "phi4 GL", "theory_role": "phase graph families and graph support"},
        {"figure": "vector_gl_theory_graph_phase_physical.pdf", "experiment": "phi4 GL", "theory_role": "graph connectivity, coherent phase communication, and physical field diagnostics"},
    ]
    pd.DataFrame(rows).to_csv(VALIDATION / "main_figure_metric_map.csv", index=False)


def write_style_registry() -> None:
    pd.DataFrame(style_registry_rows()).to_csv(VALIDATION / "method_style_registry.csv", index=False)


def generate_doublewell_release() -> None:
    setup()
    sync_experiment_figures("doublewell")
    generate_doublewell_sweep()
    write_style_registry()


def generate_triplewell_release() -> None:
    setup()
    sync_experiment_figures("threewell")
    generate_triplewell_graph_figure()
    generate_threewell_split_endpoint()
    drift_inputs = [
        TABLES / "01_double_well" / "doublewell_drift_grid.csv",
        TABLES / "02_triple_well" / "threewell_drift_grid.csv",
    ]
    if all(path.exists() for path in drift_inputs):
        generate_one_d_drift_figure()
    write_style_registry()


def generate_muller_release() -> None:
    setup()
    sync_experiment_figures("muller10d")
    write_style_registry()


def generate_gl_release() -> None:
    setup()
    sync_experiment_figures("vector_gl")
    generate_gl_geometry()
    generate_gl_theory()
    generate_gl_appendix_communication()
    write_style_registry()
    write_pdf_label_audit()
    write_main_figure_map()


def finalize_release_metadata() -> None:
    setup()
    write_pdf_label_audit()
    write_main_figure_map()
    write_style_registry()


def main() -> int:
    generate_doublewell_release()
    generate_triplewell_release()
    generate_muller_release()
    generate_gl_release()
    finalize_release_metadata()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
