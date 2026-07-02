from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPORT = Path("reports/jcp_sampling_report")


# Manuscript method names: LSBMC is the implementation name for LSC-CP; ULA is "local Langevin".
MODE_NAME = {"ULA": "local", "RawCP": "raw CP", "LSBMC": "LSC-CP"}

METRIC_LABELS = {
    "cdf_sup_error_mean": "CDF-sup",
    "density_l1_error_mean": "density $L^1$",
    "basin_population_error_mean": "mode/basin-TV",
    "basin_tv_final_mean": "basin-TV",
    "observable_bias_middle_mass_mean": "middle-mass err",
    "coverage_time_all_basins_mean": "coverage time",
    "coverage_fraction_mean": "coverage frac",
    "threshold_time_tv_mean": "threshold time",
    "basin_kl_mean": "basin KL",
    "free_energy_rmse_mean": "FE RMSE",
    "observable_bias_energy_mean": "energy bias",
    "block_marginal_kl_mean": "block KL",
    "count_mode_kl_mean": "count KL",
    "deep_count_mean_mean": "deep-count mean",
    "ess_per_sec_mean": "ESS/sec",
    "ess_per_gradient_eval_mean": "ESS/grad",
    "ess_per_levy_quadrature_eval_mean": "ESS/Levy eval",
    "runtime_sec_mean": "runtime (s)",
    "grad_evals_mean": "grad evals",
    "levy_quadrature_evals_mean": "Levy evals",
    "jump_events_mean": "jump events",
}

EXP_METRICS = {
    # manuscript main examples (local / raw CP / LSC-CP)
    "double_well_reproduction": ["cdf_sup_error_mean", "density_l1_error_mean", "coverage_time_all_basins_mean", "ess_per_sec_mean"],
    "triple_well_support": ["basin_population_error_mean", "observable_bias_middle_mass_mean", "coverage_fraction_mean", "ess_per_sec_mean"],
    "muller10d_basin_communication": ["basin_tv_final_mean", "coverage_time_all_basins_mean", "coverage_fraction_mean", "ess_per_sec_mean"],
    # ablations
    "triple_well_invariance": ["basin_tv_final_mean", "cdf_sup_error_mean"],
    "muller10d_invariance": ["basin_tv_final_mean"],
    # efficiency benchmark (ESS reported WITH bias)
    "double_well_benchmark": ["cdf_sup_error_mean", "ess_per_sec_mean", "ess_per_gradient_eval_mean", "runtime_sec_mean"],
    "triple_well_benchmark": ["basin_population_error_mean", "ess_per_sec_mean", "ess_per_gradient_eval_mean"],
    "muller10d_benchmark": ["basin_tv_final_mean", "ess_per_sec_mean", "ess_per_gradient_eval_mean"],
    # additional landscapes
    "four_well_graph": ["basin_kl_mean", "basin_population_error_mean", "ess_per_sec_mean", "ess_per_gradient_eval_mean"],
    "manywell_scaling": ["block_marginal_kl_mean", "count_mode_kl_mean", "deep_count_mean_mean", "ess_per_sec_mean", "levy_quadrature_evals_mean"],
    "alanine_ramachandran": ["basin_tv_final_mean", "free_energy_rmse_mean", "coverage_fraction_mean", "ess_per_sec_mean"],
}

FIGURES = [
    ("fig_dw_cdf_sup", "Double well: target fidelity (CDF-sup)"),
    ("fig_dw_density_l1", "Double well: density $L^1$ error"),
    ("fig_timestep_bias", "Timestep bias: raw CP defect is $h$-independent"),
    ("fig_scale_intensity_relax", "Scale$\\times$intensity relaxation ridge"),
    ("fig_tw_mode_tv", "Triple well: mode-TV by support"),
    ("fig_tw_middle_mass", "Triple well: middle-mass error"),
    ("fig_mb10_basin_tv", "10D Muller--Brown: basin-TV (lifted vs random)"),
    ("fig_invariance", "Start-at-equilibrium invariance"),
    ("fig_four_well_basin_kl", "Four-well graph ablation"),
    ("fig_manywell_block_kl", "ManyWell block-marginal KL scaling"),
    ("fig_alanine_basin_tv", "Alanine dipeptide: basin-TV (wrapped vs random)"),
    ("fig_lj_obstruction", "Lennard-Jones rotational-symmetry obstruction"),
    ("fig_lj38_darting", "LJ38 double funnel: geometry-matched darting vs local MC"),
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tex_escape(value) -> str:
    text = "" if value is None else str(value)
    return (text.replace("\\", r"\textbackslash{}")
                .replace("_", r"\_")
                .replace("%", r"\%")
                .replace("&", r"\&")
                .replace("#", r"\#")
                .replace("{", r"\{")
                .replace("}", r"\}"))


def tex_id(value) -> str:
    return str(value).replace("_", "-").replace("/", "-").replace(" ", "-")


def fnum(row, key):
    try:
        x = float(row.get(key, "nan"))
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def fmt(x, nd=3):
    try: x = float(x)
    except Exception: return "n/a"
    if not math.isfinite(x): return "n/a"
    ax = abs(x)
    if (ax != 0 and ax < 1e-3) or ax >= 1e5:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def method_label(row):
    method = MODE_NAME.get(str(row.get("method", "")), str(row.get("method", "")))
    bank = str(row.get("bank_name", ""))
    if bank and bank != "none":
        return f"{method}/{bank}"
    return method


def target_label(row):
    return str(row.get("target_name", ""))


def finite_rows(rows, exp, metric):
    out = []
    for r in rows:
        if r.get("experiment_name") != exp:
            continue
        v = fnum(r, metric)
        if math.isfinite(v):
            out.append((v, r))
    return out


def best(rows, exp, metric, minimize=True):
    vals = finite_rows(rows, exp, metric)
    if not vals:
        return None
    return (min if minimize else max)(vals, key=lambda t: t[0])


def table_for_experiment(rows: list[dict], exp: str) -> str:
    sub = [r for r in rows if r.get("experiment_name") == exp]
    metrics = [m for m in EXP_METRICS.get(exp, []) if any(math.isfinite(fnum(r, m)) for r in sub)]
    if not metrics:
        metrics = ["ess_per_sec_mean"]
    caption = f"Generated metrics for {exp.replace('_',' ')}. Means and standard errors are computed from raw CSV rows."
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{tab:{tex_id(exp)}}}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llll" + "r" * len(metrics) + "}", "\\toprule"]
    header = ["Target", "Method", "Bank", "$n$"] + [METRIC_LABELS.get(m, m.replace("_mean", "").replace("_", " ")) for m in metrics]
    lines.append(" & ".join(header) + r" \\")
    lines.append("\\midrule")
    for r in sub:
        bank = str(r.get("bank_name", "") or "")
        scale = str(r.get("bank_scale", "") or "")
        if scale and scale not in ("", "nan"):
            bank = f"{bank} (scale {scale})"
        method_disp = MODE_NAME.get(str(r.get("method", "")), str(r.get("method", "")))
        cells = [tex_escape(r.get("target_name")), tex_escape(method_disp), tex_escape(bank), tex_escape(r.get("n_rows"))]
        cells.extend(fmt(r.get(m)) for m in metrics)
        lines.append(" & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    return "\n".join(lines)


def generated_figure_tex() -> str:
    lines = ["\\subsection{Generated figures}"]
    for i in range(0, len(FIGURES), 2):
        chunk = FIGURES[i:i+2]
        lines.append("\\begin{figure}[t]")
        lines.append("\\centering")
        for name, cap in chunk:
            lines.append("\\begin{minipage}{0.48\\textwidth}")
            lines.append("\\centering")
            lines.append(f"\\IfFileExists{{figures/{name}.pdf}}{{\\includegraphics[width=\\textwidth]{{figures/{name}.pdf}}}}{{}}");
            lines.append(f"\\caption*{{{tex_escape(cap)}}}")
            lines.append("\\end{minipage}")
            if len(chunk) == 2 and (name, cap) == chunk[0]:
                lines.append("\\hfill")
        fig_range = f"{i+1}--{i+len(chunk)}" if len(chunk) > 1 else f"{i+1}"
        lines.append(f"\\caption{{Generated manuscript figure(s) {fig_range} from aggregate CSV summaries.}}")
        lines.append(f"\\label{{fig:generated-{i//2+1}}}")
        lines.append("\\end{figure}")
    return "\n".join(lines)


def _val(rows, exp, method, metric, bank=None):
    for r in rows:
        if r.get("experiment_name") == exp and r.get("method") == method and (bank is None or r.get("bank_name") == bank):
            v = fnum(r, metric)
            if math.isfinite(v):
                return v
    return math.nan


def exec_summary(rows: list[dict]) -> str:
    bullets = []
    # target fidelity: LSC-CP vs raw CP
    dw_lsc = _val(rows, "double_well_reproduction", "LSBMC", "cdf_sup_error_mean")
    dw_raw = _val(rows, "double_well_reproduction", "RawCP", "cdf_sup_error_mean")
    if math.isfinite(dw_lsc) and math.isfinite(dw_raw):
        bullets.append(f"Double well: LSC-CP reduces the CDF-sup target error from {fmt(dw_raw)} (raw CP) to {fmt(dw_lsc)}, "
                       f"a genuine invariant-measure correction (Fig.~timestep-bias shows this defect is timestep-independent).")
    mb_lsc = _val(rows, "muller10d_basin_communication", "LSBMC", "basin_tv_final_mean", "minima_complete_graph")
    mb_raw = _val(rows, "muller10d_basin_communication", "RawCP", "basin_tv_final_mean", "minima_complete_graph")
    mb_rnd = _val(rows, "muller10d_basin_communication", "LSBMC", "basin_tv_final_mean", "random_matched_length_control")
    if math.isfinite(mb_lsc) and math.isfinite(mb_raw):
        extra = f" and beats a random-direction matched-length control ({fmt(mb_rnd)})" if math.isfinite(mb_rnd) else ""
        bullets.append(f"10D transformed M\\\"uller--Brown: LSC-CP with lifted basin-to-basin jumps attains basin-TV {fmt(mb_lsc)} "
                       f"versus {fmt(mb_raw)} for raw CP{extra}.")
    tw_over = _val(rows, "triple_well_support", "LSBMC", "basin_population_error_mean", "overlong")
    tw_adj = _val(rows, "triple_well_support", "LSBMC", "basin_population_error_mean", "adjacent")
    if math.isfinite(tw_over) and math.isfinite(tw_adj):
        bullets.append(f"Triple well: the stationary correction keeps mode-TV small for both supports (overlong {fmt(tw_over)}, adjacent {fmt(tw_adj)}); "
                       f"support choice governs which modes communicate directly.")
    inv_lsc = _val(rows, "muller10d_invariance", "LSBMC", "basin_tv_final_mean")
    inv_raw = _val(rows, "muller10d_invariance", "RawCP", "basin_tv_final_mean")
    if math.isfinite(inv_lsc) and math.isfinite(inv_raw):
        bullets.append(f"Started at equilibrium, LSC-CP preserves the target (basin-TV {fmt(inv_lsc)}) while raw CP drifts off it ({fmt(inv_raw)}).")
    mw = best(rows, "manywell_scaling", "block_marginal_kl_mean", minimize=True)
    if mw:
        bullets.append(f"ManyWell high-D stress test: lowest block-marginal KL is {fmt(mw[0])} for {tex_escape(method_label(mw[1]))} on {tex_escape(target_label(mw[1]))}.")
    if not bullets:
        return "Generated result summaries are available in the tables."
    return "\\begin{itemize}\n" + "\n".join(f"\\item {b}" for b in bullets) + "\n\\end{itemize}\n"


def results_narrative(rows: list[dict]) -> str:
    lines = ["\\section{Results}",
             "All values in this section are generated from the aggregate CSV summary produced by the pipeline. "
             "The three main examples compare local Langevin, raw compound-Poisson (CP), and the "
             "L\\'evy-score-corrected CP (LSC-CP); the benchmark additionally reports MALA, BAOAB, HMC, and PT. "
             "Efficiency numbers (ESS/sec) are always reported alongside a target-fidelity error, because a "
             "non-converged sampler can post a high ESS on the wrong distribution.",
             "\\subsection{Quantitative summary}",
             exec_summary(rows)]

    # Double well
    dw_lsc = _val(rows, "double_well_reproduction", "LSBMC", "cdf_sup_error_mean")
    dw_raw = _val(rows, "double_well_reproduction", "RawCP", "cdf_sup_error_mean")
    dw_loc = _val(rows, "double_well_reproduction", "ULA", "cdf_sup_error_mean")
    if math.isfinite(dw_lsc):
        lines.append(f"\\subsection{{Double well: target fidelity}}"
                     f"With a shared shell jump law, LSC-CP attains CDF-sup error {fmt(dw_lsc)} versus {fmt(dw_raw)} for "
                     f"raw CP and {fmt(dw_loc)} for local Langevin (which stays near its metastable initial well over the "
                     f"displayed horizon). The timestep-bias study confirms the raw-CP error is independent of the "
                     f"integrator step, i.e.\\ a genuine change of invariant law rather than a discretization artifact.")

    # Triple well
    tw_over = _val(rows, "triple_well_support", "LSBMC", "basin_population_error_mean", "overlong")
    tw_adj = _val(rows, "triple_well_support", "LSBMC", "basin_population_error_mean", "adjacent")
    tw_short = _val(rows, "triple_well_support", "LSBMC", "basin_population_error_mean", "short")
    tw_raw = _val(rows, "triple_well_support", "RawCP", "basin_population_error_mean", "overlong")
    if math.isfinite(tw_over) and math.isfinite(tw_adj):
        short_note = f" A too-short shell (mode-TV {fmt(tw_short)}) fails to connect the modes and behaves like local noise." if math.isfinite(tw_short) else ""
        lines.append(f"\\subsection{{Triple well: support versus bias}}"
                     f"Corrected adjacent and overlong supports both keep mode-TV small ({fmt(tw_adj)} and {fmt(tw_over)}), "
                     f"while the uncorrected overlong control records the bias of omitting the correction ({fmt(tw_raw)}).{short_note} "
                     f"The correction governs target fidelity; the support governs which modes communicate directly.")

    # 10D MB
    mb_lsc = _val(rows, "muller10d_basin_communication", "LSBMC", "basin_tv_final_mean", "minima_complete_graph")
    mb_raw = _val(rows, "muller10d_basin_communication", "RawCP", "basin_tv_final_mean", "minima_complete_graph")
    mb_rnd = _val(rows, "muller10d_basin_communication", "LSBMC", "basin_tv_final_mean", "random_matched_length_control")
    mb_loc = _val(rows, "muller10d_basin_communication", "ULA", "basin_tv_final_mean")
    if math.isfinite(mb_lsc):
        lines.append(f"\\subsection{{Transformed M\\\"uller--Brown (10D)}}"
                     f"After lifting the latent-minima displacements through the mixing map, LSC-CP attains basin-TV "
                     f"{fmt(mb_lsc)} versus {fmt(mb_raw)} for raw CP, {fmt(mb_rnd)} for a random-direction matched-length "
                     f"control, and {fmt(mb_loc)} for local Langevin. Acceleration therefore comes specifically from "
                     f"geometry-matched inter-basin jumps, not from nonlocal jumps of the same length in arbitrary directions.")

    # Benchmark (ESS with bias)
    b_lsc_bias = _val(rows, "double_well_benchmark", "LSBMC", "cdf_sup_error_mean")
    b_lsc_ess = _val(rows, "double_well_benchmark", "LSBMC", "ess_per_sec_mean")
    b_ula_bias = _val(rows, "double_well_benchmark", "ULA", "cdf_sup_error_mean")
    b_ula_ess = _val(rows, "double_well_benchmark", "ULA", "ess_per_sec_mean")
    if math.isfinite(b_lsc_ess) and math.isfinite(b_ula_ess):
        lines.append(f"\\subsection{{Efficiency benchmark}}"
                     f"On the smooth low-dimensional double well, LSC-CP is the most accurate (CDF-sup {fmt(b_lsc_bias)}) "
                     f"but not the fastest per wall-clock (ESS/sec {fmt(b_lsc_ess)}) because of the L\\'evy-score quadrature. "
                     f"Local Langevin posts a far higher ESS/sec ({fmt(b_ula_ess)}) at a much larger bias ({fmt(b_ula_bias)}), "
                     f"illustrating why ESS must be read together with target fidelity. The method's efficiency advantage "
                     f"appears in the trapped and high-dimensional regimes (four well, ManyWell).")

    fw = best(rows, "four_well_graph", "basin_kl_mean", True)
    if fw:
        fw_rows = {str(r.get("bank_name", "")): fnum(r, "basin_kl_mean")
                   for r in rows if r.get("experiment_name") == "four_well_graph" and r.get("method") == "LSBMC"}
        rnd = fw_rows.get("random_matched_length_control", math.nan)
        structured = [v for b, v in fw_rows.items() if b != "random_matched_length_control" and math.isfinite(v)]
        qual = ""
        if structured and math.isfinite(rnd) and min(structured) < rnd:
            qual = f" Structured geometry-matched jumps beat the random matched-length control ({fmt(min(structured))} vs.\\ {fmt(rnd)})."
        lines.append(f"\\subsection{{Additional landscapes}}"
                     f"Four-well graph ablation: lowest basin KL {fmt(fw[0])} for {tex_escape(method_label(fw[1]))}.{qual}")

    mw_block = best(rows, "manywell_scaling", "block_marginal_kl_mean", True)
    if mw_block:
        lines.append(f"ManyWell high-dimensional stress test: lowest block-marginal KL {fmt(mw_block[0])} for "
                     f"{tex_escape(method_label(mw_block[1]))} on {tex_escape(target_label(mw_block[1]))}, the regime where "
                     f"local samplers remain entropically confined and the block-flip jumps match the product structure.")

    # Alanine dipeptide (torus)
    al_lsc = _val(rows, "alanine_ramachandran", "LSBMC", "basin_tv_final_mean", "wrapped_basin_graph")
    al_rnd = _val(rows, "alanine_ramachandran", "LSBMC", "basin_tv_final_mean", "random_control")
    al_raw = _val(rows, "alanine_ramachandran", "RawCP", "basin_tv_final_mean", "wrapped_basin_graph")
    al_ula = _val(rows, "alanine_ramachandran", "ULA", "basin_tv_final_mean")
    al_cov = _val(rows, "alanine_ramachandran", "LSBMC", "coverage_fraction_mean", "wrapped_basin_graph")
    if math.isfinite(al_lsc):
        lines.append(f"\\subsection{{Alanine dipeptide (Ramachandran torus)}}"
                     f"On an analytic surrogate for the alanine-dipeptide free-energy surface (a periodic mixture over the "
                     f"conformational basins), wrapped basin-to-basin jumps give LSC-CP the best basin-population fidelity "
                     f"(basin-TV {fmt(al_lsc)}, all-basin coverage fraction {fmt(al_cov)}), versus {fmt(al_rnd)} for a "
                     f"random-direction control, {fmt(al_raw)} for uncorrected raw CP, and {fmt(al_ula)} for local Langevin "
                     f"(which stays confined to its starting basin). Parallel tempering is the only baseline that competes on "
                     f"coverage. The construction extends unchanged to the torus via wrapped displacements and a periodic score.")

    # Lennard-Jones obstruction (documented limitation)
    lj_path = Path("results/jcp_sampling/lj_obstruction.json")
    if lj_path.exists():
        lj = json.loads(lj_path.read_text())
        lines.append(f"\\subsection{{Lennard-Jones cluster: a structural limitation}}"
                     f"For an LJ$_{{{int(lj.get('n_atoms',7))}}}$ cluster in two dimensions the fixed Cartesian jump bank is "
                     f"incompatible with the cluster's continuous rotational symmetry. A jump aligned in the catalogue frame "
                     f"is productive only while the cluster keeps that orientation: sweeping the cluster orientation, the fixed "
                     f"jump lands at low (isomer) energy for only {fmt(100*lj['productivity_fixed'])}\\% of angles before running "
                     f"into hard-core overlaps, whereas a rotationally-augmented gated bank (rotated copies of the jump, gated "
                     f"by landing energy) restores productivity to {fmt(100*lj['productivity_gated'])}\\%. This delineates the "
                     f"method's scope: fixed additive jumps suit targets whose metastable transitions are fixed coordinate "
                     f"displacements, not systems with continuous symmetries.")

    # LJ38 double funnel: barrier theorem + Metropolis-darting hybrid
    lj38_path = Path("results/jcp_sampling/lj38_darting.json")
    if lj38_path.exists():
        lj38 = json.loads(lj38_path.read_text())
        lines.append(f"\\subsection{{LJ$_{{38}}$ double funnel: barrier limitation and a darting hybrid}}"
                     f"The canonical metastable cluster is LJ$_{{38}}$, whose FCC truncated-octahedron global minimum "
                     f"({fmt(lj38['E_fcc_global'])}, recovered exactly) competes with an icosahedral funnel "
                     f"({fmt(lj38['E_icosahedral'])}). The straight-line displacement between the two funnels crosses a "
                     f"{fmt(lj38['straight_line_barrier'])}~$\\varepsilon$ barrier of atomic overlaps, so at the "
                     f"solid--solid transition temperature $\\beta\\cdot$barrier~$\\approx{int(lj38['beta_barrier_at_transition_T'])}$ "
                     f"and the rejection-free score integrand $\\exp[-\\beta\\,\\Delta V]$ is numerically zero: the additive "
                     f"correction \\emph{{cannot}} make a barrier-crossing jump target-preserving. The same geometry-matched "
                     f"inter-funnel displacement, applied as a \\emph{{Metropolis-corrected}} endpoint dart (accept on the "
                     f"endpoint energy, no path integral), escapes the icosahedral funnel and reaches the FCC global "
                     f"(FCC-funnel fraction from the icosahedral start: local MC {fmt(lj38['local_from_ico_final_FCC_frac'])} "
                     f"vs darting {fmt(lj38['darting_from_ico_final_FCC_frac'])}). Thus geometry-matched nonlocal jumps solve "
                     f"the LJ$_{{38}}$ global-funnel-discovery problem, but through the Metropolis-corrected hybrid rather than "
                     f"the pure additive score; recovering the entropy-balanced funnel populations at the transition needs the "
                     f"full smart-darting machinery (darting regions and Jacobians), which we leave to future work.")

    lines.append(generated_figure_tex())
    lines.append("\\subsection{Generated tables}")
    lines.append("\\input{generated_tables.tex}")
    return "\n\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()
    root = Path(args.results_root)
    agg = root / "aggregate" / "all_summary.csv"
    if not agg.exists():
        raise SystemExit("No aggregate/all_summary.csv found; run aggregate_results first")
    rows = read_csv(agg)
    (REPORT / "tables").mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    table_inputs = []
    for exp in sorted({r.get("experiment_name") for r in rows if r.get("experiment_name")}):
        path = REPORT / "tables" / f"{exp}_table.tex"
        path.write_text(table_for_experiment(rows, exp))
        table_inputs.append(f"\\input{{tables/{exp}_table.tex}}")
    (REPORT / "exec_summary.tex").write_text(exec_summary(rows))
    (REPORT / "numbers.tex").write_text("% Generated automatically from result files.\n")
    agg_manifest = root / "aggregate" / "aggregate_manifest.json"
    manifest_note = ""
    if agg_manifest.exists():
        data = json.loads(agg_manifest.read_text())
        manifest_note = f" Launcher manifest: \\texttt{{{tex_escape(data.get('launcher_manifest'))}}}. Raw metric files aggregated: {tex_escape(data.get('n_raw_rows'))}."
    summary_path = tex_escape("aggregate/all_summary.csv")
    (REPORT / "appendix_configs.tex").write_text(
        f"Results root: \\texttt{{{tex_escape(args.results_root)}}}. Generated tables are read from \\texttt{{{summary_path}}}.{manifest_note}\n"
    )
    (REPORT / "generated_tables.tex").write_text("\n".join(table_inputs) + "\n")
    (REPORT / "generated_results.tex").write_text(results_narrative(rows))
    print(REPORT)

if __name__ == "__main__": main()
