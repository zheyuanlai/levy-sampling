"""Canonical plotting grammar for the four-experiment numerical release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MethodStyle:
    color: str
    marker: str
    linestyle: str = "-"
    label: Optional[str] = None


METHOD_STYLES = {
    "Langevin": MethodStyle("#2B6CB0", "o", "-", "Langevin"),
    "Kinetic-Langevin": MethodStyle("#6B7C93", "s", "-", "Kinetic-Langevin"),
    "LSC-CP": MethodStyle("#2F855A", "^", "-", "LSC-CP"),
    "CP": MethodStyle("#C53030", "D", "--", "CP"),
    "LSC-adjacent": MethodStyle("#2F855A", "^", "-", "LSC-adjacent"),
    "LSC-overlong": MethodStyle("#805AD5", "v", "-", "LSC-overlong"),
    "CP-overlong": MethodStyle("#C53030", "X", "--", "CP-overlong"),
    "LSC-connected": MethodStyle("#2F855A", "^", "-", "LSC-connected"),
    "LSC-disconnected": MethodStyle("#805AD5", "v", "-", "LSC-disconnected"),
    "LSC-wrong": MethodStyle("#9F7AEA", "P", ":", "LSC-mismatched"),
    "LSC-atom": MethodStyle("#38A169", "^", "-", "LSC-atom"),
    "CP-atom": MethodStyle("#E53E3E", "D", "--", "CP-atom"),
    "LSC-shell": MethodStyle("#2F855A", "v", "-", "LSC-shell"),
    "CP-shell": MethodStyle("#C53030", "X", "--", "CP-shell"),
    "LSC-MST-shell": MethodStyle("#2F855A", "^", "-", "MST"),
    "LSC-cycle-shell": MethodStyle("#3182CE", "s", "-", "cycle"),
    "LSC-5-shell": MethodStyle("#805AD5", "D", "-", "5-edge"),
    "LSC-complete-shell": MethodStyle("#DD6B20", "P", "-", "complete"),
    "MST": MethodStyle("#2F855A", "^", "-", "MST"),
    "cycle": MethodStyle("#3182CE", "s", "-", "cycle"),
    "5": MethodStyle("#805AD5", "D", "-", "5-edge"),
    "5-edge": MethodStyle("#805AD5", "D", "-", "5-edge"),
    "complete": MethodStyle("#DD6B20", "P", "-", "complete"),
}

METHOD_ALIASES = {
    "LSC-MST": "LSC-MST-shell",
    "LSC-cycle": "LSC-cycle-shell",
    "LSC-5": "LSC-5-shell",
    "LSC-complete": "LSC-complete-shell",
}

REFERENCE_STYLES = {
    "form_gap": {"linestyle": "--", "linewidth": 1.35, "alpha": 0.72},
    "abscissa": {"linestyle": ":", "linewidth": 1.75, "alpha": 0.88},
    "theory": {"linestyle": "--", "linewidth": 1.8, "alpha": 0.95},
    "empirical_fit": {"linestyle": ":", "linewidth": 1.8, "alpha": 0.95},
    "data": {"linestyle": "none", "markersize": 5.0, "alpha": 1.0},
}

PHASE_COLORS = {
    "--": "#2B6CB0",
    "-+": "#DD6B20",
    "+-": "#2F855A",
    "++": "#C53030",
}

RC_PARAMS = {
    "figure.dpi": 135,
    "savefig.dpi": 300,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.55,
    "lines.linewidth": 1.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def canonical_method(method: str) -> str:
    value = str(method)
    return METHOD_ALIASES.get(value, value)


def method_style(method: str) -> MethodStyle:
    key = canonical_method(method)
    return METHOD_STYLES.get(key, MethodStyle("#718096", "o", "-", key))


def method_color(method: str) -> str:
    return method_style(method).color


def method_marker(method: str) -> str:
    return method_style(method).marker


def method_linestyle(method: str) -> str:
    return method_style(method).linestyle


def method_label(method: str) -> str:
    style = method_style(method)
    return style.label or canonical_method(method)


def apply_plot_style(plt) -> None:
    plt.rcParams.update(RC_PARAMS)


def clean_axes(ax, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)
    if grid:
        ax.grid(True, alpha=RC_PARAMS["grid.alpha"], linewidth=RC_PARAMS["grid.linewidth"])
    else:
        ax.grid(False)


def panel_label(ax, text: str, x: float = -0.11, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontweight="bold", va="bottom")


def style_registry_rows() -> list[dict[str, str]]:
    rows = []
    for method, style in sorted(METHOD_STYLES.items()):
        rows.append(
            {
                "method": method,
                "label": style.label or method,
                "color": style.color,
                "marker": style.marker,
                "linestyle": style.linestyle,
            }
        )
    return rows
