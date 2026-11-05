"""ppvp_audit/plotstyle.py — 图表样式（与 Phase 2 相同的已校验调色板：蓝 / 橙 / 绿，色觉缺陷下可分）。"""
from __future__ import annotations

SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def style() -> None:
    import logging

    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": ["Inter", "WenQuanYi Zen Hei", "DejaVu Sans"], "font.size": 9.5,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.0, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 160, "axes.unicode_minus": False,
    })
