"""Generate controlled numerical checks for the comparative-utility theorem.

The three panels are analytic constructions, not fitted experimental results:
(a) a loss-aligned covered-shift witness, (b) cancellation in the paired
radius, and (c) the anchor/sibling variance decomposition.
"""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "comparative_utility"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

blue = "#255f85"
orange = "#d97732"
green = "#3b7d5a"
gray = "#59636e"
light = "#e9eef2"

delta = np.linspace(-1.0, 1.0, 401)
m, alpha, rho = -0.12, -0.58, 1.0
reference = np.full_like(delta, m)
actual = m + alpha * delta
upper = m + np.abs(alpha) * np.abs(delta)
lower = m - np.abs(alpha) * np.abs(delta)

theta = np.linspace(0, np.pi, 301)
paired = np.sqrt(2 - 2 * np.cos(theta))
separate = np.full_like(theta, 2.0)

n = 128
sigma_a2, sigma_s2 = 0.64, 0.36
K = np.arange(1, 33)
total_var = sigma_a2 / n + sigma_s2 / (n * K)
anchor_floor = np.full_like(K, sigma_a2 / n, dtype=float)

fig, ax = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)

# Panel (a): exact interval and equality witness.
a = ax[0]
a.fill_between(delta, lower, upper, color=light, alpha=0.95,
               label=r"certificate $[\Delta_\nu\!\pm\!\sqrt{\rho V_\nu^\Delta}]$")
a.plot(delta, actual, color=blue, lw=2.0, label=r"witness $\Delta_{\pi_\delta}$")
a.axhline(m, color=gray, ls=(0, (3, 2)), lw=1.0, label=r"reference $\Delta_\nu$")
a.axhline(0, color="black", lw=0.7, alpha=0.55)
a.set_xlabel(r"covered-shift direction $\delta$")
a.set_ylabel(r"paired risk difference $\Delta$")
a.set_title("(a) Covered-shift equality")
a.set_xlim(-1, 1)
a.set_ylim(-0.85, 0.65)
a.legend(frameon=False, loc="lower right", handlelength=2.2)
a.text(0.03, 0.94,
        r"$|\Delta_{\pi_\delta}-\Delta_\nu|$" + "\n" +
        r"$=\sqrt{\chi^2 V_\nu^\Delta}$", transform=a.transAxes,
        fontsize=7.2, va="top", bbox=dict(boxstyle="round,pad=.25",
        facecolor="white", edgecolor="#c8d0d6", lw=.6))

# Panel (b): paired radius tightens by cancellation.
b = ax[1]
b.plot(np.degrees(theta), paired, color=blue, lw=2.0,
       label=r"paired $\sqrt{V_\nu^\Delta}$")
b.plot(np.degrees(theta), separate, color=orange, lw=1.8, ls="--",
       label=r"separate $\sqrt{V_\nu(w_C)}+\sqrt{V_\nu(w_F)}$")
b.fill_between(np.degrees(theta), paired, separate, color=light, alpha=0.75)
b.set_xlabel(r"angle between centered responses $\vartheta$ (degrees)")
b.set_ylabel("shift-radius contribution")
b.set_title("(b) Pairing exploits cancellation")
b.set_xlim(0, 180)
b.set_ylim(0, 2.15)
b.legend(frameon=False, loc="upper left", handlelength=2.2)
b.text(0.04, 0.09, "same two marginal sensitivities;\nonly their alignment changes",
        transform=b.transAxes, fontsize=7.2, color=gray)

# Panel (c): two-level finite variance.
c = ax[2]
c.plot(K, total_var, color=blue, lw=2.0,
        label=r"$\sigma_A^2/n+\sigma_S^2/(nK)$")
c.axhline(sigma_a2 / n, color=orange, ls="--", lw=1.8,
          label=r"anchor floor $\sigma_A^2/n$")
c.scatter([1, 4, 16, 32], total_var[[0, 3, 15, 31]],
          s=18, color=green, zorder=3, label="selected $K$")
c.set_xlabel("siblings per anchor $K$")
c.set_ylabel(r"$\operatorname{Var}(\widehat\Delta)$")
c.set_title("(c) More siblings saturate")
c.set_xlim(1, 32)
c.set_ylim(0, 0.0072)
c.set_xticks([1, 4, 8, 16, 32])
c.legend(frameon=False, loc="upper right", handlelength=2.2)
c.text(0.04, 0.09, r"$n=128,\;\sigma_A^2=0.64,\;\sigma_S^2=0.36$",
        transform=c.transAxes, fontsize=7.2, color=gray)

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", color="#d7dde2", lw=0.5, alpha=0.65)

fig.savefig(OUT / "fig_comparative_utility.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_comparative_utility.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT / "fig_comparative_utility.svg", bbox_inches="tight")

with (OUT / "comparative_utility_values.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["panel", "x", "value", "series"])
    for x, y in zip(delta, actual):
        writer.writerow(["a", f"{x:.8f}", f"{y:.8f}", "actual"])
    for x, y in zip(theta, paired):
        writer.writerow(["b", f"{x:.8f}", f"{y:.8f}", "paired"])
    for x, y in zip(K, total_var):
        writer.writerow(["c", f"{x:d}", f"{y:.8f}", "two_level"])

manifest = {
    "panel_a": {"m": m, "alpha": alpha, "rho": rho, "equality": True},
    "panel_b": {"separate_radius": 2.0, "paired_radius_at_zero": 0.0,
                 "paired_radius_at_pi": 2.0},
    "panel_c": {"n": n, "sigma_A2": sigma_a2, "sigma_S2": sigma_s2,
                 "K_values": [1, 4, 8, 16, 32]},
    "font": "Times New Roman with serif fallback",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
