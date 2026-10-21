"""Full learning curves of the paper's run as PNGs: one figure per condition, one panel per participant, evidence (bits per
held-out 20-s window) against training time for the three readouts, with the 1-bit verifier goal and each participant's decoder
goal. Input: figures/figdata/fig2_data_e4.json (verifiers/analysis/make_fig_data.py). Output: results/learning_curves/<cond>.png.
Usage: python results/make_learning_curves.py"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = json.load(open(os.path.join(ROOT, "figures", "figdata", "fig2_data_e4.json"))); OUT = os.path.join(HERE, "learning_curves"); os.makedirs(OUT, exist_ok=True)
COND = {"native": "Clean fMRI", "m6c": "Uniform noise, 4 times the noise power (-6 dB)", "m12c": "Uniform noise, 16 times (-12 dB)",
        "m18c": "Uniform noise, 63 times (-18 dB)", "usL90": "Simulated ultrasound, one probe, optimistic physics (K = 2)",
        "usB90": "Simulated ultrasound, two probes, optimistic physics (K = 2)", "usL50": "Simulated ultrasound, one probe, median physics (K = 2)"}
SER = [("dec_empBig", "Decoder (Tang et al., 2023)", "#D62629", "s"), ("ver_ETC", "Verifier, encode-then-compare", "#2F5BC4", "o"),
       ("ver_C", "Verifier, contrastive", "#2BA12B", "^")]                  # palette validated (dataviz validate_palette.js, light surface)
INK, MUTED = "#333333", "#777777"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})
for c, title in COND.items():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    ymax = max([p[1] for S in D["subjects"] for k, *_ in SER for p in D[S][f"cur_{c}_{k}"]] + [D[S]["bar"] for S in D["subjects"]]) + 0.5
    for ax, S in zip(axs, D["subjects"]):
        ax.axhline(1.0, ls=":", color=INK, lw=1.0); ax.axhline(D[S]["bar"], ls="--", color="#D62629", lw=0.8, alpha=0.7)
        ax.text(0.17, 1.0, "verifier goal 1 bit", va="bottom", fontsize=6.5, color=INK)
        ax.text(0.17, D[S]["bar"], f"decoder goal {D[S]['bar']:.2f} bits", va="bottom", fontsize=6.5, color=INK)
        for k, lab, col, mk in SER:
            cur = np.array(D[S][f"cur_{c}_{k}"], float)
            if cur.size: ax.plot(cur[:, 0], cur[:, 1], "-", color=col, lw=2, marker=mk, ms=4.5, mec="white", mew=0.6, label=lab)
        ax.set_xscale("log"); ax.set_xticks([0.2, 0.5, 1, 2, 5, 15]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "15"])
        ax.set_xlim(0.15, 18); ax.set_ylim(min(-0.5, ax.get_ylim()[0]), ymax); ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        ax.set_title(f"participant {S}", fontsize=8, color=INK); ax.set_xlabel("Training data (h)")
        if c == "m18c" and not D[S][f"cur_{c}_dec_empBig"]: ax.text(16, 0.2, "decoder not run", ha="right", fontsize=6.5, color=INK)
    axs[0].set_ylabel("Evidence per 20-s window (bits)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK); fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, f"{c}.png"), dpi=200); plt.close(fig)
    print("wrote", os.path.join("results", "learning_curves", f"{c}.png"))
