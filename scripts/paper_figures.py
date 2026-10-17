"""Generate the two paper figures (English): mean spectra and mean OOF recall/AUPRC."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HSI_ROOT = os.path.join(BASE, "dataSet", "hsi_original")
RESULTS = os.path.join(BASE, "ml", "results")
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
ATCC_PREFIXES = ["ATCC13", "ATCC16", "ATCC27"]
PRIOR = 113279 / (113279 + 744724)  # resistant-class prior (~0.13)


def mean_spectrum(folder):
    hdr = os.path.join(HSI_ROOT, folder, "capture", f"{folder}.hdr")
    if not os.path.exists(hdr):
        return None
    img = spectral.open_image(hdr).load()
    return img.reshape(-1, img.shape[2]).mean(axis=0)


def fig1():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    atcc, clin = [], []
    for p in sorted(os.listdir(HSI_ROOT)):
        m = mean_spectrum(p)
        if m is None:
            continue
        is_atcc = any(x in p for x in ATCC_PREFIXES)
        (atcc if is_atcc else clin).append(m)
        ax.plot(m, color="red" if is_atcc else "steelblue", alpha=0.35, lw=0.9)
    ax.plot(np.mean(atcc, 0), color="darkred", lw=2.5, ls="--", label="ATCC mean (resistant)")
    ax.plot(np.mean(clin, 0), color="navy", lw=2.5, ls="--", label="Clinical mean (susceptible)")
    ax.set_xlabel("Band index"); ax.set_ylabel("Mean intensity")
    ax.legend(); ax.grid(alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_mean_spectra.png"), dpi=200)
    plt.close(fig)


def fig2():
    df = pd.concat(pd.read_csv(os.path.join(RESULTS, f))
                   for f in os.listdir(RESULTS) if f.startswith("fold_scores_"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, met, title in zip(axes, ["recall", "average_precision"], ["Mean OOF recall", "Mean OOF AUPRC"]):
        s = df[df["metrica"] == met].groupby("pipeline")["score"].mean().sort_index()
        ax.bar(s.index, s.values, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axhline(PRIOR, color="gray", ls="--", label=f"Resistant-class prior ({PRIOR:.2f})")
        ax.set_title(title); ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=40); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_oof_recall_auprc.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
