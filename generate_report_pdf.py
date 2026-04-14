"""
generate_report_pdf.py  —  Clean, professional experiment comparison PDF.
Run: python generate_report_pdf.py
Output: report/experiment_comparison.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
RESULTS = Path("results")
TABLES  = RESULTS / "tables"
OUT_PDF = Path("report") / "experiment_comparison.pdf"
OUT_PDF.parent.mkdir(exist_ok=True)

# ── Design tokens ───────────────────────────────────────────────────────────
BG       = "#0f1923"   # dark navy  (page bg for headers)
ACCENT   = "#4fc3f7"   # light blue
WHITE    = "#ffffff"
GRAY     = "#f5f6fa"   # light gray page bg
DARK     = "#1a252f"
MUTED    = "#7f8c8d"

G_GOOD   = "#27ae60"   # green
G_MID    = "#f39c12"   # amber
G_BAD    = "#e74c3c"   # red
G_NEUTRAL= "#dfe6e9"   # light gray cell

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

EXPLAINERS = ["lime", "shap", "grad", "sg", "itg", "ig", "random"]
EXP_DISPLAY = ["LIME", "SHAP", "VanillaGrad", "SmoothGrad", "Grad x Input", "IntGrad", "Random"]

# ── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig, pdf):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")

    # Top accent bar
    ax.axhspan(0.88, 1.0, color=ACCENT, alpha=0.15)
    ax.text(0.5, 0.94, "OpenXAI Benchmark — Full Experiment Comparison",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=WHITE, transform=ax.transAxes)

    ax.text(0.5, 0.80,
            "Phase 1 & 2: Baseline Replication    |    Phase 3: Noise Study    |    Smoothing Fix",
            ha="center", va="center", fontsize=13, color=ACCENT, transform=ax.transAxes)

    # Divider line
    ax.plot([0.08, 0.92], [0.75, 0.75], color=ACCENT, lw=1.2, transform=ax.transAxes, alpha=0.5)

    # Info cards
    cards = [
        ("Datasets",       "Adult Income  |  COMPAS Recidivism"),
        ("Models",         "Pretrained ANN  +  Logistic Regression"),
        ("Explainers",     "LIME  SHAP  VanillaGrad  SmoothGrad  IntGrad  Grad x Input  Random"),
        ("Metrics",        "PGF (Faithfulness, higher better)   |   RIS (Stability, lower better)"),
        ("Noise Levels",   "sigma = 0.0  (clean)   0.1  (mild)   0.3  (moderate)   0.5  (heavy)"),
        ("Smoothing",      "SmoothSHAP & SmoothLIME  (K = 20 averaging copies per input)"),
    ]
    for i, (lbl, val) in enumerate(cards):
        y = 0.68 - i * 0.085
        # card background
        rect = plt.Rectangle((0.06, y - 0.030), 0.88, 0.058,
                               facecolor="#1a2a3a", edgecolor=ACCENT,
                               linewidth=0.6, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.10, y, lbl + ":", ha="left", va="center", fontsize=10,
                color=ACCENT, fontweight="bold", transform=ax.transAxes)
        ax.text(0.27, y, val, ha="left", va="center", fontsize=10,
                color=WHITE, transform=ax.transAxes)

    ax.text(0.5, 0.04,
            "NeurIPS 2022 OpenXAI Replication  |  Phase 3 Extension  |  April 2026",
            ha="center", va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
    _save(fig, pdf)


def _section_divider(pdf, number, title, subtitle, color=ACCENT):
    fig = plt.figure(figsize=(11, 3.0))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.axhspan(0.0, 1.0, color=color, alpha=0.12)
    ax.text(0.06, 0.65, f"SECTION {number}", ha="left", va="center",
            fontsize=11, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.06, 0.38, title, ha="left", va="center",
            fontsize=22, color=WHITE, fontweight="bold", transform=ax.transAxes)
    ax.text(0.06, 0.14, subtitle, ha="left", va="center",
            fontsize=11, color=MUTED, transform=ax.transAxes)
    ax.plot([0.06, 0.94], [0.55, 0.55], color=color, lw=1.5,
            transform=ax.transAxes, alpha=0.4)
    _save(fig, pdf)


def _make_table(ax, cell_text, col_labels, row_labels,
                cell_colors=None, title="", fontsize=10, row_height=1.8):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=DARK, pad=12, loc="left")
    if cell_colors is None:
        cell_colors = [[G_NEUTRAL] * len(col_labels)] * len(cell_text)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellColours=cell_colors,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, row_height)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c0c8d0")
        if r == 0:
            cell.set_facecolor("#2980b9")
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        elif c == -1:
            cell.set_facecolor("#d6eaf8")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(DARK)


def _pgf_color(v):
    try:
        v = float(v)
        if v >= 0.18: return G_GOOD + "55"
        if v >= 0.07: return G_MID  + "55"
        return G_BAD + "55"
    except: return G_NEUTRAL


def _ris_color(v):
    try:
        v = float(str(v).replace(",", ""))
        if v <= 50:    return G_GOOD + "55"
        if v <= 5000:  return G_MID  + "55"
        return G_BAD + "55"
    except: return G_NEUTRAL


def _fmt(v):
    try:
        f = float(v)
        if abs(f) >= 1e9:  return f"{f:.2e}"
        if abs(f) >= 1000: return f"{f:,.0f}"
        return f"{f:.4f}"
    except: return str(v)


def _embed_image(pdf, path: Path, caption: str):
    """Full-page single image embed."""
    fig = plt.figure(figsize=(11, 8.0))
    fig.patch.set_facecolor(GRAY)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    if path.exists():
        img = mpimg.imread(str(path))
        ax.imshow(img, aspect="auto")
    else:
        ax.text(0.5, 0.5, f"[File not found]\n{path.name}",
                ha="center", va="center", fontsize=10, color=MUTED,
                transform=ax.transAxes)
    ax.axis("off")
    fig.text(0.5, 0.02, caption, ha="center", va="bottom",
             fontsize=10, color=DARK, style="italic")
    _save(fig, pdf)


def _two_images(pdf, paths_captions: list[tuple[Path, str]], page_title: str):
    """Two images side by side on one page."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.patch.set_facecolor(GRAY)
    fig.suptitle(page_title, fontsize=12, fontweight="bold", color=DARK, y=1.01)
    for ax, (path, cap) in zip(axes, paths_captions):
        if path.exists():
            img = mpimg.imread(str(path))
            ax.imshow(img, aspect="auto")
        else:
            ax.text(0.5, 0.5, f"[Not found]\n{path.name}",
                    ha="center", va="center", fontsize=9, color=MUTED,
                    transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(cap, fontsize=9, color=DARK, pad=5)
    plt.tight_layout()
    _save(fig, pdf)


# ── DATA PAGES ──────────────────────────────────────────────────────────────

def _phase2_clean_table(pdf):
    csv = TABLES / "adult_ann_metrics.csv"
    if not csv.exists(): return
    df = pd.read_csv(csv, index_col=0)
    metrics = ["PGF", "PGU", "RIS", "RRS", "ROS"]
    direction = ["Higher", "Lower", "Lower", "Lower", "Lower"]

    col_labels = [f"{m}\n({d})" for m, d in zip(metrics, direction)]
    cell_text, cell_colors = [], []
    for exp in EXPLAINERS:
        row, rc = [], []
        for m in metrics:
            v = df.loc[exp, m] if exp in df.index and m in df.columns else float("nan")
            row.append(_fmt(v))
            rc.append(_pgf_color(v) if m == "PGF" else (_ris_color(v) if m == "RIS" else G_NEUTRAL))
        cell_text.append(row)
        cell_colors.append(rc)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(GRAY)
    _make_table(ax, cell_text, col_labels, EXP_DISPLAY, cell_colors,
                title="Phase 2 — Adult Income (ANN, Clean Data, sigma=0.0) — All 5 Metrics",
                fontsize=10, row_height=2.0)

    fig.text(0.04, 0.04,
             "Green = good performance  |  Amber = moderate  |  Red = poor  "
             "(PGF col: higher better, RIS col: lower better)",
             fontsize=8, color=MUTED)
    _save(fig, pdf)


def _degradation_table_adult(pdf):
    csv0 = TABLES / "phase3_adult_sigma0.0.csv"
    csv5 = TABLES / "phase3_adult_sigma0.5.csv"
    if not csv0.exists() or not csv5.exists(): return

    df0 = pd.read_csv(csv0, index_col=0)
    df5 = pd.read_csv(csv5, index_col=0)

    col_labels = ["PGF\nClean", "PGF\nSigma=0.5", "PGF Drop %",
                  "RIS\nClean", "RIS\nSigma=0.5", "RIS Change"]
    cell_text, cell_colors = [], []

    for exp in EXPLAINERS:
        pgf0 = df0.loc[exp, "PGF"] if exp in df0.index else float("nan")
        pgf5 = df5.loc[exp, "PGF"] if exp in df5.index else float("nan")
        ris0 = df0.loc[exp, "RIS"] if exp in df0.index else float("nan")
        ris5 = df5.loc[exp, "RIS"] if exp in df5.index else float("nan")

        drop = ((pgf5 - pgf0) / pgf0 * 100) if pgf0 and not np.isnan(pgf0) else float("nan")
        factor = (ris5 / ris0) if ris0 and not np.isnan(ris0) and ris0 != 0 else float("nan")

        drop_str = f"{drop:+.1f}%" if not np.isnan(drop) else "N/A"
        fac_str  = f"x{factor:.1f}" if not np.isnan(factor) else "N/A"

        row = [_fmt(pgf0), _fmt(pgf5), drop_str, _fmt(ris0), _fmt(ris5), fac_str]
        drop_c  = G_BAD+  "55" if (not np.isnan(drop) and drop < -30) else G_MID+"55"
        fac_c   = G_BAD+  "55" if (not np.isnan(factor) and factor > 100) else (
                  G_GOOD + "55" if (not np.isnan(factor) and factor < 2) else G_MID+"55")
        rc = [_pgf_color(pgf0), _pgf_color(pgf5), drop_c,
              _ris_color(ris0), _ris_color(ris5), fac_c]
        cell_text.append(row)
        cell_colors.append(rc)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(GRAY)
    _make_table(ax, cell_text, col_labels, EXP_DISPLAY, cell_colors,
                title="Phase 3 Study 1 — Degradation: Adult Income ANN\n"
                      "Clean (sigma=0.0) vs Heavy Noise (sigma=0.5)",
                fontsize=10, row_height=2.0)
    fig.text(0.04, 0.04,
             "PGF: faithfulness (higher = better)  |  RIS: stability (lower = better)  "
             "|  Drop% = how much faithfulness fell  |  xN = how many times RIS worsened",
             fontsize=8, color=MUTED)
    _save(fig, pdf)


def _full_sigma_table(pdf, dataset="adult"):
    """Table: PGF at all 4 sigma levels for all 7 explainers."""
    dfs = {}
    for s in [0.0, 0.1, 0.3, 0.5]:
        p = TABLES / f"phase3_{dataset}_sigma{s}.csv"
        if p.exists():
            dfs[s] = pd.read_csv(p, index_col=0)

    if not dfs: return

    col_labels = ["sigma=0.0\n(Clean)", "sigma=0.1\n(Mild)",
                  "sigma=0.3\n(Moderate)", "sigma=0.5\n(Heavy)"]
    cell_text, cell_colors = [], []
    for exp in EXPLAINERS:
        row, rc = [], []
        for s in [0.0, 0.1, 0.3, 0.5]:
            v = dfs[s].loc[exp, "PGF"] if s in dfs and exp in dfs[s].index else float("nan")
            row.append(_fmt(v))
            rc.append(_pgf_color(v))
        cell_text.append(row)
        cell_colors.append(rc)

    ds_title = "Adult Income" if dataset == "adult" else "COMPAS"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(GRAY)
    _make_table(ax, cell_text, col_labels, EXP_DISPLAY, cell_colors,
                title=f"Phase 3 — PGF (Faithfulness) Across All Noise Levels — {ds_title} (ANN)",
                fontsize=11, row_height=2.0)
    fig.text(0.04, 0.04,
             "PGF = faithfulness (higher is better).  "
             "Green >= 0.18   |   Amber >= 0.07   |   Red < 0.07",
             fontsize=8, color=MUTED)
    _save(fig, pdf)


def _smoothing_table(pdf, dataset="adult"):
    ba_csv = TABLES / f"phase3_{dataset}_before_after.csv"
    if not ba_csv.exists(): return
    df = pd.read_csv(ba_csv)
    ds_title = "Adult Income" if dataset == "adult" else "COMPAS"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.patch.set_facecolor(GRAY)
    fig.suptitle(
        f"Phase 3 Study 2 — Smoothing Fix: Vanilla vs Smooth Explainers — {ds_title}",
        fontsize=12, fontweight="bold", color=DARK, y=1.02
    )

    for ax, (vanilla_key, smooth_key, title) in zip(axes, [
        ("shap", "smooth_shap", "SHAP  vs  SmoothSHAP"),
        ("lime", "smooth_lime", "LIME  vs  SmoothLIME"),
    ]):
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", color=DARK, pad=10)
        rows_text, rows_colors = [], []
        for sigma in [0.0, 0.1, 0.3, 0.5]:
            v_df = df[(df["sigma"] == sigma) & (df["explainer"] == vanilla_key)]
            s_df = df[(df["sigma"] == sigma) & (df["explainer"] == smooth_key)]
            v_pgf = float(v_df["PGF"].values[0]) if len(v_df) else float("nan")
            s_pgf = float(s_df["PGF"].values[0]) if len(s_df) else float("nan")
            delta = ((s_pgf - v_pgf) / v_pgf * 100) if (v_pgf and not np.isnan(v_pgf)) else float("nan")
            delta_str = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
            note = "Big win!" if (not np.isnan(delta) and delta > 30) else \
                   "Improved" if (not np.isnan(delta) and delta > 5) else \
                   "Marginal" if (not np.isnan(delta) and delta > 0) else "No change"
            rows_text.append([f"sigma={sigma}", _fmt(v_pgf), _fmt(s_pgf), delta_str, note])
            d_col = G_GOOD+"66" if (not np.isnan(delta) and delta > 30) else \
                    G_MID +"66" if (not np.isnan(delta) and delta > 5)  else G_NEUTRAL
            rows_colors.append([G_NEUTRAL, G_MID+"44", G_GOOD+"44", d_col, d_col])

        tbl = ax.table(
            cellText=rows_text,
            colLabels=["Sigma", "Vanilla\nPGF", "Smooth\nPGF", "Gain %", "Verdict"],
            cellColours=rows_colors,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2.3)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#c0c8d0")
            if r == 0:
                cell.set_facecolor("#2980b9")
                cell.get_text().set_color(WHITE)
                cell.get_text().set_fontweight("bold")
    plt.tight_layout()
    _save(fig, pdf)


def _verdict_table(pdf):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(GRAY)
    ax.axis("off")
    ax.set_title("Final Verdict — Summary of All Conditions",
                 fontsize=13, fontweight="bold", color=DARK, pad=14, loc="left")

    col_labels = ["Explainer", "Clean\nFaithfulness", "Clean\nStability",
                  "At Sigma=0.5\nFaithfulness", "Smoothing\nFix Works?", "Recommendation"]
    rows = [
        ["IntGrad (ig)",     "[+] High",    "[+] Best",     "[+] Holds best",   "N/A",           "Best overall on Adult"],
        ["SmoothGrad (sg)",  "[+] High",    "[~] Moderate", "[!] Collapses",    "N/A",           "Good on clean data only"],
        ["LIME",             "[+] High",    "[+] Good",     "[~] -37% drop",    "[~] Marginal",  "Reliable, but slow"],
        ["SHAP",             "[~] Medium",  "[!] Unstable", "[!] -40% drop",    "[+] +172%",     "Use SmoothSHAP in noisy env"],
        ["VanillaGrad",      "[+] High",    "[!] Explodes", "[!] RIS ->10^14",  "N/A",           "Avoid for stability"],
        ["Grad x Input",     "[~] Medium",  "[!] Worst",    "[!] Worst",        "N/A",           "Avoid"],
        ["Random",           "[!] Floor",   "[+] Stable",   "[+] Consistent",   "N/A",           "Baseline reference only"],
    ]
    row_colors = [
        [G_NEUTRAL, G_GOOD+"55", G_GOOD+"55", G_GOOD+"55",    G_NEUTRAL,   "#e8f8f0"],
        [G_NEUTRAL, G_GOOD+"55", G_MID +"55", G_BAD +"55",    G_NEUTRAL,   "#fef9e7"],
        [G_NEUTRAL, G_GOOD+"55", G_GOOD+"55", G_MID +"55",    G_MID +"55", "#fef9e7"],
        [G_NEUTRAL, G_MID +"55", G_BAD +"55", G_BAD +"55",    G_GOOD+"55", "#fdf2e0"],
        [G_NEUTRAL, G_GOOD+"55", G_BAD +"55", G_BAD +"55",    G_NEUTRAL,   "#fdf2e0"],
        [G_NEUTRAL, G_MID +"55", G_BAD +"55", G_BAD +"55",    G_NEUTRAL,   "#fdf2e0"],
        [G_NEUTRAL, G_BAD +"55", G_GOOD+"55", G_GOOD+"55",    G_NEUTRAL,   "#f5f5f5"],
    ]

    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellColours=row_colors,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c0c8d0")
        if r == 0:
            cell.set_facecolor("#1a5276")
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")

    fig.text(0.04, 0.03,
             "[+] = Good   |   [~] = Moderate   |   [!] = Poor/Dangerous",
             fontsize=9, color=MUTED)
    _save(fig, pdf)


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print(f"Building PDF -> {OUT_PDF} ...")
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: Cover
        _cover(pdf)

        # ── SECTION 1: Phase 2 Baseline ──────────────────────────────────
        _section_divider(pdf, "1", "Phase 1 & 2 — Baseline Results",
                         "Clean data evaluation on Adult Income + COMPAS with ANN and LR models",
                         color="#4fc3f7")

        # Metrics table
        _phase2_clean_table(pdf)

        # Heatmaps — 2 per page
        _two_images(pdf, [
            (RESULTS / "adult_ann_heatmap.png",  "Adult Income — ANN Heatmap"),
            (RESULTS / "adult_lr_heatmap.png",   "Adult Income — LR Heatmap"),
        ], "Phase 2 — Metric Heatmaps: Adult Income (ANN vs LR)")

        _two_images(pdf, [
            (RESULTS / "compas_ann_heatmap.png", "COMPAS — ANN Heatmap"),
            (RESULTS / "compas_lr_heatmap.png",  "COMPAS — LR Heatmap"),
        ], "Phase 2 — Metric Heatmaps: COMPAS (ANN vs LR)")

        _two_images(pdf, [
            (RESULTS / "adult_ann_bar_charts.png",  "Adult Income — ANN Bar Charts"),
            (RESULTS / "compas_ann_bar_charts.png", "COMPAS — ANN Bar Charts"),
        ], "Phase 2 — Per-Metric Bar Charts (ANN Model)")

        _embed_image(pdf, RESULTS / "multi_dataset_comparison.png",
                     "Figure: Cross-Dataset PGF Comparison — Adult Income vs COMPAS (ANN)")

        # ── SECTION 2: Phase 3 Degradation ───────────────────────────────
        _section_divider(pdf, "2", "Phase 3 Study 1 — Degradation Under Noise",
                         "All 7 explainers stressed across sigma = 0.0 to 0.5  |  "
                         "How bad does faithfulness and stability get?",
                         color="#ce93d8")

        _full_sigma_table(pdf, "adult")
        _full_sigma_table(pdf, "compas")
        _degradation_table_adult(pdf)

        _embed_image(pdf, RESULTS / "phase3_adult_degradation_PGF.png",
                     "Figure: Adult Income — PGF (Faithfulness) Degradation Curves across all sigma levels")
        _embed_image(pdf, RESULTS / "phase3_adult_degradation_RIS.png",
                     "Figure: Adult Income — RIS (Stability) Degradation Curves across all sigma levels")
        _embed_image(pdf, RESULTS / "phase3_compas_degradation_PGF.png",
                     "Figure: COMPAS — PGF (Faithfulness) Degradation Curves across all sigma levels")
        _embed_image(pdf, RESULTS / "phase3_compas_degradation_RIS.png",
                     "Figure: COMPAS — RIS (Stability) Degradation Curves across all sigma levels")

        _two_images(pdf, [
            (RESULTS / "phase3_adult_robustness_gap.png",  "Adult Income — Robustness Gap"),
            (RESULTS / "phase3_compas_robustness_gap.png", "COMPAS — Robustness Gap"),
        ], "Phase 3 — Robustness Gap Between Best and Worst Explainer")

        # ── SECTION 3: Smoothing Fix ──────────────────────────────────────
        _section_divider(pdf, "3", "Phase 3 Study 2 — The Smoothing Fix",
                         "SmoothSHAP and SmoothLIME vs vanilla versions across all noise levels  |  "
                         "Can averaging K=20 copies recover explanation quality?",
                         color="#80cbc4")

        _smoothing_table(pdf, "adult")
        _smoothing_table(pdf, "compas")

        _two_images(pdf, [
            (RESULTS / "phase3_adult_ba_sigma03.png",  "Adult — Before vs After  sigma=0.3"),
            (RESULTS / "phase3_adult_ba_sigma05.png",  "Adult — Before vs After  sigma=0.5"),
        ], "Phase 3 — Before vs After Smoothing: Adult Income")

        _two_images(pdf, [
            (RESULTS / "phase3_compas_ba_sigma03.png", "COMPAS — Before vs After  sigma=0.3"),
            (RESULTS / "phase3_compas_ba_sigma05.png", "COMPAS — Before vs After  sigma=0.5"),
        ], "Phase 3 — Before vs After Smoothing: COMPAS")

        _embed_image(pdf, RESULTS / "phase3_compas_smoothing_heatmap.png",
                     "Figure: Smoothing Benefit Heatmap — delta PGF (SmoothSHAP / SmoothLIME vs Vanilla)")

        # ── SECTION 4: Final Verdict ──────────────────────────────────────
        _section_divider(pdf, "4", "Final Verdict",
                         "Which explainer is best — and under what conditions?",
                         color="#ef9a9a")

        _verdict_table(pdf)

    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"\nPDF ready: {OUT_PDF.resolve()}")
    print(f"Size: {size_kb:.0f} KB  |  Open in any PDF viewer.")


if __name__ == "__main__":
    main()
