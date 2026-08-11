"""
Generates the 3 v4 performance charts (LR, RF, XGBoost): AUC-ROC, PR-AUC and F1,
comparing spatial vs. temporal validation. Reads v4_model_comparison_metrics.csv
(7-15 km ring, ratio 1:1) and saves one PNG per metric in this same folder.

Re-run after any change to v4_model_comparison_metrics.csv:
    "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" generate_v4_metric_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
METRICS_CSV = HERE.parent / "v4_buffer_7_15km" / "v4_model_comparison_metrics.csv"

# Validated categorical palette (dataviz skill): slot 1 = blue, slot 2 = orange.
# Color encodes the validation TYPE (spatial vs. temporal), fixed across the
# 3 charts -- not the model, which goes as the x-axis label.
COLOR_SPATIAL = "#2a78d6"
COLOR_TEMPORAL = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost"]
MODEL_LABELS = {
    "Logistic Regression": "Logistic\nRegression",
    "Random Forest": "Random\nForest",
    "XGBoost": "XGBoost",
}

METRICS = [
    {
        "key": "roc_auc",
        "title": "AUC-ROC — v4 (Buffer ring 7–15 km, ratio 1:1)",
        "subtitle": "How good is the model to separate burned from unburned pixels (0.5 = random, 1.0 = perfect)",
        "filename": "v4_auc_roc_comparison.png",
    },
    {
        "key": "pr_auc",
        "title": "PR-AUC — v4 (Buffer ring 7–15 km, ratio 1:1)",
        "subtitle": "Precision-Recall; baseline = positive class prevalence (0.5 in the balanced 1:1 dataset)",
        "filename": "v4_pr_auc_comparison.png",
    },
    {
        "key": "f1_at_0.5",
        "title": "F1 (threshold 0.5) — v4 (Buffer ring 7–15 km, ratio 1:1)",
        "subtitle": "Balance between precision and recall, with decision threshold fixed at 0.5",
        "filename": "v4_f1_comparison.png",
    },
]

BAR_WIDTH = 0.32
GAP = 0.04


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)
    df = df.set_index("model").loc[MODEL_ORDER]
    return df


def plot_metric(df: pd.DataFrame, spec: dict) -> None:
    key = spec["key"]
    spatial_col = f"spatial_{key}"
    spatial_std_col = f"{spatial_col}_std"
    temporal_col = f"temporal_{key}"

    x = range(len(MODEL_ORDER))
    x_spatial = [i - (BAR_WIDTH / 2 + GAP / 2) for i in x]
    x_temporal = [i + (BAR_WIDTH / 2 + GAP / 2) for i in x]

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    spatial_vals = df[spatial_col].values
    spatial_err = df[spatial_std_col].values if spatial_std_col in df.columns else None
    temporal_vals = df[temporal_col].values

    ax.bar(
        x_spatial, spatial_vals, width=BAR_WIDTH, color=COLOR_SPATIAL,
        yerr=spatial_err, capsize=3,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capthick": 1},
        label="Spatial (5 folds)", zorder=3,
    )
    ax.bar(
        x_temporal, temporal_vals, width=BAR_WIDTH, color=COLOR_TEMPORAL,
        label="Temporal (holdout)", zorder=3,
    )

    # Direct value labels above each bar (above the error bar, if any).
    spatial_err_for_label = spatial_err if spatial_err is not None else [0] * len(spatial_vals)
    for xi, v, e in zip(x_spatial, spatial_vals, spatial_err_for_label):
        ax.text(xi, v + e + 0.022, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9.5, color=INK_PRIMARY)
    for xi, v in zip(x_temporal, temporal_vals):
        ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9.5, color=INK_PRIMARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10.5, color=INK_PRIMARY)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9.5)
    ax.tick_params(axis="x", length=0)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.set_title(spec["title"], fontsize=13, color=INK_PRIMARY, weight="bold", pad=28, loc="left")
    fig.text(0.02, 0.925, spec["subtitle"], fontsize=9.5, color=INK_SECONDARY)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False,
        fontsize=10, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    out_path = HERE / spec["filename"]
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    df = load_metrics()
    for spec in METRICS:
        plot_metric(df, spec)


if __name__ == "__main__":
    main()
