"""Consistent, high-resolution visualizations for EDA and model evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


NAVY = "#102A43"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
SLATE = "#486581"
PALETTE = {"Legitimate": TEAL, "Fraud": ORANGE, 0: TEAL, 1: ORANGE}


def configure_style() -> None:
    """Apply the project-wide academic chart style."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.titlecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _slug(value: str) -> str:
    """Return a safe lowercase filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _inr(value: float, _position: int) -> str:
    """Format a numeric axis value using compact INR notation."""
    if abs(value) >= 10000000:
        return f"₹{value / 10000000:.1f}Cr"
    if abs(value) >= 100000:
        return f"₹{value / 100000:.1f}L"
    if abs(value) >= 1000:
        return f"₹{value / 1000:.0f}K"
    return f"₹{value:.0f}"


def _save(fig: plt.Figure, path: Path, title: str, caption: str, dpi: int) -> dict[str, str]:
    """Save a figure and return its index metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"file": str(path), "title": title, "caption": caption}


def generate_eda_plots(
    raw: pd.DataFrame, engineered: pd.DataFrame, target: pd.Series, output_dir: Path, dpi: int = 180
) -> list[dict[str, str]]:
    """Generate EDA figures before model fitting.

    Args:
        raw: Validated source dataframe.
        engineered: Target-free engineered feature dataframe.
        target: Binary fraud label.
        output_dir: Root images directory.
        dpi: Output resolution.
    Returns:
        Figure index records for documentation.
    """
    configure_style()
    records: list[dict[str, str]] = []
    labels = target.map({0: "Legitimate", 1: "Fraud"})
    eda = output_dir / "eda"
    numeric = engineered.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in engineered.columns if column not in numeric]

    counts = labels.value_counts().reindex(["Legitimate", "Fraud"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=[TEAL, ORANGE])
    ax.set_title("Class distribution — supplied claims")
    ax.set_ylabel("Number of claims")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(value):,}", ha="center", va="bottom")
    records.append(
        _save(
            fig,
            eda / "target_distribution_bar.png",
            "Class distribution",
            "Fraud is the positive class; bars show all validated rows.",
            dpi,
        )
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=[TEAL, ORANGE],
        autopct="%.1f%%",
        startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    ax.set_title("Fraud prevalence — supplied claims")
    records.append(
        _save(
            fig,
            eda / "target_distribution_pie.png",
            "Fraud prevalence",
            "The supplied snapshot contains a six-percent fraud prevalence.",
            dpi,
        )
    )

    missing = raw.isna().mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(missing.index[::-1], missing.values[::-1], color=SLATE)
    ax.set_title("Missing-value audit — raw workbook")
    ax.set_xlabel("Missing values (%)")
    records.append(
        _save(
            fig,
            eda / "missingness_bar.png",
            "Missing-value audit",
            "No source column is silently dropped; the pipeline records the measured null rate.",
            dpi,
        )
    )

    for column in numeric:
        values = engineered[column].astype(float)
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, color in [(0, TEAL), (1, ORANGE)]:
            subset = values[target == label].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset):
                ax.hist(subset, bins=25, alpha=0.58, color=color, label="Fraud" if label else "Legitimate")
        ax.set_title(f"Distribution of {column} by class")
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Number of claims")
        ax.legend(frameon=True)
        if "amount" in column or "income" in column:
            ax.xaxis.set_major_formatter(FuncFormatter(_inr))
        records.append(
            _save(
                fig,
                eda / f"numeric_{_slug(column)}_by_class.png",
                f"{column} distribution",
                "Training-free distribution view; extreme values are retained for fraud analysis.",
                dpi,
            )
        )

    headline = [
        column
        for column in ["claim_amount_inr", "patient_income_inr", "patient_age_years", "claim_to_income_ratio"]
        if column in engineered
    ]
    if headline:
        melted = engineered[headline].copy()
        melted["class"] = labels.values
        fig, axes = plt.subplots(1, len(headline), figsize=(4 * len(headline), 4))
        axes = np.atleast_1d(axes)
        for axis, column in zip(axes, headline):
            sns.boxplot(
                data=melted,
                x="class",
                y=column,
                order=["Legitimate", "Fraud"],
                palette=[TEAL, ORANGE],
                ax=axis,
                hue="class",
                legend=False,
            )
            axis.set_title(column.replace("_", " ").title())
            axis.set_xlabel("")
            axis.set_ylabel("Value")
            if "amount" in column or "income" in column:
                axis.yaxis.set_major_formatter(FuncFormatter(_inr))
        records.append(
            _save(
                fig,
                eda / "headline_boxplots_by_class.png",
                "Headline feature spread",
                "Box plots preserve extreme values rather than treating every outlier as an error.",
                dpi,
            )
        )

    corr = engineered[numeric].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=False, cbar_kws={"label": "Spearman correlation"})
    ax.set_title("Spearman correlation heatmap — engineered numeric features")
    records.append(
        _save(
            fig,
            eda / "correlation_heatmap.png",
            "Correlation heatmap",
            "Correlation supports redundancy review; it is not interpreted as causation.",
            dpi,
        )
    )

    for column in categorical:
        counts_by_value = engineered[column].astype(str).value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(counts_by_value.index, counts_by_value.values, color=SLATE)
        ax.set_title(f"Top categories — {column}")
        ax.set_xlabel("Number of claims")
        records.append(
            _save(
                fig,
                eda / f"categorical_{_slug(column)}_counts.png",
                f"{column} categories",
                "Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels.",
                dpi,
            )
        )

        rates = engineered.assign(_target=target.values).groupby(column, dropna=False)["_target"].agg(["mean", "count"])
        rates = rates[rates["count"] >= max(10, int(len(engineered) * 0.005))].sort_values("mean").tail(12)
        if len(rates):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(rates.index.astype(str), rates["mean"] * 100, color=ORANGE)
            ax.set_title(f"Observed fraud rate — {column}")
            ax.set_xlabel("Fraud rate (%)")
            records.append(
                _save(
                    fig,
                    eda / f"fraud_rate_{_slug(column)}.png",
                    f"Fraud rate by {column}",
                    "Rates are descriptive and should not be treated as causal or sufficient for an adverse decision.",
                    dpi,
                )
            )

    for x_column, y_column, filename in [
        ("patient_income_inr", "claim_amount_inr", "claim_amount_vs_income.png"),
        ("patient_age_years", "claim_amount_inr", "claim_amount_vs_age.png"),
    ]:
        if x_column in engineered and y_column in engineered:
            fig, ax = plt.subplots(figsize=(7, 5))
            for value, color, label in [(0, TEAL, "Legitimate"), (1, ORANGE, "Fraud")]:
                subset = target == value
                ax.scatter(
                    engineered.loc[subset, x_column],
                    engineered.loc[subset, y_column],
                    s=14,
                    alpha=0.40,
                    c=color,
                    label=label,
                )
            ax.set_title(f"{y_column.replace('_', ' ').title()} versus {x_column.replace('_', ' ').title()}")
            ax.set_xlabel(x_column.replace("_", " ").title())
            ax.set_ylabel(y_column.replace("_", " ").title())
            ax.legend()
            if "income" in x_column or "amount" in y_column:
                ax.xaxis.set_major_formatter(FuncFormatter(_inr))
                ax.yaxis.set_major_formatter(FuncFormatter(_inr))
            records.append(
                _save(
                    fig,
                    eda / filename,
                    "Claim amount relationship",
                    "Scatter points are colored by the supplied binary label; overlap is expected in real triage.",
                    dpi,
                )
            )

    if "ClaimDate" in raw:
        temporal = (
            raw.assign(is_fraud=target.values)
            .set_index("ClaimDate")
            .resample("MS")
            .agg(claims=("ClaimID", "count"), fraud_rate=("is_fraud", "mean"))
        )
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(temporal.index, temporal["claims"], color=SLATE, alpha=0.50, width=20, label="Claims")
        ax1.set_ylabel("Claims")
        ax2 = ax1.twinx()
        ax2.plot(temporal.index, temporal["fraud_rate"] * 100, color=ORANGE, marker="o", label="Fraud rate")
        ax2.set_ylabel("Fraud rate (%)")
        ax1.set_title("Monthly claim volume and observed fraud rate")
        records.append(
            _save(
                fig,
                eda / "monthly_volume_and_fraud_rate.png",
                "Monthly claim patterns",
                "Monthly counts and rates are descriptive; the two-year supplied window is not a causal time series.",
                dpi,
            )
        )

    if {"claimtype", "providerspecialty"}.issubset(engineered.columns):
        table = pd.crosstab(
            engineered["claimtype"], engineered["providerspecialty"], values=target, aggfunc="mean"
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(table * 100, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Fraud rate (%)"}, ax=ax)
        ax.set_title("Fraud rate heatmap — claim type by specialty")
        records.append(
            _save(
                fig,
                eda / "fraud_rate_heatmap_claim_type_specialty.png",
                "Fraud-rate heatmap",
                "Cells show descriptive percentages; sparse cells should be interpreted cautiously.",
                dpi,
            )
        )

    # Additional relationship figures close the coverage requirement with
    # compact views that are useful in both the report appendix and defense.
    if "claim_amount_inr" in engineered and "cluster_category" in engineered:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(
            data=pd.DataFrame(
                {"cluster": engineered["cluster_category"], "amount": engineered["claim_amount_inr"], "label": labels}
            ),
            x="cluster",
            y="amount",
            hue="label",
            split=True,
            inner="quartile",
            palette=[TEAL, ORANGE],
            ax=ax,
        )
        ax.set_title("Claim amount distribution by supplied cluster")
        ax.set_xlabel("Cluster category")
        ax.set_ylabel("Claim amount")
        ax.yaxis.set_major_formatter(FuncFormatter(_inr))
        records.append(
            _save(
                fig,
                eda / "claim_amount_by_cluster_violin.png",
                "Claim amount by cluster",
                "The cluster is a supplied source field and is shown for shortcut-risk auditing.",
                dpi,
            )
        )
    if "ClaimDate" in raw and "ClaimAmount" in raw:
        monthly_amount = (
            raw.assign(_target=target.values)
            .groupby(raw["ClaimDate"].dt.to_period("M"))
            .agg(median_claim=("ClaimAmount", "median"), mean_claim=("ClaimAmount", "mean"))
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            monthly_amount.index.astype(str),
            monthly_amount["median_claim"],
            color=TEAL,
            marker="o",
            label="Median claim",
        )
        ax.plot(
            monthly_amount.index.astype(str), monthly_amount["mean_claim"], color=ORANGE, marker="o", label="Mean claim"
        )
        ax.set_title("Monthly claim amount trend")
        ax.set_xlabel("Claim month")
        ax.set_ylabel("Claim amount")
        ax.tick_params(axis="x", rotation=60)
        ax.yaxis.set_major_formatter(FuncFormatter(_inr))
        ax.legend()
        records.append(
            _save(
                fig,
                eda / "monthly_claim_amount_trend.png",
                "Monthly claim amount",
                "Mean and median amounts are descriptive; the supplied observation window is limited.",
                dpi,
            )
        )
    if "ClaimType" in raw:
        cross = pd.crosstab(raw["ClaimType"], labels).reindex(columns=["Legitimate", "Fraud"], fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        cross.plot(kind="bar", stacked=True, color=[TEAL, ORANGE], ax=ax)
        ax.set_title("Claim-type composition by supplied label")
        ax.set_xlabel("Claim type")
        ax.set_ylabel("Number of claims")
        ax.legend(title="Label")
        records.append(
            _save(
                fig,
                eda / "claim_type_composition_stacked.png",
                "Claim-type composition",
                "Counts show volume and class composition without implying a causal claim-type effect.",
                dpi,
            )
        )
    if "ClaimAmount" in raw:
        deciles = pd.qcut(raw["ClaimAmount"], q=10, duplicates="drop")
        decile_rate = (
            pd.DataFrame({"decile": deciles.astype(str), "target": target.values})
            .groupby("decile", sort=False)["target"]
            .agg(["mean", "count"])
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(decile_rate)), decile_rate["mean"] * 100, color=ORANGE)
        ax.set_title("Observed fraud rate by claim-amount decile")
        ax.set_xlabel("Claim-amount decile, low to high")
        ax.set_ylabel("Fraud rate (%)")
        ax.set_xticks(range(len(decile_rate)), [str(i + 1) for i in range(len(decile_rate))])
        records.append(
            _save(
                fig,
                eda / "fraud_rate_by_claim_amount_decile.png",
                "Fraud rate by amount decile",
                "Decile rates are descriptive and should be checked against policy and provider context.",
                dpi,
            )
        )

    return records


def generate_model_curves(results: list[dict[str, Any]], output_dir: Path, dpi: int = 180) -> list[dict[str, str]]:
    """Generate ROC, precision-recall, leaderboard, and confusion figures.

    Args:
        results: Model result dictionaries containing validation and test arrays.
        output_dir: Root images directory.
        dpi: Figure resolution.
    Returns:
        Figure index records.
    """
    configure_style()
    records: list[dict[str, str]] = []
    model_dir = output_dir / "models"
    curves_dir = model_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for row in results if row.get("status") == "complete"])
    if frame.empty:
        return records
    frame = frame.sort_values(["val_f2", "val_pr_auc"], ascending=False)
    palette = sns.color_palette("tab20", n_colors=len(frame))

    fig, ax = plt.subplots(figsize=(11, 6))
    for color, row in zip(palette, frame.to_dict("records")):
        fpr, tpr, _ = roc_curve(row["y_validation"], row["validation_probabilities"])
        ax.plot(fpr, tpr, label=f"{row['display_name']} ({row['val_roc_auc']:.3f})", color=color, linewidth=1.6)
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="Chance")
    ax.set_title("Validation ROC curves — all complete models")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    records.append(
        _save(
            fig,
            curves_dir / "roc_curves_validation.png",
            "Validation ROC curves",
            "All models use the same validation split; legend values are ROC-AUC.",
            dpi,
        )
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    prevalence = frame.iloc[0]["y_validation"].mean()
    for color, row in zip(palette, frame.to_dict("records")):
        precision, recall, _ = precision_recall_curve(row["y_validation"], row["validation_probabilities"])
        ax.plot(recall, precision, label=f"{row['display_name']} ({row['val_pr_auc']:.3f})", color=color, linewidth=1.6)
    ax.axhline(prevalence, linestyle="--", color="#999999", linewidth=1, label=f"Prevalence ({prevalence:.3f})")
    ax.set_title("Validation precision-recall curves — all complete models")
    ax.set_xlabel("Recall (fraud catch rate)")
    ax.set_ylabel("Precision (flag purity)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    records.append(
        _save(
            fig,
            curves_dir / "pr_curves_validation.png",
            "Validation PR curves",
            "PR-AUC is the primary ranking metric under class imbalance.",
            dpi,
        )
    )

    metric_columns = [
        ("val_precision", "Precision"),
        ("val_recall", "Recall"),
        ("val_f1", "F1"),
        ("val_f2", "F2"),
        ("val_pr_auc", "PR-AUC"),
    ]
    top = frame.head(12).copy()
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.15
    for index, (column, label) in enumerate(metric_columns):
        ax.bar(x + (index - 2) * width, top[column], width, label=label)
    ax.set_xticks(x, [str(name)[:18] for name in top["display_name"]], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation score")
    ax.set_title("Validation leaderboard — key fraud metrics")
    ax.legend(ncol=5)
    records.append(
        _save(
            fig,
            model_dir / "validation_metric_comparison.png",
            "Validation metric comparison",
            "Models are ordered by validation F2 and show the recall-weighted trade-off.",
            dpi,
        )
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        frame["train_seconds"],
        frame["val_pr_auc"],
        s=60,
        c=[ORANGE if key == frame.iloc[0]["key"] else TEAL for key in frame["key"]],
        alpha=0.85,
    )
    for _, row in frame.iterrows():
        ax.annotate(
            row["key"], (row["train_seconds"], row["val_pr_auc"]), fontsize=7, xytext=(4, 3), textcoords="offset points"
        )
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Efficiency versus fraud-ranking quality")
    records.append(
        _save(
            fig,
            model_dir / "training_time_vs_pr_auc.png",
            "Training time versus PR-AUC",
            "The highlighted point is the validation-selected winner; efficiency is reported, not used to hide slower models.",
            dpi,
        )
    )

    for _, row in frame.head(8).iterrows():
        matrix = np.array(
            [
                [row["val_true_negative"], row["val_false_positive"]],
                [row["val_false_negative"], row["val_true_positive"]],
            ]
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap=sns.light_palette(ORANGE, as_cmap=True), cbar=False, ax=ax)
        ax.set_xticklabels(["Predicted legitimate", "Predicted fraud"])
        ax.set_yticklabels(["Actual legitimate", "Actual fraud"], rotation=0)
        ax.set_title(f"Confusion matrix — {row['display_name']}")
        records.append(
            _save(
                fig,
                model_dir / f"confusion_matrix_{_slug(row['key'])}.png",
                f"Confusion matrix for {row['display_name']}",
                "Fraud is the positive class; matrix is evaluated at the validation-selected threshold.",
                dpi,
            )
        )

    return records


def feature_importance_plot(importances: pd.DataFrame, output_path: Path, title: str, dpi: int = 180) -> dict[str, str]:
    """Render a horizontal feature-importance chart from a tidy table."""
    configure_style()
    table = importances.sort_values("importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(table["feature"], table["importance"], color=ORANGE)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _save(fig, output_path, title, "Global importance is descriptive; it does not establish causation.", dpi)


def calibration_plot(
    y_true: np.ndarray, before: np.ndarray, after: np.ndarray, output_path: Path, dpi: int = 180
) -> dict[str, str]:
    """Render pre/post calibration reliability curves."""
    configure_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for probabilities, label, color in [(before, "Before calibration", SLATE), (after, "After calibration", ORANGE)]:
        observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect calibration")
    ax.set_title("Reliability diagram — validation probabilities")
    ax.set_xlabel("Mean predicted fraud probability")
    ax.set_ylabel("Observed fraud frequency")
    ax.legend()
    return _save(
        fig,
        output_path,
        "Reliability diagram",
        "Calibration is fitted with validation data only and is reported before any locked test evaluation.",
        dpi,
    )
