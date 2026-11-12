"""Deterministic reports for validated saved forecasts (no model fitting)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import CLASSES, score

RULE_ID = "validation_macro_f1_then_min_extreme_recall_then_simplicity_v1"
MODEL_ORDER = ["majority", "persistence", "logistic", "lightgbm"]
REGIONS = [("west", 120, 180), ("central", 180, 240), ("east", 240, 300)]
TOLERANCE = 1e-12
LEADERBOARD_COLUMNS = [
    "model",
    "split",
    "n",
    "macro_f1",
    "accuracy",
    "delta_macro_f1_vs_persistence",
    "label_coverage",
    "prediction_coverage",
    "warming_recall",
    "cooling_recall",
]
PER_CLASS_COLUMNS = [
    "model",
    "split",
    "class",
    "precision",
    "recall",
    "f1",
    "support",
    "true_class_absent",
]
DIAGNOSTIC_COLUMNS = [
    "model",
    "split",
    "group_type",
    "group_value",
    "n",
    "macro_f1",
    "accuracy",
    "cooling_support",
    "neutral_support",
    "warming_support",
    "absent_true_classes",
]
TIMELINE_COLUMNS = [
    "region",
    "site_id",
    "date",
    "target_date",
    "observed_target_date",
    "y_true",
    "model",
    "y_pred",
]


def metric_tables(metrics, split, audit):
    """Keep unrounded metric values and the shared reference denominator."""
    counts = audit["splits"][split]
    reference_n, eligible_n = counts["reference_n"], counts["eligible_n"]
    persistence = metrics.get("persistence", {}).get(split)
    board, classes = [], []
    for model in sorted(metrics):
        result = metrics[model][split]
        if result["n"] != eligible_n:
            raise ValueError("Metrics and audited eligible population disagree")
        board.append(
            {
                "model": model,
                "split": split,
                "n": result["n"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "delta_macro_f1_vs_persistence": None
                if persistence is None
                else result["macro_f1"] - persistence["macro_f1"],
                "label_coverage": eligible_n / reference_n if reference_n else None,
                "prediction_coverage": audit["models"][model]["prediction_coverage"],
                "warming_recall": result["per_class"]["warming"]["recall"],
                "cooling_recall": result["per_class"]["cooling"]["recall"],
            }
        )
        for label in CLASSES:
            values = result["per_class"][label]
            classes.append(
                {
                    "model": model,
                    "split": split,
                    "class": label,
                    **values,
                    "true_class_absent": values["support"] == 0,
                }
            )
    return (
        pd.DataFrame(board, columns=LEADERBOARD_COLUMNS).sort_values(
            ["macro_f1", "model"],
            ascending=[False, True],
            kind="stable",
            ignore_index=True,
        ),
        pd.DataFrame(classes, columns=PER_CLASS_COLUMNS),
    )


def diagnostic_table(aligned, reference):
    """Apply the same reference-defined slices to every model."""
    context = reference[["site_id", "date", "current_class"]].copy()
    frame = aligned.merge(
        context, on=["site_id", "date"], how="left", validate="many_to_one"
    )
    missing = frame["current_class"].isna() | frame["current_class"].eq("").fillna(
        False
    )
    frame["issue_year"] = pd.to_datetime(
        frame["date"], format="%Y-%m-%d"
    ).dt.year.astype(str)
    frame["persistence_context"] = np.where(
        missing, "majority_fallback", "current_class_available"
    )
    frame["transition"] = np.where(
        missing,
        "current_class_missing",
        np.where(
            frame["current_class"].eq(frame["y_true"]).fillna(False),
            "same_class",
            "changed_class",
        ),
    )
    output = []
    for model, model_rows in frame.groupby("model", sort=True):
        split = model_rows["split"].iloc[0]
        for group_type in [
            "issue_year",
            "site_id",
            "persistence_context",
            "transition",
        ]:
            for group_value, rows in model_rows.groupby(group_type, sort=True):
                result = score(rows["y_true"], rows["y_pred"])
                supports = {
                    label: result["per_class"][label]["support"] for label in CLASSES
                }
                output.append(
                    {
                        "model": model,
                        "split": split,
                        "group_type": group_type,
                        "group_value": str(group_value),
                        "n": result["n"],
                        "macro_f1": result["macro_f1"],
                        "accuracy": result["accuracy"],
                        **{f"{label}_support": supports[label] for label in CLASSES},
                        "absent_true_classes": json.dumps(
                            [label for label in CLASSES if supports[label] == 0]
                        ),
                    }
                )
    return pd.DataFrame(output, columns=DIAGNOSTIC_COLUMNS)


def select_timeline_sites(aligned, reference):
    """Pick by eligible key count and fixed longitude bins, never outcomes."""
    keys = aligned[["site_id", "date"]].drop_duplicates()
    sites = keys.merge(
        reference[["site_id", "date", "nominal_lon"]],
        on=["site_id", "date"],
        validate="one_to_one",
    )
    sites["longitude_east"] = pd.to_numeric(sites["nominal_lon"], errors="raise") % 360
    selected = []
    for region, lower, upper in REGIONS:
        candidates = sites.loc[
            sites["longitude_east"].ge(lower) & sites["longitude_east"].lt(upper)
        ]
        counts = (
            candidates.groupby("site_id", sort=True).size().rename("n").reset_index()
        )
        counts = counts.sort_values(
            ["n", "site_id"], ascending=[False, True], kind="stable"
        )
        selected.append(
            {
                "region": region,
                "site_id": None if counts.empty else str(counts.iloc[0]["site_id"]),
                "eligible_validation_n": 0
                if counts.empty
                else int(counts.iloc[0]["n"]),
                "reason": "no_eligible_site" if counts.empty else None,
            }
        )
    return selected


def select_model(metrics, split, config, audit, timeline_sites, frozen_selection=None):
    """Choose on validation only; copy that choice unchanged for test."""
    if split == "test":
        if frozen_selection is None or frozen_selection.get("status") != "frozen_ready":
            raise ValueError(
                "Test reporting requires the verified frozen validation selection"
            )
        if frozen_selection["selected_model"] not in metrics:
            raise ValueError("Frozen model is absent from the test comparison")
        result = copy.deepcopy(frozen_selection)
        result.update(split="test", status="test_assessment")
        return result
    results = {name: by_split[split] for name, by_split in metrics.items()}
    best = max(result["macro_f1"] for result in results.values())
    tied = sorted(
        name
        for name, result in results.items()
        if best - result["macro_f1"] <= TOLERANCE
    )
    extreme = {
        name: min(
            results[name]["per_class"][c]["recall"] for c in ["cooling", "warming"]
        )
        for name in tied
    }
    best_recall = max(extreme.values())
    recall_tied = sorted(
        name for name in tied if best_recall - extreme[name] <= TOLERANCE
    )
    winner = min(recall_tied, key=MODEL_ORDER.index)
    tradeoffs = []
    if "persistence" in results:
        for label in ["cooling", "warming"]:
            selected = results[winner]["per_class"][label]["recall"]
            baseline = results["persistence"]["per_class"][label]["recall"]
            if selected - baseline < -TOLERANCE:
                tradeoffs.append(
                    {
                        "class": label,
                        "selected_recall": selected,
                        "persistence_recall": baseline,
                        "delta": selected - baseline,
                    }
                )
    ready = (
        config["dataset_kind"] == "final"
        and config["target_contract"]["status"] == "team_final"
        and set(results) == set(MODEL_ORDER)
        and audit.get("final_comparison_ready") is True
    )
    return {
        "schema_version": 1,
        "rule_id": RULE_ID,
        "split": split,
        "status": "frozen_ready" if ready else "provisional",
        "selected_model": winner,
        "macro_f1_tied_models": tied,
        "recall_tied_models": recall_tied,
        "recall_tradeoffs": tradeoffs,
        "timeline_sites": copy.deepcopy(timeline_sites),
    }


def timeline_table(aligned, selected_sites):
    frames = []
    for site in selected_sites:
        if site["site_id"] is None:
            continue
        frame = aligned.loc[aligned["site_id"].eq(site["site_id"])].copy()
        frame.insert(0, "region", site["region"])
        frames.append(
            frame[TIMELINE_COLUMNS].sort_values(
                ["site_id", "target_date", "model"], kind="stable"
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=TIMELINE_COLUMNS)
    )


def normalized_confusion(matrix):
    values = np.asarray(matrix, dtype=float)
    support = values.sum(axis=1, keepdims=True)
    return np.divide(
        values, support, out=np.full(values.shape, np.nan), where=support != 0
    )


def timeline_grid(rows, models, dates):
    """Daily categorical grid with unobserved dates left NaN (no interpolation)."""
    frame = rows.copy()
    if frame.empty:
        return np.full((len(models) + 1, len(dates)), np.nan)
    consistency = frame.groupby(["site_id", "date"], sort=False)[
        ["target_date", "observed_target_date", "y_true"]
    ].nunique(dropna=False)
    if consistency.gt(1).any().any():
        raise ValueError("Timeline models disagree on shared target truth or dates")
    observation = frame.drop_duplicates(["site_id", "date"])
    if observation["target_date"].duplicated().any():
        raise ValueError("Timeline contains duplicate requested target dates")
    grid = np.full((len(models) + 1, len(dates)), np.nan)
    mapping = {label: index for index, label in enumerate(CLASSES)}

    def put(index, part, column):
        positions = dates.get_indexer(
            pd.to_datetime(part["target_date"], format="%Y-%m-%d")
        )
        if (positions < 0).any():
            raise ValueError(
                "Timeline target date falls outside the selected date grid"
            )
        grid[index, positions] = part[column].map(mapping).to_numpy(dtype=float)

    put(0, observation, "y_true")
    for index, model in enumerate(models, 1):
        put(index, frame.loc[frame["model"].eq(model)], "y_pred")
    return grid


def _plot_reports(output_dir, board, metrics, split, timeline, selection):
    # Imported only when figures are requested. Metrics modules need no GUI backend.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    files = []

    def save(fig, name):
        try:
            fig.savefig(figures / name, dpi=160, bbox_inches="tight")
            files.append(f"figures/{name}")
        finally:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3 + 0.35 * len(board)))
    ax.barh(board["model"], board["macro_f1"], color="#356985")
    ax.invert_yaxis()
    ax.set(
        xlim=(0, 1.08),
        xlabel="Macro F1 (fixed three classes)",
        title=f"{split.title()} comparison",
    )
    for index, value in enumerate(board["macro_f1"]):
        ax.text(value + 0.015, index, f"{value:.3f}", va="center")
    save(fig, "macro_f1.png")

    count_max = max(
        1,
        max(
            np.max(result[split]["confusion_matrix"]["rows_true_cols_pred"])
            for result in metrics.values()
        ),
    )
    for model in sorted(metrics):
        counts = np.asarray(
            metrics[model][split]["confusion_matrix"]["rows_true_cols_pred"], dtype=int
        )
        for normalized in [False, True]:
            values = normalized_confusion(counts) if normalized else counts
            fig, ax = plt.subplots(figsize=(6.4, 5))
            cmap = plt.get_cmap("Blues").with_extremes(bad="white")
            plotted = ax.imshow(
                np.ma.masked_invalid(values),
                cmap=cmap,
                vmin=0,
                vmax=1 if normalized else count_max,
            )
            fig.colorbar(
                plotted,
                ax=ax,
                label="Fraction of actual class" if normalized else "Forecast count",
            )
            ax.set(
                xticks=range(3),
                yticks=range(3),
                xticklabels=CLASSES,
                yticklabels=CLASSES,
                xlabel="Predicted class",
                ylabel="Actual class",
                title=f"{model} · {split}\n{'Row-normalized' if normalized else 'Count'} confusion matrix",
            )
            for row in range(3):
                for col in range(3):
                    value = values[row, col]
                    label = (
                        "N/A\n(no true samples)"
                        if not np.isfinite(value)
                        else (f"{value:.2f}" if normalized else str(int(value)))
                    )
                    ax.text(
                        col,
                        row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8 if not np.isfinite(value) else 11,
                        color="white"
                        if np.isfinite(value)
                        and value > (0.6 if normalized else 0.6 * count_max)
                        else "black",
                    )
            save(
                fig,
                f"{model}_confusion_{'row_normalized' if normalized else 'counts'}.png",
            )

    models = sorted(metrics)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), squeeze=False)
    category_cmap = ListedColormap(["#2976ad", "#b5b5b5", "#d35b35"]).with_extremes(
        bad="white"
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], category_cmap.N)
    if not timeline.empty:
        dates = pd.date_range(
            timeline["target_date"].min(), timeline["target_date"].max(), freq="D"
        )
    else:
        dates = pd.DatetimeIndex([])
    for ax, chosen in zip(axes[:, 0], selection["timeline_sites"], strict=True):
        rows = timeline.loc[timeline["region"].eq(chosen["region"])]
        if chosen["site_id"] is None or rows.empty:
            ax.text(
                0.5,
                0.5,
                f"{chosen['region'].title()}: "
                + (
                    "no eligible validation site"
                    if chosen["site_id"] is None
                    else f"{chosen['site_id']} has no eligible {split} forecasts"
                ),
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        grid = timeline_grid(rows, models, dates)
        x = mdates.date2num(dates.to_pydatetime())
        ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            interpolation="nearest",
            cmap=category_cmap,
            norm=norm,
            extent=[x[0] - 0.5, x[-1] + 0.5, len(models) + 0.5, -0.5],
        )
        ax.set_yticks(range(len(models) + 1), ["Observed target"] + models)
        ax.xaxis_date()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        n = rows[["site_id", "date"]].drop_duplicates().shape[0]
        ax.set_title(
            f"{chosen['region'].title()} · {chosen['site_id']} · {n} eligible {split} forecasts; white = no forecast",
            fontsize=10,
        )
        ax.set_xlabel("Requested target date (forecast issued 90 days earlier)")
    fig.suptitle(f"Local anomaly forecast categories · {split}", fontsize=14)
    fig.legend(
        handles=[
            Patch(facecolor=category_cmap(i), label=label)
            for i, label in enumerate(CLASSES)
        ]
        + [Patch(facecolor="white", edgecolor="grey", label="Missing coverage")],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.text(
        0.5,
        0.015,
        "Verifying observations occur 90–97 days after issue. Sites selected by validation coverage within fixed longitude bins.\nForecast-horizon alignment is not event-onset lead time; gaps are not interpolated.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    save(fig, "local_condition_timeline.png")
    return files


def _markdown_table(frame, columns):
    labels = list(columns)
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in frame[labels].itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("N/A")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _report(config, audit, board, per_class, diagnostics, aligned, selection, warnings):
    split = config["evaluation_split"]
    counts = audit["splits"][split]
    contract = config["target_contract"]
    text = [
        f"# Evaluation: {config['run_id']}",
        f"Dataset: **{config['dataset_kind']}**. Split: **{split}**. Target contract: `{contract['id']}`.",
        f"Forecast issue period: {aligned['date'].min()} through {aligned['date'].max()}. The target is a local temperature-anomaly category at 90 days, with the first same-site nonmissing class accepted up to 7 days later; actual label lead is 90–97 days. Threshold endpoints ±0.5 °C are neutral.",
        f"Coverage: {counts['eligible_n']:,} eligible forecasts / {counts['reference_n']:,} supplied assigned-site reference forecasts in {split}. This denominator does not include unassigned raw observations. Every compared model covers every eligible key.",
        "Fit protocol: "
        + (
            "eligible train observations only; model recipes selected using training-period chronological CV for project data."
            if split == "validation"
            else "eligible train + validation observations, using the already frozen validation-selected recipe; shared climatology remains train-only."
        ),
        "## Model comparison",
        _markdown_table(board, LEADERBOARD_COLUMNS),
        "Stored metrics retain full precision. Displayed values are rounded to three decimals. F1 deltas are absolute F1 units, not relative percentages.",
        "![Macro F1](figures/macro_f1.png)",
        "## Selection",
        f"Rule: `{selection['rule_id']}`. Status: **{selection['status']}**. Selected model: **{selection['selected_model']}**.",
    ]
    if split == "test":
        text.append(
            "This is assessment of the frozen validation choice. Test scores do not reselect the model or timeline sites; tie and recall-tradeoff records below refer to validation."
        )
    if len(selection["macro_f1_tied_models"]) > 1:
        text.append(
            "Macro F1 was numerically tied among "
            + ", ".join(selection["macro_f1_tied_models"])
            + ". Selection uses minimum warming/cooling recall, then the predetermined simplicity order; this is not a Macro F1 improvement claim."
        )
    if selection["selected_model"] == "majority":
        text.append(
            "The constant benchmark was selected. This experiment has not established useful learned early-warning skill."
        )
    for tradeoff in selection["recall_tradeoffs"]:
        text.append(
            f"Recall tradeoff: {tradeoff['class']} recall {tradeoff['selected_recall']:.3f} versus persistence {tradeoff['persistence_recall']:.3f} (delta {tradeoff['delta']:.3f})."
        )
    text.extend(["## Per-class results", _markdown_table(per_class, PER_CLASS_COLUMNS)])
    for model in sorted(aligned["model"].unique()):
        text.extend(
            [
                f"### {model}",
                f"![{model} counts](figures/{model}_confusion_counts.png)",
                f"![{model} normalized](figures/{model}_confusion_row_normalized.png)",
            ]
        )
    text.extend(
        [
            "Matrices use actual rows and predicted columns in cooling/neutral/warming order. Unsupported true classes retain zero F1 under the fixed-label policy; their normalized confusion rows are undefined and marked N/A.",
            "## Diagnostics",
            _markdown_table(
                diagnostics.loc[diagnostics["group_type"].eq("issue_year")],
                DIAGNOSTIC_COLUMNS,
            ),
            "[All site, year, persistence-context and transition slices](diagnostics.csv) include sample counts and per-class support. All models use the same subsets. Tiny groups and absent classes make site rankings unreliable. Transition groups use future truth retrospectively and are not known at issue time.",
            "## Local-condition timeline",
            "![Local forecast timeline](figures/local_condition_timeline.png)",
            "Sites are chosen by eligible validation coverage within fixed display longitude bins, with site ID breaking ties. The test report reuses those choices. [Timeline records](timeline.csv) retain issue, requested and actual observed dates. Forecast-horizon alignment is not event-onset lead time. These are local category forecasts, not official ENSO events, independent alert events, or measured operational warning lead times.",
            "## Eligibility and provenance",
            "Requested and observed target dates must both stay inside their own split. Filtering report rows cannot repair a model trained using a future split's labels.",
        ]
    )
    for name, values in audit["splits"].items():
        reasons = (
            ", ".join(
                f"{reason}: {count}"
                for reason, count in sorted(values["primary_exclusions"].items())
            )
            or "none"
        )
        text.append(
            f"{name}: reference={values['reference_n']}, eligible={values['eligible_n']}, excluded={values['excluded_n']}; primary reasons: {reasons}."
        )
    text.extend(
        [
            "[Audit](audit.json) records checks actually performed; [manifest](manifest.json) records source identity, versions, input/receipt/output hashes, and the exact forecast population. Receipts and file checks alone cannot prove how every model or feature was constructed. Explicit team agreement, upstream U01–U06 tests and chronological tuning evidence remain producer responsibilities for project results.",
            "Project-history limitation: earlier notebooks examined late-year data and the baseline PR used validation for tuning. The revised protocol does not erase that exposure. Canonical sites also incorporate retrospective record-based decisions; these results do not establish a fully online site-discovery system.",
            "## SHAP",
            (
                "Status: **not_run**. LightGBM was included and can be explained even if another model won. Run the separate explanation command, then read its [separate report](shap/report.md); this immutable metric bundle does not certify explanation completion."
                if "lightgbm" in set(aligned["model"])
                else "Status: **blocked**: LightGBM is absent from this evaluated model set. No explanation result has been fabricated."
            ),
            "## Limits and remaining work",
        ]
    )
    if config["dataset_kind"] == "synthetic":
        text.append(
            "This run uses synthetic fixtures. It is software-verification evidence only, not a project-data forecast result; its fixture fitting recipe is not evidence that project CV or team agreement is complete."
        )
    elif selection["status"] == "provisional":
        text.append(
            "This is a development/interim comparison. The final integrated dataset, full model set and verified upstream evidence remain necessary before final model selection."
        )
    missing_models = [name for name in MODEL_ORDER if name not in set(aligned["model"])]
    if missing_models:
        text.append(
            "Missing final comparison models: " + ", ".join(missing_models) + "."
        )
    text.extend(str(warning) for warning in warnings)
    text.append(
        "No statistical significance, calibration, causal effects, supply-chain savings, or deployment utility is established here. One-hot baseline columns encode deterministic predictions. Additional calibration, event detection, ablations, and operational thresholds are deferred."
    )
    return "\n\n".join(text) + "\n"


def write_reports(
    aligned, reference, audit, config, metrics, output_dir, frozen_selection=None
):
    """Write reports only after the caller validates inputs and fitting evidence."""
    output_dir = Path(output_dir)
    split = config["evaluation_split"]
    board, per_class = metric_tables(metrics, split, audit)
    diagnostics = diagnostic_table(aligned, reference)
    sites = (
        frozen_selection["timeline_sites"]
        if split == "test" and frozen_selection is not None
        else select_timeline_sites(aligned, reference)
    )
    selection = select_model(metrics, split, config, audit, sites, frozen_selection)
    timeline = timeline_table(aligned, selection["timeline_sites"])
    warnings = [
        "Fill-status diagnostic deferred: no proven fill-log-to-model-table linkage was supplied."
    ]
    if "persistence" not in metrics:
        warnings.append(
            "Persistence comparison unavailable in this development fixture; delta Macro F1 is blank."
        )
    for model in sorted(metrics):
        absent = [
            c for c in CLASSES if metrics[model][split]["per_class"][c]["support"] == 0
        ]
        if absent:
            warnings.append(
                f"{model}/{split}: absent true classes {', '.join(absent)} retain zero F1 in the fixed three-class macro average."
            )
    files = []
    for name, frame in [
        ("leaderboard.csv", board),
        ("per_class_metrics.csv", per_class),
        ("diagnostics.csv", diagnostics),
        ("timeline.csv", timeline),
    ]:
        frame.to_csv(
            output_dir / name, index=False, lineterminator="\n", float_format="%.17g"
        )
        files.append(name)
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    files.append("selection.json")
    files.extend(_plot_reports(output_dir, board, metrics, split, timeline, selection))
    (output_dir / "report.md").write_text(
        _report(
            config, audit, board, per_class, diagnostics, aligned, selection, warnings
        ),
        encoding="utf-8",
    )
    files.append("report.md")
    return {"selection": selection, "warnings": warnings, "files": files}
