import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-occt")
import matplotlib.pyplot as plt


def _parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        label = Path(raw_path).parent.name or Path(raw_path).stem
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid input spec '{spec}': missing label before '='.")
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return label, path


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_summary_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"Summary CSV is empty: {path}")
    return {row["group_name"]: row for row in rows if row.get("group_name")}


def _build_metric_matrix(
    inputs: list[tuple[str, Path]],
    metric: str,
    normalize_by_dim: bool,
) -> tuple[list[str], list[str], np.ndarray]:
    per_label_rows = []
    all_groups = set()
    for label, path in inputs:
        rows = _load_summary_csv(path)
        per_label_rows.append((label, rows))
        all_groups.update(rows.keys())

    labels = [label for label, _ in per_label_rows]
    groups = sorted(all_groups)
    matrix = np.full((len(groups), len(labels)), np.nan, dtype=float)

    for col_idx, (_, rows) in enumerate(per_label_rows):
        for row_idx, group_name in enumerate(groups):
            row = rows.get(group_name)
            if row is None:
                continue
            if metric not in row:
                available = ", ".join(row.keys())
                raise KeyError(
                    f"Metric '{metric}' is not present in {inputs[col_idx][1]}. "
                    f"Available columns: {available}"
                )
            value = _safe_float(row.get(metric))
            if normalize_by_dim:
                group_dim = _safe_float(row.get("group_dim"))
                if group_dim > 0:
                    value = value / group_dim
            matrix[row_idx, col_idx] = value
    return labels, groups, matrix


def _order_groups(
    groups: list[str],
    matrix: np.ndarray,
    sort_by: str,
    reference_label: str | None,
    labels: list[str],
    max_groups: int | None,
) -> tuple[list[str], np.ndarray]:
    if reference_label is not None:
        if reference_label not in labels:
            raise ValueError(
                f"reference label '{reference_label}' not found. Available labels: {labels}"
            )
        sort_by = "reference"

    if sort_by == "mean":
        scores = np.nanmean(matrix, axis=1)
    elif sort_by == "max":
        scores = np.nanmax(matrix, axis=1)
    elif sort_by == "reference":
        if reference_label is None:
            raise ValueError("reference-label is required when sort-by=reference.")
        ref_idx = labels.index(reference_label)
        scores = matrix[:, ref_idx]
    else:
        raise ValueError(f"Unsupported sort mode '{sort_by}'.")

    order = np.argsort(np.nan_to_num(scores, nan=-math.inf))[::-1]
    if max_groups is not None:
        order = order[:max_groups]
    ordered_groups = [groups[idx] for idx in order]
    ordered_matrix = matrix[order]
    return ordered_groups, ordered_matrix


def _format_metric_name(metric: str, normalize_by_dim: bool) -> str:
    return f"{metric}_per_dim" if normalize_by_dim else metric


def _save_metric_matrix_csv(
    output_path: Path,
    groups: list[str],
    labels: list[str],
    matrix: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["group_name", *labels])
        for group_name, row in zip(groups, matrix, strict=True):
            writer.writerow([group_name, *row.tolist()])


def _plot_heatmap(
    output_path: Path,
    labels: list[str],
    groups: list[str],
    matrix: np.ndarray,
    metric_label: str,
    use_log_color: bool,
) -> None:
    plot_matrix = np.array(matrix, copy=True)
    colorbar_label = metric_label
    if use_log_color:
        positive_mask = plot_matrix > 0
        plot_matrix[positive_mask] = np.log10(plot_matrix[positive_mask])
        colorbar_label = f"log10({metric_label})"

    masked_matrix = np.ma.masked_invalid(plot_matrix)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#e8eaed")

    fig_width = max(5.0, 1.5 * len(labels))
    fig_height = max(4.0, 0.38 * len(groups))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(masked_matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Observation Construction")
    ax.set_ylabel("Observation Group")
    ax.set_title(f"Observation Change Comparison: {metric_label}")

    if len(groups) <= 16 and len(labels) <= 6:
        for row_idx in range(len(groups)):
            for col_idx in range(len(labels)):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    text = "NA"
                else:
                    text = f"{value:.2e}" if abs(value) < 1e-2 else f"{value:.3f}"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bar(
    output_path: Path,
    labels: list[str],
    groups: list[str],
    matrix: np.ndarray,
    metric_label: str,
    use_log_y: bool,
) -> None:
    x = np.arange(len(groups), dtype=float)
    width = min(0.8 / max(len(labels), 1), 0.28)
    fig_width = max(6.0, 0.7 * len(groups) + 1.4 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)

    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 3)))
    for idx, label in enumerate(labels):
        offsets = x + (idx - (len(labels) - 1) / 2) * width
        ax.bar(
            offsets,
            matrix[:, idx],
            width=width,
            label=label,
            color=cmap[idx % len(cmap)],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Observation Group")
    ax.set_title(f"Top Observation Groups by {metric_label}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    if use_log_y:
        ax.set_yscale("log")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_bar(
    output_path: Path,
    labels: list[str],
    matrix: np.ndarray,
    metric_label: str,
) -> None:
    overall = np.nanmean(matrix, axis=0)
    fig_width = max(5.0, 1.3 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 3.6), constrained_layout=True)
    cmap = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    ax.bar(labels, overall, color=cmap[: len(labels)], alpha=0.9)
    ax.set_ylabel(f"Mean {metric_label}")
    ax.set_xlabel("Observation Construction")
    ax.set_title(f"Overall Observation Change Level: {metric_label}")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare observation-change statistics across different observation "
            "construction inputs using summary CSVs."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more comparison inputs in LABEL=PATH form. "
            "PATH should point to observation_group_jump_summary.csv."
        ),
    )
    parser.add_argument(
        "--metric",
        default="delta_variance",
        help=(
            "Column to visualize, for example delta_variance, abs_delta_variance, "
            "step_max_abs_delta_variance, mean_abs_delta, or p99_abs_delta."
        ),
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=12,
        help="Number of observation groups to keep after sorting.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("mean", "max", "reference"),
        default="mean",
        help="How to rank observation groups before truncating to max-groups.",
    )
    parser.add_argument(
        "--reference-label",
        type=str,
        default=None,
        help="Reference label used when sort-by=reference.",
    )
    parser.add_argument(
        "--normalize-by-dim",
        action="store_true",
        help="Divide the selected metric by group_dim before plotting.",
    )
    parser.add_argument(
        "--log-color",
        action="store_true",
        help="Apply log10 scaling to heatmap colors.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a log scale on the grouped bar chart y-axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "observation_variance_plots",
        help="Directory used to save generated plots and the comparison matrix CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [_parse_input_spec(spec) for spec in args.inputs]
    labels, groups, matrix = _build_metric_matrix(
        inputs=inputs,
        metric=args.metric,
        normalize_by_dim=args.normalize_by_dim,
    )
    ordered_groups, ordered_matrix = _order_groups(
        groups=groups,
        matrix=matrix,
        sort_by=args.sort_by,
        reference_label=args.reference_label,
        labels=labels,
        max_groups=args.max_groups,
    )

    metric_label = _format_metric_name(args.metric, args.normalize_by_dim)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv_path = output_dir / f"observation_metric_matrix_{metric_label}.csv"
    heatmap_path = output_dir / f"observation_metric_heatmap_{metric_label}.png"
    topk_bar_path = output_dir / f"observation_metric_topk_{metric_label}.png"
    overall_bar_path = output_dir / f"observation_metric_overall_{metric_label}.png"

    _save_metric_matrix_csv(matrix_csv_path, ordered_groups, labels, ordered_matrix)
    _plot_heatmap(
        heatmap_path,
        labels=labels,
        groups=ordered_groups,
        matrix=ordered_matrix,
        metric_label=metric_label,
        use_log_color=args.log_color,
    )
    _plot_grouped_bar(
        topk_bar_path,
        labels=labels,
        groups=ordered_groups,
        matrix=ordered_matrix,
        metric_label=metric_label,
        use_log_y=args.log_y,
    )
    _plot_overall_bar(
        overall_bar_path,
        labels=labels,
        matrix=ordered_matrix,
        metric_label=metric_label,
    )

    print(f"Saved matrix CSV: {matrix_csv_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved top-k bar chart: {topk_bar_path}")
    print(f"Saved overall bar chart: {overall_bar_path}")


if __name__ == "__main__":
    main()
