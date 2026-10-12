from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from timesigma.algorithms import EASYTSAD_SCHEMAS, METHOD_CATALOG

from .runner import leaderboard_items
from .zero_shot import (
    MEMBERSHIP_FILENAME,
    ZeroShotSplit,
    load_zero_shot_membership,
    split_for_dataset,
    validate_zero_shot_scores,
)


OFFICIAL_METRIC_IDS = (
    "point-wise-f1",
    "reduced-length-f1",
    "event-wise-f1",
    "delay-reduced-length-f1",
    "point-wise-auprc",
    "reduced-length-auprc",
    "event-wise-auprc",
    "delay-reduced-length-auprc",
)


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    name: str
    summary_column: str


@dataclass(frozen=True)
class SummaryEntry:
    method: str
    schema: str
    dataset: str
    status: str
    values: tuple[float, ...] | None
    reason: str | None
    summary_path: Path | None

    def value(self, metric_id: str) -> float | None:
        if self.values is None:
            return None
        return self.values[OFFICIAL_METRIC_IDS.index(metric_id)]


def metric_definitions(dataset: str) -> tuple[MetricDefinition, ...]:
    try:
        items = leaderboard_items(dataset)
    except KeyError as exc:
        raise ValueError(f"unsupported EasyTSAD leaderboard dataset: {dataset}") from exc
    if len(items) != len(OFFICIAL_METRIC_IDS):
        raise RuntimeError("EasyTSAD leaderboard metric configuration is incomplete")
    return tuple(
        MetricDefinition(metric_id, name, path[0])
        for metric_id, (name, path) in zip(OFFICIAL_METRIC_IDS, items)
    )


def build_leaderboard(
    *,
    results_dirs: Sequence[Path],
    schema: str,
    dataset: str,
    metric_id: str,
) -> dict[str, object]:
    if schema not in EASYTSAD_SCHEMAS:
        raise ValueError(f"unsupported EasyTSAD schema: {schema}")
    definitions = metric_definitions(dataset)
    definitions_by_id = {item.metric_id: item for item in definitions}
    if metric_id not in definitions_by_id:
        raise ValueError(f"unsupported EasyTSAD leaderboard metric: {metric_id}")
    if not results_dirs:
        raise ValueError("at least one EasyTSAD Results directory is required")

    entries: dict[str, SummaryEntry] = {}
    found_dataset = False
    frozen_split: ZeroShotSplit | None = None
    for results_dir in (path.resolve() for path in results_dirs):
        summary_path = results_dir / "Summary" / "CSVs" / f"{schema}.csv"
        if not summary_path.is_file():
            continue
        source_entries = _read_summary(summary_path, schema, dataset, definitions)
        if source_entries is None:
            continue
        found_dataset = True

        split = None
        if schema == "zero_shot":
            splits = load_zero_shot_membership(
                results_dir / MEMBERSHIP_FILENAME
            )
            split = split_for_dataset(splits, dataset)
            if frozen_split is not None and split != frozen_split:
                raise ValueError(
                    f"zero-shot membership differs across Results directories: {dataset}"
                )
            frozen_split = split

        for entry in source_entries:
            if entry.method in entries:
                raise ValueError(
                    f"duplicate EasyTSAD Summary result: "
                    f"{entry.method}/{schema}/{dataset}"
                )
            if split is not None and entry.status == "ok":
                validate_zero_shot_scores(results_dir, (entry.method,), (split,))
            entries[entry.method] = entry

    if not found_dataset:
        raise ValueError(
            f"dataset {dataset} was not found in the {schema} EasyTSAD Summary files"
        )

    for method, spec in METHOD_CATALOG.items():
        if method in entries:
            continue
        capability = spec.capability(schema)
        entries[method] = SummaryEntry(
            method=method,
            schema=schema,
            dataset=dataset,
            status="missing" if capability.supported else "N/A",
            values=None,
            reason=(
                "method was not run for this dataset"
                if capability.supported
                else capability.unavailable_reason
            ),
            summary_path=None,
        )

    selected = definitions_by_id[metric_id]
    ordered = sorted(
        entries.values(),
        key=lambda entry: (
            0 if entry.status == "ok" else 1 if entry.status == "N/A" else 2,
            -(entry.value(metric_id) or 0.0),
            entry.method,
        ),
    )
    rank = 0
    rows = []
    for entry in ordered:
        if entry.status == "ok":
            rank += 1
        rows.append(
            {
                "rank": rank if entry.status == "ok" else None,
                "method": entry.method,
                "status": entry.status,
                "value": entry.value(metric_id),
                "reason": entry.reason,
                "summary_path": (
                    str(entry.summary_path) if entry.summary_path is not None else None
                ),
            }
        )

    return {
        "source": "EasyTSAD Summary",
        "schema": schema,
        "dataset": dataset,
        "metric": {
            "id": selected.metric_id,
            "name": selected.name,
            "summary_column": selected.summary_column,
        },
        "zero_shot_membership": (
            frozen_split.as_dict() if frozen_split is not None else None
        ),
        "rows": rows,
    }


def _read_summary(
    path: Path,
    schema: str,
    dataset: str,
    definitions: Sequence[MetricDefinition],
) -> tuple[SummaryEntry, ...] | None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) != len(rows[1]):
        raise ValueError(f"invalid EasyTSAD Summary CSV: {path}")

    indices = []
    for definition in definitions:
        matches = [
            index
            for index, (dataset_name, column) in enumerate(
                zip(rows[0], rows[1])
            )
            if dataset_name == dataset and column == definition.summary_column
        ]
        if len(matches) != 1:
            if not matches and dataset not in rows[0]:
                return None
            raise ValueError(
                f"EasyTSAD Summary column is not unique: "
                f"{dataset}/{definition.summary_column}"
            )
        indices.append(matches[0])

    entries = []
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        if len(row) != len(rows[0]):
            raise ValueError(f"invalid EasyTSAD Summary row: {path}")
        method = row[0].strip()
        values = [row[index].strip() for index in indices]
        missing = [value == "Not Founded" for value in values]
        if any(missing) and not all(missing):
            raise ValueError(f"partial EasyTSAD Summary result: {method}/{dataset}")

        capability = (
            METHOD_CATALOG[method].capability(schema)
            if method in METHOD_CATALOG
            else None
        )
        if all(missing):
            status = (
                "N/A"
                if capability is not None and not capability.supported
                else "missing"
            )
            reason = (
                capability.unavailable_reason
                if status == "N/A"
                else "EasyTSAD Summary reports Not Founded"
            )
            numeric_values = None
        else:
            if capability is not None and not capability.supported:
                raise ValueError(
                    f"EasyTSAD Summary contains a result for N/A schema: "
                    f"{method}/{schema}"
                )
            try:
                numeric_values = tuple(float(value) for value in values)
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric EasyTSAD Summary value: {method}/{dataset}"
                ) from exc
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(
                    f"non-finite EasyTSAD Summary value: {method}/{dataset}"
                )
            status = "ok"
            reason = None

        entries.append(
            SummaryEntry(
                method=method,
                schema=schema,
                dataset=dataset,
                status=status,
                values=numeric_values,
                reason=reason,
                summary_path=path,
            )
        )
    return tuple(entries)
