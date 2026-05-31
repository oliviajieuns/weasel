"""Run WEASEL greedy subset selection from precomputed score records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select trajectory steps with the WEASEL greedy objective."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Goal-record JSON produced by weasel.prepare_scores.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON containing selected dataset indices per trajectory.",
    )
    parser.add_argument(
        "--annotated-output",
        default=None,
        help="Optional copy of the input records with selected indices attached.",
    )
    parser.add_argument(
        "--score-key",
        default="bert_scores_obs_history_norm",
        help="Goal-record score key used as the unary importance term.",
    )
    parser.add_argument(
        "--distance-key",
        default="distance_matrix",
        help="Trajectory-group key containing the pairwise WEASEL distance matrix.",
    )
    parser.add_argument(
        "--t0-mode",
        choices=["fixed", "percentage"],
        default="fixed",
        help="How to choose the per-trajectory selection budget.",
    )
    parser.add_argument(
        "--t0-fixed",
        type=int,
        default=3,
        help="Fixed number of steps selected per trajectory.",
    )
    parser.add_argument(
        "--t0-percentage",
        type=float,
        default=0.25,
        help="Fraction of trajectory steps selected when --t0-mode=percentage.",
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=1.0,
        help="Tradeoff weight for pairwise diversity.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write one flat list of selected dataset indices instead of grouped lists.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def write_json(data: Any, path: Path, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_indent = None if indent == 0 else indent
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=actual_indent)


def compute_t0(length: int, mode: str, fixed: int, percentage: float) -> int:
    if length <= 0:
        return 0
    if mode == "fixed":
        value = fixed
    elif mode == "percentage":
        value = math.ceil(length * percentage)
    else:
        raise ValueError(f"Unsupported t0 mode: {mode}")
    return max(1, min(length, value))


def validate_square_matrix(matrix: Sequence[Sequence[float]], n: int) -> None:
    if len(matrix) != n:
        raise ValueError(f"Expected a {n}x{n} distance matrix, got {len(matrix)} rows.")
    for row in matrix:
        if len(row) != n:
            raise ValueError(f"Expected a {n}x{n} distance matrix.")


def greedy_select(
    importance: Sequence[float],
    distance: Sequence[Sequence[float]],
    t0: int,
    lambda_weight: float,
) -> List[int]:
    """Return selected local indices sorted by original trajectory order."""
    n = len(importance)
    if n == 0 or t0 <= 0:
        return []
    if n <= t0:
        return list(range(n))
    if t0 == 1:
        return [max(range(n), key=lambda i: importance[i])]

    validate_square_matrix(distance, n)

    best_pair: Tuple[int, int] = (0, 1)
    best_value = float("-inf")
    for i in range(n):
        for j in range(i + 1, n):
            value = importance[i] + importance[j] + lambda_weight * distance[i][j]
            if value > best_value:
                best_value = value
                best_pair = (i, j)

    selected: Set[int] = set(best_pair)
    while len(selected) < t0:
        best_idx: Optional[int] = None
        best_gain = float("-inf")

        for candidate in range(n):
            if candidate in selected:
                continue
            diversity_gain = sum(distance[candidate][idx] for idx in selected)
            gain = importance[candidate] + lambda_weight * diversity_gain
            if gain > best_gain:
                best_gain = gain
                best_idx = candidate

        if best_idx is None:
            break
        selected.add(best_idx)

    return sorted(selected)


def segment_importance(
    record: Dict[str, Any],
    score_key: str,
    segment_indices: Sequence[int],
) -> List[float]:
    all_indices = record.get("dataset_indices", [])
    all_scores = record.get(score_key, [])
    index_to_score = {
        dataset_idx: float(all_scores[pos])
        for pos, dataset_idx in enumerate(all_indices)
        if pos < len(all_scores)
    }
    return [index_to_score.get(dataset_idx, 0.0) for dataset_idx in segment_indices]


def select_from_records(args: argparse.Namespace) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    records = load_json(Path(args.input))
    selected_groups: List[List[int]] = []

    for record_idx, record in enumerate(records):
        trajectory_groups = record.get("trajectory_groups", [])
        if not isinstance(trajectory_groups, list):
            continue

        for group_idx, group in enumerate(trajectory_groups):
            segment_indices = group.get("dataset_indices", [])
            distance = group.get(args.distance_key)

            if not segment_indices:
                group["selected_dataset_indices"] = []
                continue
            if distance is None:
                raise ValueError(
                    f"Missing {args.distance_key!r} for record {record_idx}, "
                    f"trajectory group {group_idx}."
                )

            t0 = compute_t0(
                len(segment_indices),
                args.t0_mode,
                args.t0_fixed,
                args.t0_percentage,
            )
            importance = segment_importance(record, args.score_key, segment_indices)
            local_selected = greedy_select(
                importance=importance,
                distance=distance,
                t0=t0,
                lambda_weight=args.lambda_weight,
            )
            selected_dataset_indices = [
                segment_indices[local_idx] for local_idx in local_selected
            ]

            group["selected_dataset_indices"] = selected_dataset_indices
            group["selection_metadata"] = {
                "t0": t0,
                "score_key": args.score_key,
                "distance_key": args.distance_key,
                "lambda_weight": args.lambda_weight,
            }
            selected_groups.append(selected_dataset_indices)

    return selected_groups, records


def main() -> None:
    args = parse_args()
    selected_groups, annotated_records = select_from_records(args)

    output: Any
    if args.flat:
        output = [idx for group in selected_groups for idx in group]
    else:
        output = selected_groups

    write_json(output, Path(args.output), args.indent)
    print(f"Saved selected indices to {args.output}")
    print(f"Selected {sum(len(group) for group in selected_groups)} total steps")

    if args.annotated_output:
        write_json(annotated_records, Path(args.annotated_output), args.indent)
        print(f"Saved annotated records to {args.annotated_output}")


if __name__ == "__main__":
    main()
