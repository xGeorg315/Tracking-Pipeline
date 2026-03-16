from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_cost_matrix(
    predicted_centers: np.ndarray,
    predicted_extents: np.ndarray,
    detection_centers: np.ndarray,
    detection_extents: np.ndarray,
    size_weight: float,
) -> np.ndarray:
    if len(predicted_centers) == 0 or len(detection_centers) == 0:
        return np.zeros((len(predicted_centers), len(detection_centers)), dtype=np.float32)
    dist = np.linalg.norm(predicted_centers[:, None, :] - detection_centers[None, :, :], axis=2)
    if float(size_weight) <= 0:
        return dist.astype(np.float32)
    safe_den = np.maximum(np.maximum(predicted_extents[:, None, :], detection_extents[None, :, :]), 1e-3)
    size_delta = np.mean(np.abs(predicted_extents[:, None, :] - detection_extents[None, :, :]) / safe_den, axis=2)
    return (dist + float(size_weight) * size_delta).astype(np.float32)


def assign_cost_matrix(cost_matrix: np.ndarray, valid_mask: np.ndarray, method: str) -> tuple[dict[int, int], list[int], list[int]]:
    rows, cols = cost_matrix.shape
    if rows == 0 and cols == 0:
        return {}, [], []
    if rows == 0:
        return {}, [], list(range(cols))
    if cols == 0:
        return {}, list(range(rows)), []

    masked = np.where(valid_mask, cost_matrix, np.inf)
    assignments: dict[int, int] = {}

    if method == "hungarian":
        feasible_rows = np.isfinite(masked).any(axis=1)
        feasible_cols = np.isfinite(masked).any(axis=0)
        reduced = masked[np.ix_(feasible_rows, feasible_cols)]
        if reduced.size == 0 or not np.isfinite(reduced).any():
            return {}, list(range(rows)), list(range(cols))
        row_map = np.flatnonzero(feasible_rows)
        col_map = np.flatnonzero(feasible_cols)
        finite_costs = reduced[np.isfinite(reduced)]
        sentinel = float(np.max(finite_costs) + max(1.0, np.max(finite_costs) * 10.0))
        safe_reduced = np.where(np.isfinite(reduced), reduced, sentinel)
        row_ind, col_ind = linear_sum_assignment(safe_reduced)
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            mapped_row = int(row_map[row])
            mapped_col = int(col_map[col])
            if not np.isfinite(masked[mapped_row, mapped_col]):
                continue
            assignments[mapped_row] = mapped_col
            used_rows.add(mapped_row)
            used_cols.add(mapped_col)
        unmatched_rows = [row for row in range(rows) if row not in used_rows]
        unmatched_cols = [col for col in range(cols) if col not in used_cols]
        return assignments, unmatched_rows, unmatched_cols

    valid_pairs = np.argwhere(np.isfinite(masked))
    order = np.argsort(masked[valid_pairs[:, 0], valid_pairs[:, 1]]) if len(valid_pairs) else []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for order_index in order:
        row = int(valid_pairs[order_index, 0])
        col = int(valid_pairs[order_index, 1])
        if row in used_rows or col in used_cols:
            continue
        assignments[row] = col
        used_rows.add(row)
        used_cols.add(col)
    unmatched_rows = [row for row in range(rows) if row not in used_rows]
    unmatched_cols = [col for col in range(cols) if col not in used_cols]
    return assignments, unmatched_rows, unmatched_cols
