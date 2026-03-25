from __future__ import annotations

import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from tracking_pipeline.config.models import ClassificationConfig
from tracking_pipeline.config.validation import ConfigError
from tracking_pipeline.domain.models import ClassificationPrediction


class _AttrDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path.resolve())
    original = list(sys.path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        sys.path[:] = original


class PointNextObjectClassifier:
    backend = "pointnext"

    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.pointnext_root = Path(config.pointnext_root).resolve()
        self.checkpoint_path = Path(config.checkpoint_path).resolve()
        self.model_cfg_path = Path(config.model_cfg_path).resolve()
        self.class_names = sorted(str(class_name) for class_name in config.class_names)

        openpoints_dir = self.pointnext_root / "openpoints"
        if not openpoints_dir.exists() or not openpoints_dir.is_dir():
            raise ConfigError(f"PointNeXt openpoints directory does not exist: {openpoints_dir}")
        if not any(openpoints_dir.iterdir()):
            raise ConfigError(
                "PointNeXt openpoints directory is empty. Initialize the submodule before enabling classification: "
                f"{openpoints_dir}"
            )

        self.model_data = self._load_model_cfg()
        self.num_points = int(self.model_data.get("num_points", 1024))
        model_cfg = _to_attr_dict(self.model_data["model"])

        extra_global_channels = int(model_cfg.get("extra_global_channels", 0) or 0)
        if extra_global_channels != 0:
            raise ConfigError("PointNeXt classification currently supports only model.extra_global_channels == 0")
        in_channels = int(model_cfg.get("encoder_args", {}).get("in_channels", 3) or 0)
        if in_channels != 3:
            raise ConfigError("PointNeXt classification currently supports only model.encoder_args.in_channels == 3")

        expected_classes = int(model_cfg.get("cls_args", {}).get("num_classes", 0) or 0)
        if expected_classes <= 0:
            raise ConfigError(f"Could not infer PointNeXt num_classes from {self.model_cfg_path}")
        if len(self.class_names) != expected_classes:
            raise ConfigError(
                "classification.class_names length must match the PointNeXt model num_classes "
                f"({expected_classes})"
            )

        self.device = self._resolve_device(config.device)
        with _temporary_sys_path(self.pointnext_root):
            try:
                from openpoints.models import build_model_from_cfg
            except Exception as exc:
                raise ConfigError(
                    "Could not import PointNeXt/OpenPoints. Ensure the PointNeXt repository and its openpoints "
                    f"submodule are available under {self.pointnext_root}: {exc}"
                ) from exc
            try:
                self.model = build_model_from_cfg(model_cfg).to(self.device)
            except Exception as exc:
                raise ConfigError(f"Could not build the PointNeXt classification model: {exc}") from exc

        state_dict = self._load_state_dict(self.checkpoint_path)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ConfigError(f"Could not load the PointNeXt checkpoint {self.checkpoint_path}: {exc}") from exc
        self.model.eval()

    def classify_points(self, points: np.ndarray) -> ClassificationPrediction:
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("PointNextObjectClassifier expects points with shape [N, 3+]")
        arr = arr[:, :3]
        finite_mask = np.all(np.isfinite(arr), axis=1)
        arr = arr[finite_mask]
        if len(arr) == 0:
            raise ValueError("PointNextObjectClassifier received no finite xyz points")

        normalized = self._normalize(arr)
        sampled = self._resample(normalized)
        batch_points = torch.from_numpy(sampled[None, ...]).to(self.device)
        batch = {
            "pos": batch_points[:, :, :3].contiguous(),
            "x": batch_points[:, :, :3].transpose(1, 2).contiguous(),
        }
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)
            class_id = int(torch.argmax(probabilities, dim=1).item())
            score = float(probabilities[0, class_id].item())
        return ClassificationPrediction(
            class_id=class_id,
            class_name=self.class_names[class_id],
            score=score,
        )

    def _load_model_cfg(self) -> dict[str, Any]:
        with self.model_cfg_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        base_path = self.model_cfg_path.with_name("default.yaml")
        if self.model_cfg_path.name != "default.yaml" and base_path.exists():
            with base_path.open("r", encoding="utf-8") as handle:
                raw = _deep_merge(yaml.safe_load(handle) or {}, raw)
        if "model" not in raw:
            raise ConfigError(f"PointNeXt model config does not contain a model section: {self.model_cfg_path}")
        return raw

    def _resolve_device(self, device_name: str) -> torch.device:
        device_name = str(device_name).strip().lower()
        if device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device_name == "cuda":
            if not torch.cuda.is_available():
                raise ConfigError("classification.device is set to cuda, but CUDA is not available")
            return torch.device("cuda")
        if device_name == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise ConfigError(
                    "classification.device is set to mps, but MPS is not available. "
                    "Check that PyTorch MPS support is enabled and that your Torch version supports this macOS release."
                )
            return torch.device("mps")
        if device_name == "cpu":
            return torch.device("cpu")
        raise ConfigError(f"Unsupported classification device: {device_name}")

    def _load_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ConfigError(f"Unsupported PointNeXt checkpoint format: {checkpoint_path}")
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped_key = str(key)
            if stripped_key.startswith("module."):
                stripped_key = stripped_key[len("module.") :]
            cleaned[stripped_key] = value
        return cleaned

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        centroid = np.mean(points, axis=0, dtype=np.float32)
        return points - centroid

    def _resample(self, points: np.ndarray) -> np.ndarray:
        point_count = len(points)
        if point_count >= self.num_points:
            indices = self._stratified_fps_indices(points, self.num_points)
            return points[indices]
        base_indices = self._fps_indices(points, point_count)
        base_points = points[base_indices]
        if len(base_points) == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)
        repeats = int(np.ceil(float(self.num_points) / float(len(base_points))))
        tiled = np.tile(base_points, (repeats, 1))
        return tiled[: self.num_points]

    def _fps_indices(self, points: np.ndarray, target_count: int, candidate_idx: np.ndarray | None = None) -> np.ndarray:
        point_count = points.shape[0]
        if candidate_idx is None:
            candidate_idx = np.arange(point_count, dtype=np.int64)
        else:
            candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
        if target_count >= len(candidate_idx):
            return candidate_idx

        candidates = points[candidate_idx]
        centroid = np.mean(candidates, axis=0, dtype=np.float32)
        distances = np.linalg.norm(candidates - centroid, axis=1)
        current = int(np.argmax(distances))
        selected = np.empty((target_count,), dtype=np.int64)
        min_dist_sq = np.full((len(candidates),), np.inf, dtype=np.float64)
        for index in range(target_count):
            selected[index] = current
            diff = candidates - candidates[current]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)
            current = int(np.argmax(min_dist_sq))
        return candidate_idx[selected]

    def _stratified_fps_indices(self, points: np.ndarray, target_count: int, bins: int = 8) -> np.ndarray:
        point_count = points.shape[0]
        if target_count >= point_count:
            return np.arange(point_count, dtype=np.int64)

        axis = int(np.argmax(np.ptp(points, axis=0)))
        axis_values = points[:, axis]
        edges = np.linspace(float(np.min(axis_values)), float(np.max(axis_values)), int(bins) + 1)
        bin_indices: list[np.ndarray] = []
        for index in range(int(bins)):
            if index == int(bins) - 1:
                mask = (axis_values >= edges[index]) & (axis_values <= edges[index + 1])
            else:
                mask = (axis_values >= edges[index]) & (axis_values < edges[index + 1])
            current = np.where(mask)[0]
            if len(current) > 0:
                bin_indices.append(current.astype(np.int64, copy=False))

        if not bin_indices:
            return self._fps_indices(points, target_count)

        quota = max(1, target_count // len(bin_indices))
        selected_parts = [
            self._fps_indices(points, min(quota, len(indices)), candidate_idx=indices)
            for indices in bin_indices
            if len(indices) > 0
        ]
        selected = (
            np.concatenate(selected_parts, axis=0)
            if selected_parts
            else np.empty((0,), dtype=np.int64)
        )
        if len(selected) >= target_count:
            return selected.astype(np.int64, copy=False)[:target_count]

        remaining = np.setdiff1d(np.arange(point_count, dtype=np.int64), selected, assume_unique=False)
        extra = self._fps_indices(points, min(target_count - len(selected), len(remaining)), candidate_idx=remaining)
        combined = np.concatenate([selected, extra], axis=0)
        if len(combined) >= target_count:
            return combined[:target_count]
        repeats = int(np.ceil(float(target_count) / float(max(1, len(combined)))))
        tiled = np.tile(combined, repeats)
        return tiled[:target_count]
