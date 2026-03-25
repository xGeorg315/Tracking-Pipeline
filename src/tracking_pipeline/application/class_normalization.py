from __future__ import annotations

from dataclasses import replace

from tracking_pipeline.config.models import ClassNormalizationConfig
from tracking_pipeline.domain.models import ObjectLabelData


class ClassNormalizer:
    def __init__(self, enabled: bool, aliases: dict[str, str]):
        self.enabled = bool(enabled)
        self.aliases = {
            self._normalize_key(raw_name): str(canonical_name).strip()
            for raw_name, canonical_name in aliases.items()
            if str(raw_name).strip() and str(canonical_name).strip()
        }

    @classmethod
    def from_config(cls, config: ClassNormalizationConfig) -> ClassNormalizer:
        return cls(bool(config.enabled), dict(config.aliases))

    def normalize(self, class_name: str | None) -> str:
        raw_name = str(class_name or "").strip()
        if not raw_name:
            return ""
        if not self.enabled:
            return raw_name
        return self.aliases.get(self._normalize_key(raw_name), raw_name)

    def normalize_object_label(self, label: ObjectLabelData) -> ObjectLabelData:
        normalized = self.normalize(label.obj_class)
        if normalized == str(label.obj_class or ""):
            return label
        return replace(label, obj_class=normalized)

    @staticmethod
    def _normalize_key(value: str) -> str:
        return str(value).strip().casefold()
