from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tracking_pipeline.shared.numpy_utils import to_serializable


class ManifestWriter:
    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(to_serializable(row), sort_keys=True))
                handle.write("\n")
