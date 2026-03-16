from __future__ import annotations


def aggregate_file_stem(track_id: int) -> str:
    return f"track_{track_id:04d}"


def object_file_stem(object_id: int) -> str:
    return f"object_{object_id:04d}"
