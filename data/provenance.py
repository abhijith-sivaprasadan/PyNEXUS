"""Provenance records for externally fetched datasets.

Every dataset this repository pulls from an external source (ERA5, ENTSO-E,
...) must be accompanied by a committed record answering: what was requested,
from where, when, at what exact location, and did it arrive complete. The
schema is fixed here so every data source produces the same shape of record.

Records are small JSON files and are committed to `data/provenance_records/`.
The raw downloads they describe are not committed (see .gitignore) — the
record plus its checksum is what lets someone verify a re-fetch matches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large NetCDF files are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ProvenanceRecord:
    """One committed record for one fetched dataset.

    Field names and meaning are fixed by REVAMP_PLAN.md Phase C2 — do not
    rename without updating every writer and reader.
    """

    source: str
    variables: list[str]
    spatial_point: dict[str, float]
    temporal_range: dict[str, str]
    retrieval_timestamp: str
    cds_request: dict[str, Any]
    file_sha256: str
    row_count: dict[str, int]
    calendar_check: dict[str, Any]
    schema_version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_provenance_record(
    *,
    source: str,
    variables: list[str],
    requested_latitude: float,
    requested_longitude: float,
    actual_latitude: float,
    actual_longitude: float,
    start_date: str,
    end_date: str,
    timezone_name: str,
    cds_request: dict[str, Any],
    raw_file: Path,
    expected_row_count: int,
    actual_row_count: int,
    calendar_check: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> ProvenanceRecord:
    """Assemble a ProvenanceRecord from the pieces a fetch/load step produces."""
    return ProvenanceRecord(
        source=source,
        variables=list(variables),
        spatial_point={
            "requested_latitude": requested_latitude,
            "requested_longitude": requested_longitude,
            "grid_cell_center_latitude": actual_latitude,
            "grid_cell_center_longitude": actual_longitude,
        },
        temporal_range={
            "start": start_date,
            "end": end_date,
            "timezone": timezone_name,
        },
        retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
        cds_request=cds_request,
        file_sha256=sha256_file(raw_file),
        row_count={"expected": expected_row_count, "actual": actual_row_count},
        calendar_check=calendar_check,
        extra=extra or {},
    )


def write_provenance_record(record: ProvenanceRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_provenance_record(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
