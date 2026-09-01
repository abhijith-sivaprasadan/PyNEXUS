import hashlib
import json

from data.provenance import build_provenance_record, read_provenance_record, write_provenance_record


def test_provenance_record_round_trips(tmp_path) -> None:
    raw_file = tmp_path / "raw.nc"
    raw_file.write_bytes(b"fake netcdf bytes for a checksum test")
    expected_sha256 = hashlib.sha256(raw_file.read_bytes()).hexdigest()

    record = build_provenance_record(
        source="ERA5 reanalysis-era5-single-levels (hourly)",
        variables=["10m_u_component_of_wind", "10m_v_component_of_wind"],
        requested_latitude=52.5,
        requested_longitude=3.5,
        actual_latitude=52.5,
        actual_longitude=3.5,
        start_date="2023-01-01",
        end_date="2023-01-02",
        timezone_name="Europe/Amsterdam",
        cds_request={"variable": ["10m_u_component_of_wind"]},
        raw_file=raw_file,
        expected_row_count=48,
        actual_row_count=48,
        calendar_check={"no_gaps": True, "no_duplicates": True},
    )

    assert record.file_sha256 == expected_sha256
    assert record.row_count == {"expected": 48, "actual": 48}

    out_path = tmp_path / "provenance.json"
    write_provenance_record(record, out_path)

    loaded = read_provenance_record(out_path)
    assert loaded["source"] == "ERA5 reanalysis-era5-single-levels (hourly)"
    assert loaded["file_sha256"] == expected_sha256
    assert loaded["spatial_point"]["grid_cell_center_latitude"] == 52.5
    assert loaded["schema_version"] == 1

    # Written file must be valid, deterministic JSON (sorted keys) that a
    # human reviewing a diff can actually read.
    raw_text = out_path.read_text(encoding="utf-8")
    assert json.loads(raw_text) == loaded
