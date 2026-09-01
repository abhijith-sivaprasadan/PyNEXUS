import json

import optimization.cli as cli
from optimization.cli import main


def test_tiny_run_is_verified_and_tampering_fails(tmp_path):
    out = tmp_path / "run"
    assert (
        main(["solve", "--config", "configs/tiny_test.yaml", "--synthetic", "--output", str(out)])
        == 0
    )
    assert main(["verify", "--run", str(out)]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["snapshots"] == 4
    assert manifest["verified"]
    assert manifest["solver"]["threads"] == 1
    assert manifest["result"]["termination_condition"] == "optimal"
    assert manifest["model"]["binary_variables"] == 4
    with (out / "dispatch.csv").open("a") as stream:
        stream.write("tampered\n")
    assert main(["verify", "--run", str(out)]) == 2


def test_solve_with_all_phase_a_b_flags_end_to_end(tmp_path):
    """The CLI --enable-storage/--enable-grid/--enable-heat flags, backed by
    an auto-generated synthetic heat_demand_mw column for --synthetic, must
    round-trip through solve -> manifest -> verify exactly like the baseline
    path — Phase A/B are otherwise only reachable from direct Python calls."""
    out = tmp_path / "run"
    exit_code = main(
        [
            "solve",
            "--config",
            "configs/tiny_test.yaml",
            "--synthetic",
            "--output",
            str(out),
            "--demand-mode",
            "hourly",
            "--enable-storage",
            "--enable-grid",
            "--enable-heat",
        ]
    )
    assert exit_code == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["enable_storage"] and manifest["enable_grid"] and manifest["enable_heat"]
    assert manifest["verified"]
    inputs = (out / "inputs.csv").read_text()
    assert "heat_demand_mw" in inputs.splitlines()[0]

    assert main(["verify", "--run", str(out)]) == 0


def test_fetch_era5_wires_arguments_through(monkeypatch, tmp_path, capsys):
    calls = {}

    class FakeRecord:
        file_sha256 = "deadbeef"
        spatial_point = {"latitude": 52.5, "longitude": 3.5}

    class FakeFrame(list):
        pass  # len() of an empty-but-sized list stands in for the wind dataframe

    def fake_fetch_and_load(cfg, start_date, end_date, cache_dir, provenance_dir):
        calls["args"] = (start_date, end_date, str(cache_dir), str(provenance_dir))
        return FakeFrame(range(48)), FakeRecord()

    monkeypatch.setattr(cli, "fetch_and_load", fake_fetch_and_load)

    exit_code = main(
        [
            "fetch-era5",
            "--config",
            "config.yaml",
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-01-02",
            "--cache-dir",
            str(tmp_path / "raw"),
            "--provenance-dir",
            str(tmp_path / "prov"),
        ]
    )

    assert exit_code == 0
    assert calls["args"] == (
        "2023-01-01",
        "2023-01-02",
        str(tmp_path / "raw"),
        str(tmp_path / "prov"),
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["file_sha256"] == "deadbeef"
    assert printed["rows"] == 48


def test_existing_output_directory_is_not_overwritten(tmp_path):
    assert (
        main(
            [
                "solve",
                "--config",
                "configs/tiny_test.yaml",
                "--synthetic",
                "--output",
                str(tmp_path),
            ]
        )
        == 2
    )
