import json

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
