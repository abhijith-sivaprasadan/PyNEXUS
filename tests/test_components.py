import pytest

from components.electrolyzer import PEMElectrolyzer
from components.pipeline import HydrogenPipeline

CONFIG = "configs/tiny_test.yaml"


def test_electrolyser_respects_minimum_load_and_capacity() -> None:
    electrolyser = PEMElectrolyzer(CONFIG)

    assert electrolyser.compute_h2_output(9.0) == 0.0
    assert electrolyser.compute_h2_output(200.0) == pytest.approx(
        electrolyser.compute_h2_output(100.0)
    )


def test_pipeline_constrains_infeasible_flow() -> None:
    pipeline = HydrogenPipeline(CONFIG)
    requested = pipeline.max_feasible_flow_kg_s * 1.2

    assert not pipeline.is_feasible(requested)
    assert pipeline.constrained_flow(requested) == pytest.approx(pipeline.max_feasible_flow_kg_s)
