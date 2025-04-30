from pathlib import Path
import pytest

from mrcad import Line, Arc, Circle, Design
import json
import jsonlines

TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


@pytest.mark.datafiles(TEST_DATA_DIR)
def test_model_validation(datafiles):
    with jsonlines.open(datafiles / "test_design_pool.jsonl", mode="r") as reader:
        for record in reader:
            Design.model_validate(record["design"])


@pytest.mark.datafiles(TEST_DATA_DIR)
def test_model_dump(datafiles):
    with jsonlines.open(datafiles / "test_design_pool.jsonl", mode="r") as reader:
        for record in reader:
            assert json.dumps(
                Design.model_validate(record["design"]).model_dump(mode="json")
            ) == json.dumps(record["design"])


@pytest.mark.datafiles(TEST_DATA_DIR)
def test_rounding(datafiles):
    with jsonlines.open(
        datafiles / "test_design_pool.jsonl", mode="r"
    ) as reader_original:
        with jsonlines.open(
            datafiles / "test_design_pool_rounded.jsonl", mode="r"
        ) as reader_rounded:
            for record_original, record_rounded in zip(reader_original, reader_rounded):
                assert json.dumps(
                    Design.model_validate(record_original["design"])
                    .round(record_rounded["rounding_precision"])
                    .model_dump(mode="json")
                ) == json.dumps(record_rounded["design"])
