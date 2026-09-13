
from core.config import model_path
from core.validator import check_model_integrity


def test_model_file_exists():
    result = check_model_integrity(model_path())
    assert result["status"] != "FAIL", result["details"]


def test_model_can_be_loaded():
    result = check_model_integrity(model_path())
    assert result["status"] == "PASS", result["details"]


def test_model_has_prediction_method():
    result = check_model_integrity(model_path())
    assert result["status"] == "PASS", result["details"]
