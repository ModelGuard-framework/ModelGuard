
from core.config import dataset_path
from core.validator import load_dataset


def test_dataset_file_exists():
    assert dataset_path().exists()


def test_dataset_can_be_loaded():
    df = load_dataset(dataset_path())
    assert df is not None


def test_dataset_is_not_empty():
    df = load_dataset(dataset_path())
    assert not df.empty


def test_dataset_has_columns():
    df = load_dataset(dataset_path())
    assert len(df.columns) > 0
