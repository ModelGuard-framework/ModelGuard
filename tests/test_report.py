
from core.config import dataset_path, label_column, model_path, report_path
from core.validator import run_validation_pipeline
from reports.report_generator import generate_report


def test_report_generation():
    results = run_validation_pipeline(
        model_path(),
        dataset_path(),
        label_column(),
    )

    output_path = report_path()
    generate_report(
        results,
        output_path,
        model_path=model_path(),
        dataset_path=dataset_path(),
        label_column=label_column(),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
