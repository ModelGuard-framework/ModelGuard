import os

from reports.report_generator import generate_report


def test_report_generation():
    results = {
        "Model Integrity": "PASS",
        "Dataset Input": "PASS",
        "Data Validation": "PASS",
        "Preprocessing": "PASS",
        "Model Inference": "PASS",
        "Result Validation": "PASS",
        "Robustness Testing": "PASS",
        "Data Integrity": "PASS"
    }

    output_path = "reports/test_report.docx"

    generate_report(results, output_path)

    assert os.path.exists(output_path)