from docx import Document
from datetime import datetime
import os


def generate_report(results, output_path="reports/ModelGuard_Report.docx"):
    document = Document()

    document.add_heading("MODEL GUARD — ML TEST REPORT", 0)

    document.add_paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    document.add_heading("Testing Results", level=1)

    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"

    header = table.rows[0].cells
    header[0].text = "Testing Stage"
    header[1].text = "Status"

    for stage, status in results.items():
        row = table.add_row().cells
        row[0].text = stage
        row[1].text = status

    if all(status == "PASS" for status in results.values()):
        overall_status = "PASS"
    else:
        overall_status = "FAIL"

    document.add_heading("Overall Result", level=1)
    document.add_paragraph(f"Overall Status: {overall_status}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    document.save(output_path)

    print(f"Report generated successfully: {output_path}")