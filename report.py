from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def generate_pdf_report(metadata_info, is_real_image, report_index):
    report_folder = "Report"
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    report_file = f"Report/Report_{report_index}.pdf"

    c = canvas.Canvas(report_file, pagesize=letter)
    c.drawString(100, 750, "Metadata Collected:")
    metadata_lines = metadata_info.split('\n')
    metadata_y = 730
    for line in metadata_lines:
        c.drawString(100, metadata_y, line)
        metadata_y -= 10  # Adjust vertical position for each line
    fake_flag = "Fake Image" if not is_real_image else "Real Image"
    c.drawString(100, metadata_y - 20, f"Analysis Result: {fake_flag}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(100, metadata_y - 40, f"Timestamp: {timestamp}")
    c.save()
