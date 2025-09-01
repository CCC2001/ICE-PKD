"""
Command-line entry point for generating uplift reports.

Usage
-----
python -m run_report.py input.csv ./output_dir
"""
import sys
from uplift_report import ReportGenerator

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m run_report.py <input_csv> <save_dir>")
        sys.exit(1)

    csv_file, save_dir = sys.argv[1], sys.argv[2]
    ReportGenerator().generate_report(csv_file, save_dir)