"""Extract text from review PDFs."""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'pymupdf'], capture_output=True)

import fitz

for path in [r'C:\Users\ASUS\Downloads\review_report.pdf', 
             r'C:\Users\ASUS\Downloads\annotated_manuscript.pdf']:
    print(f"\n{'='*80}")
    print(f"FILE: {path}")
    print('='*80)
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        print(f"\n--- Page {i+1} ---")
        print(page.get_text())
    doc.close()
