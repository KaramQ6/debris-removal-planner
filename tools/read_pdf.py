import sys
import os

def extract_pdf_text(pdf_path, output_txt_path):
    try:
        import pypdf
    except ImportError:
        print("pypdf is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        import pypdf

    print(f"Reading PDF from: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    text_content = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        text_content.append(f"--- PAGE {i+1} ---")
        text_content.append(text)
    
    full_text = "\n".join(text_content)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Extracted text successfully written to: {output_txt_path}")

if __name__ == "__main__":
    pdf_path = r"C:\Users\ASUS\Downloads\Documents\AEECT-2026-CFP.pdf"
    output_txt = r"c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner\docs\aeect_2026_cfp_text.txt"
    extract_pdf_text(pdf_path, output_txt)
