import fitz, sys
doc = fitz.open(r"C:\Users\ASUS\Downloads\review_report (1).pdf")
for p in doc:
    print(p.get_text())
