import re
import sys

def verify_paper(paper_path):
    print(f"Verifying paper: {paper_path}")
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    
    # 1. Verify Abstract Length
    abstract_match = re.search(r"### Abstract\s*\n(.*?)\n\n", content, re.DOTALL)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        words = abstract_text.split()
        word_count = len(words)
        print(f"Abstract Word Count: {word_count}")
        if word_count < 75 or word_count > 150:
            errors.append(f"Abstract word count is {word_count}, which is outside the [75, 150] word limit.")
    else:
        errors.append("Abstract section not found.")

    # 2. Verify Keywords
    keywords_match = re.search(r"\*\*Index Terms \(Keywords\):\*\*\s*(.*?)\n", content)
    if keywords_match:
        keywords_text = keywords_match.group(1).strip()
        keywords = [k.strip() for k in keywords_text.split(";") if k.strip()]
        kw_count = len(keywords)
        print(f"Keywords Count: {kw_count} ({keywords_text})")
        if kw_count < 4 or kw_count > 6:
            errors.append(f"Keyword count is {kw_count}, which is outside the [4, 6] limit.")
        if not keywords_text.endswith("."):
            print("Note: Index terms should end with a period in final IEEE formatting.")
    else:
        errors.append("Keywords section not found.")

    # 3. Verify Citations
    citations = set(re.findall(r"\[([0-9]+)\]", content))
    # Filter out references section from citation matching
    body_content = content.split("## References")[0]
    body_citations = set(re.findall(r"\[([0-9]+)\]", body_content))
    
    references_section = content.split("## References")[-1]
    reference_items = set(re.findall(r"\*?\s*\[([0-9]+)\]", references_section))
    
    print(f"Citations used in text: {sorted(list(body_citations))}")
    print(f"References listed in Bibliography: {sorted(list(reference_items))}")
    
    missing_refs = body_citations - reference_items
    if missing_refs:
        errors.append(f"Citations used in text but missing from references: {missing_refs}")
        
    unused_refs = reference_items - body_citations
    if unused_refs:
        print(f"Warning: References listed but not cited in text: {unused_refs}")

    if errors:
        print("\nVerification FAILED with the following errors:")
        for err in errors:
            print(f" - [ERROR] {err}")
        return False
    else:
        print("\nVerification PASSED! The paper matches all tested AEECT 2026 Author Kit requirements.")
        return True

if __name__ == "__main__":
    paper_path = r"c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner\docs\AEECT_2026_Paper_Draft.md"
    success = verify_paper(paper_path)
    sys.exit(0 if success else 1)
