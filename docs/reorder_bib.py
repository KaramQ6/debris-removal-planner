import re
import sys

file_path = r"c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner\docs\AEECT_2026_Paper.tex"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split the document into main text and bibliography
    main_match = re.search(r'(.*?)\\begin\{thebibliography\}\{[^}]+\}(.*)\\end\{thebibliography\}(.*)', text, re.DOTALL)
    if not main_match:
        print("Could not find thebibliography environment.")
        sys.exit(1)

    main_text = main_match.group(1)
    bib_text = main_match.group(2)
    post_bib = main_match.group(3)

    # Extract all \cite{...} commands and find order
    cites = re.findall(r'\\cite\{([^}]+)\}', main_text)
    order = []
    for cite in cites:
        keys = [k.strip() for k in cite.split(',')]
        for key in keys:
            if key not in order:
                order.append(key)

    # Parse bibliography items
    bib_items = {}
    items = bib_text.split('\\bibitem{')
    for item in items[1:]:
        key, content = item.split('}', 1)
        bib_items[key.strip()] = content

    # Reconstruct bibliography
    # Count the number of cited items
    num_items = len(order)
    new_bib = f"\\begin{{thebibliography}}{{{num_items}}}\n"
    
    missing_items = []
    for key in order:
        if key in bib_items:
            new_bib += f"\\bibitem{{{key}}}" + bib_items[key]
        else:
            missing_items.append(key)
            new_bib += f"\\bibitem{{{key}}} MISSING CITATION TEXT\n"

    new_bib += "\\end{thebibliography}"

    new_text = main_text + new_bib + post_bib

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"Success! Found {num_items} cited items. Reordered bibliography.")
    if missing_items:
        print("Warning! Some cited keys were not found in the bibliography:", missing_items)

except Exception as e:
    print(f"Error: {e}")
