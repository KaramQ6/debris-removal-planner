import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import sys

def search_arxiv(query_str, max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": query_str,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = base_url + urllib.parse.urlencode(params)
    print(f"Requesting URL: {url}")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
    except Exception as e:
        print(f"Error fetching data from arXiv: {e}")
        return []

    root = ET.fromstring(xml_data)
    results = []
    
    # Namespace dictionary for Atom XML
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    
    for entry in root.findall('atom:entry', ns):
        paper = {}
        
        # ID
        id_node = entry.find('atom:id', ns)
        if id_node is not None:
            paper["id"] = id_node.text.split('/abs/')[-1]
            paper["url"] = id_node.text
            
        # Title
        title_node = entry.find('atom:title', ns)
        if title_node is not None:
            paper["title"] = title_node.text.replace('\n', ' ').strip()
            
        # Summary (Abstract)
        summary_node = entry.find('atom:summary', ns)
        if summary_node is not None:
            paper["summary"] = summary_node.text.replace('\n', ' ').strip()
            
        # Published
        published_node = entry.find('atom:published', ns)
        if published_node is not None:
            paper["published"] = published_node.text
            
        # Authors
        authors = []
        for author_node in entry.findall('atom:author', ns):
            name_node = author_node.find('atom:name', ns)
            if name_node is not None:
                authors.append(name_node.text)
        paper["authors"] = authors
        
        # Link (PDF)
        pdf_url = ""
        for link_node in entry.findall('atom:link', ns):
            if link_node.get('title') == 'pdf':
                pdf_url = link_node.get('href')
                break
        if not pdf_url:
            # Fallback to look for PDF in href
            for link_node in entry.findall('atom:link', ns):
                href = link_node.get('href')
                if href and 'pdf' in href:
                    pdf_url = href
                    break
        paper["pdf_url"] = pdf_url
        
        # DOI
        doi_node = entry.find('arxiv:doi', ns)
        if doi_node is not None:
            paper["doi"] = doi_node.text
        else:
            paper["doi"] = None
            
        results.append(paper)
        
    return results

if __name__ == "__main__":
    # Query for Low-Thrust / Electric Propulsion Active Debris Removal Trajectory Optimization
    query = '(all:"low-thrust" OR all:"electric propulsion") AND all:"debris"'
    papers = search_arxiv(query, max_results=8)
    
    output_path = r"c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner\docs\arxiv_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully retrieved {len(papers)} papers and saved to {output_path}")
    for idx, p in enumerate(papers, 1):
        print(f"\n[{idx}] {p.get('title')}")
        print(f"    Authors: {', '.join(p.get('authors', []))}")
        print(f"    ArXiv ID: {p.get('id')} | PDF: {p.get('pdf_url')} | DOI: {p.get('doi')}")
