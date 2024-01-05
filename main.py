import urllib.request
import xml.etree.ElementTree as ET
from utils import *

def fetch_papers():
    """Fetches papers from the arXiv API and returns them as a list of strings."""

    base_url = 'http://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    request = urllib.request.Request(base_url, headers=headers)
    response = urllib.request.urlopen(request)
    data = response.read().decode('utf-8')
    root = ET.fromstring(data)
    papers_list = []

    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    for entry in root.findall('.//atom:entry', namespaces=ns):
        title = entry.find('./atom:title', namespaces=ns).text
        summary = entry.find('./atom:summary', namespaces=ns).text

        paper_info = f"Title: {title}\nSummary: {summary}\n"
        papers_list.append(paper_info)

    return papers_list


def main():
    # Fetch papers
    papers = fetch_papers()
    for i, paper in enumerate(papers, start=1):
        print(f"Paper {i}:\n{paper}")

    while True:
        # Prompt user to ask a question
        question = input("Ask a question about Llama-2 (Type 'exit' to quit): ")

        # Check if the user wants to exit the loop
        if question.lower() == 'exit':
            print("Exiting the question and answer app.")
            break

        # Return answer to user's question
        answer = answer_question(question, papers)
        print("Answer:", answer)
        print("\n" + "=" * 50 + "\n")  # Separate previous answer from new question prompt

if __name__ == "__main__":
    main()
