# Overview
This program is designed to facilitate team knowledge building around Llama-2 by providing a question-answering system based on research papers from the arXiv API. The solution uses the open source GPT-2 model for generating embeddings and answering questions related to Llama-2.

## Features
- Fetches papers from the arXiv API using a specified search query.
- Generates GPT-2 embeddings for questions and article summaries.
- Ranks documents based on similarity scores between question and article embeddings.
- Extracts relevant text from the top-ranked article.
- Summarizes answers to user queries about Llama-2.
- Libraries listed in `requirements.txt`

## Use
1. Clone the repository:
   ```bash
   git clone https://github.com/jordan-elias/llama-2_uses.git

2. Navigate to the project directory:
   ```bash
   cd llama-2_uses
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the main script:
   ```bash
   python main.py

Enter your Llama-2 related question when prompted.
Receive answers based on the content of the library of papers.
Type 'exit' to quit the question and answer loop.

Example:

Ask a question about Llama-2 (Type 'exit' to quit): For which tasks has Llama-2 already been used successfully?

Answer: Relevant information about successful tasks performed by Llama-2.




