import nltk
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2Model
from nltk.corpus import stopwords

# Download nltk stopwords
nltk.download('stopwords')

""" All functions for:
    - preprocessing text
    - generating embeddings
    - calculating similarity between question and sources
    - generating an answer to the question
"""

def preprocess(text):
    """preprocess text - removes stopwords, changes to lowercase"""
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

def generate_gpt2_embeddings(tokens):
    """Generate GPT-2 embeddings - open source option was chosen for simplicity"""
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)

    input_ids = tokenizer.encode(" ".join(tokens), return_tensors="pt")
    with torch.no_grad():
        embeddings = model(input_ids)[0]

    # Extract the embeddings for each token
    token_embeddings = embeddings[:, 1:-1, :]

    # Average the token embeddings to get a single vector for the entire text
    avg_embedding = torch.mean(token_embeddings, dim=1).squeeze().numpy()

    return avg_embedding

def calculate_similarity(embedding_question, embedding_document):
    """Calculate similarity between question and papers"""
    similarity_score = cosine_similarity([embedding_question], [embedding_document])[0][0]
    return similarity_score

def answer_question(question, papers):
    """
        Parameters:
        - question (str): Input question to be answered.
        - papers (list of str): List of papers from the arXiv API.

        Returns:
        - str: The generated answer to the input question.

        Description:
        1. Preprocessing the input question.
        2. Generates GPT-2 embeddings for the preprocessed question.
        3. For each paper in the provided list, calculates the similarity score between the question
           and the paper's summary using GPT-2 embeddings.
        4. Ranks papers based on their similarity scores in descending order.
        5. Selects top-ranked paper, and relevant text is extracted based on the question.
        6. Relevant text is used to generate a concise answer to the question.
    """
    # Preprocess the question
    tokens_question = preprocess(question)

    # Generate GPT-2 embeddings for the question
    embeddings_question = generate_gpt2_embeddings(tokens_question)

    # Calculate similarity scores for each document
    scores = []
    for paper_info in papers:
        # Extract title and summary from each paper
        title, summary = extract_title_and_summary(paper_info)

        # Preprocess the text
        tokens_document = preprocess(summary)

        # Generate GPT-2 embeddings
        embeddings_document = generate_gpt2_embeddings(tokens_document)

        # Calculate similarity score
        similarity_score = calculate_similarity(embeddings_question, embeddings_document)

        # Store the score and document information
        scores.append({'score': similarity_score, 'paper_info': paper_info})

    # Check if there are valid papers to analyze
    if not scores:
        return "No relevant papers found."

    # Rank documents based on similarity scores
    ranked_documents = sorted(scores, key=lambda x: x['score'], reverse=True)

    # Add function to extract relevant text from the top-ranked document
    top_paper_info = ranked_documents[0]['paper_info']
    relevant_text = extract_relevant_text(question, top_paper_info)

    # Add function to generate an answer based on the relevant text
    answer = generate_answer(question, relevant_text)

    return answer


def extract_title_and_summary(paper_info):
    """Extract title and summary from paper_info string"""

    # Split the paper_info string into title and summary
    title_start = paper_info.find("Title:") + len("Title:")
    summary_start = paper_info.find("Summary:") + len("Summary:")

    title = paper_info[title_start:summary_start].strip()
    summary = paper_info[summary_start:].strip()

    return title, summary

""" With more time to work on this I would refine the output to provide more relevent result sections of the papers
    and cite their sources. The following would be examples to build out further. Currently the output is still
    the title and summary of the paper ranked most similar to the question.
"""
def extract_relevant_text(question, document_text):
    """Extract relevant text from a document"""
    return document_text


def generate_answer(question, relevant_text):
    """Generate an answer based on relevant text"""
    return relevant_text