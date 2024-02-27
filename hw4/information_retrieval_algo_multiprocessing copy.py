import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from string import punctuation
from stop_list import closed_class_stop_words
from time import time


# Function to tokenize and clean text
def tokenize_clean(text, vocabulary, output_tokens_list):
    lemmatizer = WordNetLemmatizer()
    tokens = []

    if not text:
        return tokens

    dirty_tokens = word_tokenize(text)
    for dirty_token in dirty_tokens:
        if "-" in dirty_token:
            dirty_tokens.extend(dirty_token.split("-"))
        if (
            dirty_token not in closed_class_stop_words
            and dirty_token not in punctuation
            and dirty_token.isalpha()
            and len(dirty_token) > 2
        ):
            cleaned_token = dirty_token.lower().strip(punctuation)
            lemmatized_token = lemmatizer.lemmatize(cleaned_token)
            tokens.append(lemmatized_token)
            vocabulary.add(lemmatized_token)

    output_tokens_list.append(tokens)


# Define the worker function at the top level of the module
def worker_process(chunk_data):
    chunk, vocab2index, tokens_list = chunk_data
    partial_matrix = np.zeros((len(chunk), len(vocab2index)))
    N = len(tokens_list)

    for text_row, tokens in enumerate(chunk):
        token_length = len(tokens)
        for token in tokens:
            if token in vocab2index:
                token_col = vocab2index[token]
                tf = tokens.count(token) / token_length
                df = sum([1 for query in tokens_list if token in query])
                idf = np.log(N + 1 / df + 1)
                partial_matrix[text_row, token_col] = (tf / token_length) * idf
    return partial_matrix


def populate_matrix_parallel(tokens_list, vocab2index, num_workers=12):
    tokens_list_length = len(tokens_list)
    chunk_size = tokens_list_length // num_workers + (
        tokens_list_length % num_workers > 0
    )

    chunks = [
        tokens_list[i : i + chunk_size]
        for i in range(0, tokens_list_length, chunk_size)
    ]
    chunk_data = [(chunk, vocab2index, tokens_list) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_results = executor.map(worker_process, chunk_data)

    matrix = np.vstack(list(future_results))

    return matrix


def main():
    start = time()

    # Load and preprocess queries
    with open("./Cranfield_collection_HW/cran.qry") as file:
        queries = file.read().split(".I")

    query_tokens_list = []
    vocab = set()
    for query in queries:
        query = query.split(".W")[-1].strip()
        tokenize_clean(query, vocab, query_tokens_list)

    # Load and preprocess documents
    with open("./Cranfield_collection_HW/cran.all.1400") as file:
        documents = file.read().split(".I")

    document_tokens_list = []
    for document in documents:
        document = document.split(".W")[-1].strip()
        tokenize_clean(document, vocab, document_tokens_list)

    # Create and populate matrices
    sorted_vocab = sorted(vocab)
    vocab2index = {word: i for i, word in enumerate(sorted_vocab)}

    query_matrix = populate_matrix_parallel(query_tokens_list, vocab2index)
    document_matrix = populate_matrix_parallel(document_tokens_list, vocab2index)

    # Calculate similarity
    Q_norm = query_matrix / np.linalg.norm(query_matrix, axis=1, keepdims=True)
    D_norm = document_matrix / np.linalg.norm(document_matrix, axis=1, keepdims=True)
    similarity_matrix = Q_norm @ D_norm.T

    # Output results
    with open("output.txt", encoding="utf8", mode="w") as file:
        for query_idx in range(similarity_matrix.shape[0]):
            # Only the top 100 results are needed
            for document_idx in similarity_matrix[query_idx, :].argsort()[:-101:-1]:
                score = similarity_matrix[query_idx, document_idx]
                if score > 0:
                    file.write(f"{query_idx+1} {document_idx+1} {score}\n")

    print(f"Elapsed time: {time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
