import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from string import punctuation
from stop_list import closed_class_stop_words

# Function to tokenize and clean text
def tokenize_clean(text, vocabulary, output_tokens_list):
    lemmatizer = WordNetLemmatizer()
    tokens = []
    
    if not text:
        return tokens
    
    dirty_tokens = word_tokenize(text)
    for dirty_token in dirty_tokens:
        if '-' in dirty_token:
            dirty_tokens.extend(dirty_token.split('-'))
        if dirty_token not in closed_class_stop_words and dirty_token not in punctuation and dirty_token.isalpha() and len(dirty_token) > 2:
            cleaned_token = dirty_token.lower().strip(punctuation)
            lemmatized_token = lemmatizer.lemmatize(cleaned_token)
            tokens.append(lemmatized_token)
            vocabulary.add(lemmatized_token)
    
    output_tokens_list.append(tokens)

# Function to populate matrix with TF-IDF values
def populate_matrix(tokens_list, matrix, vocab2index):
    for text_row, tokens in enumerate(tqdm(tokens_list)):
        for token in tokens:
            token_col = vocab2index[token]
            tf = tokens.count(token) / len(tokens)
            df = sum([1 for query in tokens_list if token in query])
            idf = np.log(len(tokens_list) / df)
            matrix[text_row, token_col] = tf * idf

# Main function to orchestrate the processing
def main():
    # Load and preprocess queries
    with open('./Cranfield_collection_HW/cran.qry') as file:
        queries = file.read().split('.I')

    query_tokens_list = []
    vocab = set()
    for query in queries:
        query = query.split('.W')[-1].strip()
        tokenize_clean(query, vocab, query_tokens_list)

    # Load and preprocess documents
    with open('./Cranfield_collection_HW/cran.all.1400') as file:
        documents = file.read().split('.I')

    document_tokens_list = []
    for document in documents:
        document = document.split('.W')[-1].strip()
        tokenize_clean(document, vocab, document_tokens_list)

    # Create and populate matrices
    sorted_vocab = sorted(vocab)
    vocab2index = {word: i for i, word in enumerate(sorted_vocab)}
    query_matrix = np.zeros((len(query_tokens_list), len(vocab)))
    document_matrix = np.zeros((len(document_tokens_list), len(vocab)))
    
    populate_matrix(query_tokens_list, query_matrix, vocab2index)
    populate_matrix(document_tokens_list, document_matrix, vocab2index)

    # Calculate similarity
    Q_norm = query_matrix / np.linalg.norm(query_matrix, axis=1, keepdims=True)
    D_norm = document_matrix / np.linalg.norm(document_matrix, axis=1, keepdims=True)
    similarity_matrix = Q_norm @ D_norm.T

    # Output results
    with open('output.txt', encoding='utf8', mode='w') as file:
        for query_idx in range(similarity_matrix.shape[0]):
            for document_idx in similarity_matrix[query_idx,:].argsort()[::-1]:
                score = similarity_matrix[query_idx, document_idx]
                if score > 0:
                    file.write(f'{query_idx+1} {document_idx+1} {score}\n')

if __name__ == '__main__':
    main()
