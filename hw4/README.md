# README for Enhanced Information Retrieval System

## Implementation

- **Preprocessing**: Queries and documents are tokenized, cleaned, and lemmatized to reduce to their base forms.

- **Vectorization**: Builds a vocabulary of unique words and converts text data into TF-IDF vectors.

- **Parallel Processing**: Employs `ProcessPoolExecutor` for concurrent vectorization and similarity computation.

- **Similarity Calculation**: Uses the normalized dot product of query and document vectors as a work around for the intensive nested for loop.

- **Output**: Generates a list of the top 20 most relevant documents for each query, saved to a file.
