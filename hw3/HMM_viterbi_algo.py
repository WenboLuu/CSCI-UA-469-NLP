import numpy as np
from collections import defaultdict


class HMM:
    def __init__(self, corpus_path="./corpus/train.pos"):
        self.likelihoods = defaultdict(lambda: defaultdict(int))
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.vocabularies = set()
        self.unique_pos_tags = []
        self.DEFAULT_PROB = 1e-7  # Probability for OOV words 1*e-7
        self._train(corpus_path)

    def _read_corpus(self, corpus_path):
        """
        Reads the corpus and extracts words and their corresponding POS tags.
        """
        words = []
        pos_tags = []

        with open(corpus_path, "r") as file:
            for line in file:
                if line.strip():
                    word, pos = line.split()
                    words.append(word)
                    pos_tags.append(pos)
                else:
                    words.extend(["</s>", "<s>"])
                    pos_tags.extend(["END_SENT", "BEGIN_SENT"])
        # Ensure proper closure of the last sentence
        if words[-1] != "</s>":
            words.append("</s>")
            pos_tags.append("END_SENT")

        return words, pos_tags

    def _calculate_likelihoods_and_transitions(self, words, pos_tags):
        """
        Calculates likelihoods and transitions based on the extracted corpus data.
        """
        for word, pos in zip(words, pos_tags):
            self.likelihoods[pos][word] += 1

        for i in range(len(pos_tags) - 1):
            self.transitions[pos_tags[i]][pos_tags[i + 1]] += 1

    def _convert_to_probabilities(self):
        """
        Converts counts to probabilities for both likelihoods and transitions.
        """
        for pos in self.likelihoods:
            total = sum(self.likelihoods[pos].values())
            for word in self.likelihoods[pos]:
                self.likelihoods[pos][word] /= total

        for pos in self.transitions:
            total = sum(self.transitions[pos].values())
            for next_pos in self.transitions[pos]:
                self.transitions[pos][next_pos] /= total

    def _train(self, corpus_path):
        """
        Trains the HMM model by reading the corpus, calculating likelihoods and transitions,
        and then converting these counts into probabilities.
        """
        words, pos_tags = self._read_corpus(corpus_path)
        self._calculate_likelihoods_and_transitions(words, pos_tags)
        self._convert_to_probabilities()
        self.vocabularies = set(words)
        self.unique_pos_tags = list(self.transitions.keys())

        # Convert probabilities to log probabilities
        self.log_transitions = {k: {kk: np.log(vv) for kk, vv in v.items()} for k, v in self.transitions.items()}
        # self.log_likelihoods = {k: {kk: np.log(vv) for kk, vv in v.items()} for k, v in self.likelihoods.items()}
        self.log_default_prob = np.log(self.DEFAULT_PROB)

    def get_pos_tags(self, sentence):
        """
        Public method to get the POS tags for a given sentence.
        """
        if isinstance(sentence, str):
            tokens = sentence.split()
        elif isinstance(sentence, list):
            tokens = sentence
        else:
            raise ValueError("Invalid input type. Please provide a string or a list.")

        # Prepend and append start/end tokens if not already present
        if tokens[0] != "<s>":
            tokens = ["<s>"] + tokens + ["</s>"]
        return self._viterbi_algorithm(tokens)

    def _get_log_likelihood(self, pos_tag, token):
        """
        Private method for dealing with token, including OOV handling
        """
        likelihood = self.likelihoods[pos_tag].get(token, self.DEFAULT_PROB)

        if token[0].isupper() and pos_tag in ["NNP", "NNPS", "NN"]:
            likelihood *= 16
        if token.endswith("ing") and pos_tag in ["VBG", "NN", "JJ"]:
            likelihood *= 8
        if token.endswith("ed") and pos_tag in ["VBD", "VBN", "JJ"]:
            likelihood *= 8
        if token.endswith("ly") and pos_tag in ["RB", "JJ"]:
            likelihood *= 8
        if token.endswith("s") and pos_tag in ["NNS", "VBZ", "POS"]:
            likelihood *= 16
        if pos_tag in ["NNS", "NNPS"] and not (token.endswith("es") or token.endswith("s")):
            likelihood *= 1e-10
        if token.endswith("es") and pos_tag in ["VBZ", "NNS"]:
            likelihood *= 8
        if token.endswith("er") and pos_tag in ["JJ", "RB"]:
            likelihood *= 4.4

        log_likelihood = np.log(likelihood)

        return log_likelihood

    def _viterbi_algorithm(self, tokens):
        """
        Private method implementing the Viterbi algorithm using log probabilities.
        """
        num_tags = len(self.unique_pos_tags)
        num_tokens = len(tokens)

        # Initialize matrices
        viterbi = np.full((num_tags, num_tokens), -np.inf)
        backpointer = np.zeros((num_tags, num_tokens), dtype=int)

        # Handle the first actual word (assuming tokens[0] is a start symbol or similar)
        token = tokens[1]  # First actual word
        for i, pos_tag in enumerate(self.unique_pos_tags):
            emission_prob = self._get_log_likelihood(pos_tag, token)
            viterbi[i, 1] = self.log_transitions["BEGIN_SENT"].get(pos_tag, self.log_default_prob) + emission_prob

        # Loop through the rest of the tokens
        for j in range(2, num_tokens):  # Start from the third token in the list
            for i, pos_tag in enumerate(self.unique_pos_tags):
                probs = viterbi[:, j - 1] + np.array(
                    [self.log_transitions[self.unique_pos_tags[k]].get(pos_tag, self.log_default_prob) for k in
                     range(num_tags)])
                max_prob_index = np.argmax(probs)
                viterbi[i, j] = probs[max_prob_index] + self._get_log_likelihood(pos_tag, tokens[j])
                backpointer[i, j] = max_prob_index

        # Backtrace
        best_path_pointer = np.argmax(viterbi[:, -1])
        best_path = [self.unique_pos_tags[best_path_pointer]]

        for j in range(len(tokens) - 1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, j]
            best_path.append(self.unique_pos_tags[best_path_pointer])

        # Reverse the path, omitting the start and end tokens
        return best_path[-2:0:-1]


def predict(path_to_corpus):
    hmm = HMM()

    sentences = []
    sentence = []
    with open(path_to_corpus, "r") as file:
        for line in file:
            word = line.strip()
            if word:
                sentence.append(word)
            else:
                sentences.append(sentence)
                sentence = []
    if sentence:  # Add the last sentence if not empty
        sentences.append(sentence)

    path_to_corpus = path_to_corpus.replace(".words", "_predicted.pos")

    with open(path_to_corpus, "w") as file:
        for sentence in sentences:
            pos_tags = hmm.get_pos_tags(sentence)
            for word, pos in zip(sentence, pos_tags):
                file.write(f"{word}\t{pos}\n")
            file.write("\n")


if __name__ == "__main__":
    predict('./corpus/WSJ_24.words')
