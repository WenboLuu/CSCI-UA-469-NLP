import json
import re
from utils import compute_metrics

with open("/scratch/wl2707/research/pii_detection/data/original/train.json", "r") as file:
    data = json.load(file)

patterns = {"EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}", "ID_NUM": r"\b\d{6,10}\b",
            "NAME_STUDENT": r"\b[A-Z][a-z]{1,10}( [A-Z][a-z]{1,10})+\b",
            "PHONE_NUM": r"\b\d{10}\b|\b(?:\d{3}-){2}\d{4}\b|\b(?:\d{3} )\d{3} \d{4}\b|\b\+\d{1,3}\s?\(?\d{1,3}\)?[\s-]?\d{3}[\s-]?\d{4}\b",
            "STREET_ADDRESS": r"\b\d{1,4} [A-Za-z ]+\b", "URL_PERSONAL": r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\S*",
            "USERNAME": r"\b(?=[a-z]*\d)(?=\d*[a-z])[a-z\d]+\b", }


def evaluate_model(data, patterns):
    all_true_labels = []
    all_pred_labels = []

    for doc in data:
        tokens = doc["tokens"]
        true_labels = doc["labels"]
        all_true_labels.extend(true_labels)

        full_text = " ".join(tokens)
        pred_labels = ["O"] * len(tokens)

        for label, pattern in patterns.items():
            for match in re.finditer(pattern, full_text):
                start, end = match.span()
                for i in range(len(tokens)):
                    token_start = full_text.find(tokens[i])
                    token_end = token_start + len(tokens[i])
                    if start <= token_start < token_end:
                        if pred_labels[i] == "O":
                            pred_labels[i] = "B-" + label
                        else:
                            pred_labels[i] = "I-" + label

        all_pred_labels.extend(pred_labels)

    return compute_metrics(all_pred_labels, all_true_labels)


if __name__ == "__main__":
    precision, recall, f1 = evaluate_model(data, patterns)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
