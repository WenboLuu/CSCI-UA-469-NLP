import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset

from utils import timeit


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length=2048):
        """
        Initializes a Dataset object.

        Args:
            data (list): The input data.
            tokenizer: The tokenizer object used for tokenization.
            label2id (dict): A dictionary mapping labels to their corresponding IDs.
            max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 2048.
        """
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.max_workers = os.cpu_count() if os.cpu_count() <= 42 else 42
        print(f"Tokenizing data with {self.max_workers} workers...")
        self.data = self.multithread_tokenize(data)
        print(f"Done tokenizing data. Length: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def tokenize(self, item):
        """
        Tokenizes the input text and converts labels to numerical format.

        Args:
            item (dict): A dictionary containing the input tokens and labels.

        Returns:
            input_ids (torch.Tensor): The tokenized input text converted to input IDs.
            labels (torch.Tensor): The labels converted to numerical format.
        """
        tokens = item["tokens"]
        labels = item["labels"]
        text = " ".join(tokens)

        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length,
                                  return_tensors="pt", )
        new_labels = [self.label2id[label] for label in labels[: self.max_length]]
        labels_padding = [self.label2id["O"]] * (self.max_length - len(new_labels))
        new_labels.extend(labels_padding)
        return encoding["input_ids"].squeeze(0), torch.tensor(new_labels)

    @timeit
    def multithread_tokenize(self, data):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.tokenize, data)

        return list(results)

    def __getitem__(self, idx):
        return self.data[idx]
