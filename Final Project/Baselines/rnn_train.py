import json
import os

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import NERDataset
from rnn_models import RNN, LSTM, GRU, BiLSTM
from utils import compute_metrics, label2id
from visualization import plot_label_distribution, plot_loss_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_dataloaders(visualize=False):
    path_to_original_data = "data/original/train.json"
    path_to_extra_data = "data/mpware_mixtral8x7b_v1.1/mpware_mixtral8x7b_v1.1-no-i-username.json"

    # Load data
    with open(path_to_original_data, "r") as f:
        original_data = json.load(f)

    with open(path_to_extra_data, "r") as f:
        extra_data = json.load(f)

    combined_data = original_data + extra_data

    # preprocess data
    # Define tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = NERDataset(combined_data, tokenizer, label2id)

    if visualize:
        plot_label_distribution([label for item in dataset for label in item[1].numpy()])

    train_test_split = 0.8
    trainset_size = int(len(dataset) * train_test_split)
    valset_size = len(dataset) - trainset_size

    trainset, valset = random_split(dataset, [trainset_size, valset_size])

    # DataLoader
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=512, shuffle=False, num_workers=4)

    # Instantiate model and setup training
    vocab_size = tokenizer.vocab_size
    num_labels = len(label2id)

    del tokenizer

    # normalize class weights
    labels = [label for item in trainset for label in item[1].numpy()]
    class_weights = compute_class_weight("balanced", classes=np.arange(0, 13), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / 1e2
    class_weights[12] = 7.7037e-02
    print("Class weights: ", class_weights)

    return train_loader, val_loader, vocab_size, num_labels, class_weights


def define_model(model, vocab_size, num_labels, class_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "rnn":
        model = RNN(vocab_size, embedding_dim=128, hidden_dim=256, num_labels=num_labels)
    elif model == "lstm":
        model = LSTM(vocab_size, embedding_dim=128, hidden_dim=256, num_labels=num_labels)
    elif model == "gru":
        model = GRU(vocab_size, embedding_dim=128, hidden_dim=256, num_labels=num_labels)
    elif model == "bilstm":
        model = BiLSTM(vocab_size, embedding_dim=128, hidden_dim=256, num_labels=num_labels)
    else:
        raise ValueError("Invalid model name.")

    model.train()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)
    return model, optimizer, criterion, lr_scheduler, device


def train_eval(model, optimizer, criterion, lr_scheduler, device, train_loader, val_loader, num_labels,
               visualize=False):
    train_losses, val_losses = [], []
    precision_scores, recall_scores, f5_scores = [], [], []

    threshold = 0.9

    for epoch in tqdm(range(30)):
        model.train()  # Set model to training mode
        total_loss = 0
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            logits = model(input_ids)
            loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update parameters
            total_loss += loss.item()
        average_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        all_predicted_labels = []
        all_true_labels = []
        with torch.no_grad():  # Disable gradient calculation
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits = model(input_ids)

                probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
                class_0_excluded = probabilities[:, label2id["O"]] < threshold
                probabilities[class_0_excluded, label2id["O"]] = -1
                predictions = torch.argmax(probabilities, dim=-1)

                loss = criterion(logits.view(-1, num_labels), labels.view(-1))
                val_loss += loss.item()
                all_predicted_labels.extend(predictions.cpu().view(-1).numpy())
                all_true_labels.extend(labels.cpu().view(-1).numpy())

        average_val_loss = val_loss / len(val_loader)
        lr_scheduler.step(average_val_loss)

        precision, recall, f5 = compute_metrics(all_predicted_labels, all_true_labels)
        print(f"Epoch {epoch + 1}")
        print(f"Training Loss: {average_train_loss:.2f}, Validation Loss: {average_val_loss:.2f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F5: {f5:.2f}")

        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f5_scores.append(f5)

    if visualize:
        plot_loss_metrics(train_losses, val_losses, precision_scores, recall_scores, f5_scores)
