from time import process_time

import numpy as np
import pandas as pd

label2id = {label: i for i, label in enumerate(
    ["B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM", "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME",
     "I-ID_NUM", "I-NAME_STUDENT", "I-PHONE_NUM", "I-STREET_ADDRESS", "I-URL_PERSONAL", "O", ])}
id2label = {i: label for label, i in label2id.items()}


def timeit(func):
    def wrapper(*args, **kwargs):
        start = process_time()
        result = func(*args, **kwargs)
        end = process_time()
        print(f"{func.__name__} took {end - start:.2f} seconds.")
        return result

    return wrapper


def compute_metrics(predicted_labels, true_labels):
    """
    Calculate precision, recall, and F5-score after filtering out non-PII (O) labels
    from both predicted and true labels using micro-averaging.

    Parameters:
    - predicted_labels (list of int): List of predicted label IDs.
    - true_labels (list of int): List of true label IDs.

    Returns:
    - tuple: (precision, recall, F5-score)
    """
    # make two dataframes, one for true labels and one for predicted labels
    df_groundtruth = pd.DataFrame({"index": np.arange(len(true_labels)), "True_Labels": true_labels, })
    df_predictions = pd.DataFrame({"index": np.arange(len(predicted_labels)), "Predicted_Labels": predicted_labels, })

    # filter out non-PII labels
    df_groundtruth = df_groundtruth[df_groundtruth["True_Labels"] != label2id["O"]]
    df_predictions = df_predictions[df_predictions["Predicted_Labels"] != label2id["O"]]

    # outer join
    df = df_predictions.merge(df_groundtruth, on="index", how="outer")
    print(df)

    # Add a column to classify each prediction
    df["confus_mtx"] = ""

    # Define FP, FN, TP based on conditions
    df.loc[df["True_Labels"].isna(), "confus_mtx"] = "FP"
    df.loc[df["Predicted_Labels"].isna(), "confus_mtx"] = "FN"
    df.loc[(df["True_Labels"].notna()) & (df["True_Labels"] != df["Predicted_Labels"]), "confus_mtx"] = "FN"
    df.loc[(df["Predicted_Labels"].notna()) & (df["Predicted_Labels"] == df["True_Labels"]), "confus_mtx"] = "TP"

    FP = (df["confus_mtx"] == "FP").sum()
    FN = (df["confus_mtx"] == "FN").sum()
    TP = (df["confus_mtx"] == "TP").sum()

    # Calculate precision, recall, and F5-score
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    beta = 5
    fbeta_score = ((1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall + 1e-6))

    return precision, recall, fbeta_score
