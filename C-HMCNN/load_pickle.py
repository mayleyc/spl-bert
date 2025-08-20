import pickle
from timeit import default_timer as timer
import pandas as pd


def load_pickle(filepath):
    """
    Load a Python object from a pickle file.

    Args:
        filepath (str): Path to the .pickle file.

    Returns:
        Any: The object stored in the pickle file.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    path = "embeddings/emb_bert-base-uncased_amazon_20250806-134354_train_batch0.pickle"

    start = timer()
    

    try:
        texts, embeddings, labels = load_pickle(path)
        print(f"Loaded {len(texts)} texts.")
        print(f"Emb length: {len(embeddings[0])}")
        print(f"first label: {labels[0]}")
    except Exception as e:
        print("Error unpacking pickle contents:", e)

    # Check if a label is in the csv file
    label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0] #labels[0]

    # Load CSV into a DataFrame
    df = pd.read_csv("amazon_tax_one_hot.csv", header=None)

    # Drop the first column
    df_no_first = df.iloc[:, 1:]

    label_series = pd.Series(label)

    # Check if any row matches the label exactly
    matching_rows = df_no_first.apply(lambda row: row.tolist() == label, axis=1)

    if matching_rows.any():
        matching_indices = matching_rows[matching_rows].index.tolist()
        print("Label found on line(s):", [i + 1 for i in matching_indices])
    else:
        print("Label not found in CSV.")


    end = timer()
    print(f"Loaded pickle in {end - start:.4f} seconds")

