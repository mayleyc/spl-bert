import os
import datetime
import json
from time import perf_counter
import copy
import pickle
import glob

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score, 
    average_precision_score, 
    hamming_loss, 
    jaccard_score
)

# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'hmc-utils'))
sys.path.append(os.path.join(sys.path[0],'hmc-utils', 'pypsdd'))

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE
from pysdd.sdd import SddManager, Vtree

# misc
from common import *

from torch.utils.data import Dataset
from PIL import Image

import re
import numpy as np
import pandas as pd

pred_y_file = os.path.join(pred_y_folder, "20250424-112958_250425_model-a/predicted_test_ConstrainedFFNNModel_oeFalse_250425_model-a_20250424-112958.csv")

emb_file = find_latest_emb_file(emb_model_name, "cub_others")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#Check for mutual exclusivity violations
def count_me(predictions):
    # Select last 200 columns (species-level predictions)
    species_preds = predictions[:, -200:]

    # Count how many 1s (i.e., positive predictions) per row
    species_counts = np.sum(species_preds, axis=1)

    # Identify rows with incorrect number of predictions
    non_exclusive_rows = np.where(species_counts > 1)[0]
    zero_rows = np.where(species_counts == 0)[0]

    # Count and optionally display some examples
    return non_exclusive_rows, zero_rows


# shouldn't use emb_file, instead get ohe_dict from the csv file
with open(emb_file, "rb") as f:
    all_paths, all_embeddings, labels_unprocessed = pickle.load(f)
label_species = [label.split('.')[-1] for label in labels_unprocessed]
label_species = [re.sub('_', ' ', label) for label in label_species] # the species-level label for each image

ohe_dict, _ = get_one_hot_labels(label_species, csv_path_full)


def main(): 
    df = pd.read_csv(pred_y_file)

    # Convert to NumPy array
    predictions = df.values
    non_exclusive_rows, zero_rows = count_me(predictions)

    print(f"Total rows with ME violations: {len(non_exclusive_rows)}\nTotal rows with 0 species predicted: {len(zero_rows)}")

if __name__ == "__main__":
    main()