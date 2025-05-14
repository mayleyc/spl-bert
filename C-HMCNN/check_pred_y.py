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

pred_y_file = "pred_y/20250514-150737_250508_model-d/predicted_test_ConstrainedFFNNModel_oeFalse_250508_model-d_20250514-150737.csv"

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
    
def compare_hierarchy_violations(predictions, ohe_dict): #convert to tuples for hashability -> faster?
    # Convert all values to tuples once and store in a set
    allowed_set = {tuple(v) for v in ohe_dict.values()}

    count = 0
    for i in predictions:
        i_tuple = tuple(i)  # Convert prediction to tuple
        if i_tuple not in allowed_set:
            count += 1
    return count

ohe_dict_from_csv, _ = get_one_hot_labels_all(csv_path_full)

#include ME violations as hierarchy violations


def main(): 
    df = pd.read_csv(pred_y_file)

    # Convert to NumPy array
    predictions = df.values
    non_exclusive_rows, zero_rows = count_me(predictions)

    hv_count = compare_hierarchy_violations(predictions, ohe_dict_from_csv)
    
    print(f"Total rows with ME violations: {len(non_exclusive_rows)}\nTotal rows with 0 species predicted: {len(zero_rows)}")
    print(f"Total rows with hierarchy violations: {hv_count}")
if __name__ == "__main__":
    main()