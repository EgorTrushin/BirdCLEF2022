#!/usr/bin/env python3

import ast
import random
import os
import torch
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def read_config(filename="config.yaml"):
    with open(filename, "r") as file_obj:
        config = yaml.safe_load(file_obj)
        config["AudioParams"]["n_mels"] = config["n_mels"]
    return config


def process_data(data_path):
    df_train = pd.read_csv(Path(data_path, "train_metadata.csv"))
    df_train["new_target"] = (df_train["primary_label"] + " " +
                              df_train["secondary_labels"].map(
                                  lambda x: " ".join(ast.literal_eval(x))))
    df_train["len_new_target"] = df_train["new_target"].map(
        lambda x: len(x.split()))

    df_train["file_path"] = data_path + '/train_audio/' + df_train['filename']

    return df_train


def create_folds(df_train, **kwargs):
    Fold = StratifiedKFold(shuffle=True, **kwargs)
    for n, (trn_index, val_index) in enumerate(
            Fold.split(df_train, df_train['primary_label'])):
        df_train.loc[val_index, 'kfold'] = int(n)
    df_train['kfold'] = df_train['kfold'].astype(int)
    return df_train


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
