#!/usr/bin/env python3
"""Training script for BirdCLEF 2022."""

import ast
import gc
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from BirdCLEF_DataModule import BirdCLEFDataModule
from BirdCLEF_Model import BirdCLEFModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

config = Namespace(
    data_path="/home/egortrushin/datasets/birdclef-2022",
    folds=Namespace(n_splits=5, random_state=42),
    train_folds=[0],
    seed=71,
    data_module=Namespace(
        train_bs=32,
        valid_bs=128,
        workers=8,
        AudioParams={
            "sr": 32000,
            "fmin": 20,
            "fmax": 16000,
            "hop_length": 512,
        },
    ),
    trainer=Namespace(
        gpus=1,
        max_epochs=50,
        precision=16,
        deterministic=True,
        stochastic_weight_avg=False,
        progress_bar_refresh_rate=1,
    ),
    model=Namespace(
        p_spec_augmenter=0.25,
        n_mels=128,
        base_model=Namespace(model_name="tf_efficientnet_b0_ns", pretrained=True, in_chans=3),
        optimizer_params={"lr": 1.0e-3, "weight_decay": 0.01},
        scheduler=Namespace(
            name="CosineAnnealingLR",
            scheduler_params={"CosineAnnealingLR": {"T_max": 500, "eta_min": 1.0e-6, "last_epoch": -1}},
        ),
    ),
    es_callback={"monitor": "val_loss", "mode": "min", "patience": 8},
    ckpt_callback={
        "monitor": "val_score",
        "dirpath": "ckpts",
        "mode": "max",
        "save_top_k": 1,
        "verbose": 1,
    },
)

config.data_module.AudioParams["n_mels"] = config.model.n_mels


def process_data(data_path):
    """Read and process metadata file."""
    df = pd.read_csv(Path(data_path, "train_metadata.csv"))
    df["new_target"] = df["primary_label"] + " " + df["secondary_labels"].map(lambda x: " ".join(ast.literal_eval(x)))
    df["len_new_target"] = df["new_target"].map(lambda x: len(x.split()))
    df["file_path"] = data_path + "/train_audio/" + df["filename"]
    return df


def create_folds(df, **kwargs):
    """Perform fold splitting."""
    Fold = StratifiedKFold(shuffle=True, **kwargs)
    for n, (trn_index, val_index) in enumerate(Fold.split(df, df["primary_label"])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


if __name__ == "__main__":
    df = process_data(config.data_path)
    df = create_folds(df, **vars(config.folds))

    gc.enable()

    pl.seed_everything(config.seed)

    for fold in config.train_folds:
        print(f"\n###### Fold {fold}")

        train_df = df[df.kfold != fold].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)

        data_module = BirdCLEFDataModule(train_df, valid_df, config.data_module)

        chkpt_callback = ModelCheckpoint(
            filename=f"f{fold}-{{val_score:.5f}}-{{val_loss:.5f}}",
            **config.ckpt_callback,
        )
        es_callback = EarlyStopping(**config.es_callback)

        trainer = pl.Trainer(
            callbacks=[chkpt_callback, es_callback],
            logger=None,
            **vars(config.trainer),
        )

        model = BirdCLEFModel(config.model)
        trainer.fit(model, data_module)

        del data_module, trainer, model, chkpt_callback, es_callback
        torch.cuda.empty_cache()
        gc.collect()
