#!/usr/bin/env python3
"""Training script for BirdCLEF 2022."""

import glob
import os
import shutil
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from src.data import ALL_BIRDS, AllBirdsDataset
from src.logger import init_logger
from src.model import TimmSED
from src.train import train_fn, train_mixup_cutmix_fn
from src.utils import create_folds, get_device, process_data, read_config, set_seed
from src.valid import valid_fn

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = read_config()
    if os.path.isdir(config["output_path"]):
        shutil.rmtree(config["output_path"])
    os.mkdir(config["output_path"])
    with open(Path(config["output_path"], "config.yaml"), "w") as file_obj:
        yaml.dump(config, file_obj, default_flow_style=False)
    df = process_data(config["data_path"])

    if config["nocall_path"]:
        nocall = glob.glob(config["nocall_path"] + "/*/*/*.wav")
        df_nocall = pd.DataFrame()
        df_nocall["file_path"] = nocall
        df_nocall["new_target"] = "nocall"
        df_nocall["primary_label"] = "nocall"
        df = pd.concat([df, df_nocall]).reset_index()
        ALL_BIRDS.append("nocall")

    df = create_folds(df, **config["folds"])

    logger = init_logger(log_file=Path(config["output_path"], "train.log"))
    set_seed(config["seed"])
    device = get_device()

    for fold in config["train_folds"]:
        logger.info(f"\n###### Fold {fold}")

        trn_df = df[df.kfold != fold].reset_index(drop=True)
        val_df = df[df.kfold == fold].reset_index(drop=True)

        train_dataset = AllBirdsDataset(trn_df, config["AudioParams"], config["image_size"], mode="train")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config["train_bs"], num_workers=config["workers"], pin_memory=False, shuffle=True
        )

        valid_dataset = AllBirdsDataset(val_df, config["AudioParams"], config["image_size"], mode="valid")
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config["valid_bs"], num_workers=config["workers"], pin_memory=False, shuffle=False
        )

        model = TimmSED(
            base_model_name=config["base_model_name"],
            num_classes=len(ALL_BIRDS),
            n_mels=config["n_mels"],
            p_spec_augmenter=config["p_spec_augmenter"],
        )

        optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
        if config["lr_scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **config["lr_scheduler"]["ReduceLROnPlateau"]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **config["lr_scheduler"]["CosineAnnealingLR"]
            )

        model = model.to(device)

        best_score = -np.inf

        for epoch in range(config["epochs"]):
            logger.info("#### Epoch {}".format(epoch + 1))

            start_time = time.time()

            if epoch < config["cutmix_and_mixup_epochs"]:
                train_avg, train_loss = train_mixup_cutmix_fn(
                    model,
                    train_dataloader,
                    device,
                    optimizer,
                    scheduler,
                    apex=config["apex"],
                    tqdm_disable=config["tqdm_disable"],
                )
            else:
                train_avg, train_loss = train_fn(
                    model,
                    train_dataloader,
                    device,
                    optimizer,
                    scheduler,
                    apex=config["apex"],
                    tqdm_disable=config["tqdm_disable"],
                )

            valid_avg, valid_loss = valid_fn(model, valid_dataloader, device, tqdm_disable=config["tqdm_disable"])

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch+1} - train_loss: {train_loss:.5f}  val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s"
            )

            logger.info(f"trn_m02:{train_avg['m_0.2']: 0.5f}  val_m02:{valid_avg['m_0.2']: 0.5f}")
            logger.info(f"trn_m03:{train_avg['m_0.3']: 0.5f}  val_m03:{valid_avg['m_0.3']: 0.5f}")
            logger.info(f"trn_m04:{train_avg['m_0.4']: 0.5f}  val_m04:{valid_avg['m_0.4']: 0.5f}")
            logger.info(f"trn_m05:{train_avg['m_0.5']: 0.5f}  val_m05:{valid_avg['m_0.5']: 0.5f}")

            logger.info(f"trn_m02a:{train_avg['m_0.2a']: 0.5f}  val_m02a:{valid_avg['m_0.2a']: 0.5f}")
            logger.info(f"trn_m03a:{train_avg['m_0.3a']: 0.5f}  val_m03a:{valid_avg['m_0.3a']: 0.5f}")
            logger.info(f"trn_m04a:{train_avg['m_0.4a']: 0.5f}  val_m04a:{valid_avg['m_0.4a']: 0.5f}")
            logger.info(f"trn_m05a:{train_avg['m_0.5a']: 0.5f}  val_m05a:{valid_avg['m_0.5a']: 0.5f}")

            logger.info(f"trn_f1_02:{train_avg['f1_0.2']: 0.5f}  val_f1_02:{valid_avg['f1_0.2']: 0.5f}")
            logger.info(f"trn_f1_03:{train_avg['f1_0.3']: 0.5f}  val_f1_03:{valid_avg['f1_0.3']: 0.5f}")
            logger.info(f"trn_f1_04:{train_avg['f1_0.4']: 0.5f}  val_f1_04:{valid_avg['f1_0.4']: 0.5f}")
            logger.info(f"trn_f1_05:{train_avg['f1_0.5']: 0.5f}  val_f1_05:{valid_avg['f1_0.5']: 0.5f}")

            logger.info(f"trn_f1_02a:{train_avg['f1_0.2a']: 0.5f}  val_f1_02a:{valid_avg['f1_0.2a']: 0.5f}")
            logger.info(f"trn_f1_03a:{train_avg['f1_0.3a']: 0.5f}  val_f1_03a:{valid_avg['f1_0.3a']: 0.5f}")
            logger.info(f"trn_f1_04a:{train_avg['f1_0.4a']: 0.5f}  val_f1_04a:{valid_avg['f1_0.4a']: 0.5f}")
            logger.info(f"trn_f1_05a:{train_avg['f1_0.5a']: 0.5f}  val_f1_05a:{valid_avg['f1_0.5a']: 0.5f}")

            new_score = valid_avg[config["target_metric"]]
            if new_score > best_score:
                logger.info(f">>>>>>>> Model Improved From {best_score} ----> {new_score}")
                torch.save(model.state_dict(), Path(config["output_path"], f"fold-{fold}.bin"))
                best_score = new_score
