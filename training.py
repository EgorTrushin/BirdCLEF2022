#!/usr/bin/env python3

import os
import torch
import time
import yaml
import shutil
from pathlib import Path
from src.utils import *
from src.logger import init_logger
from src.data import AllBirdsDataset, ALL_BIRDS
from src.model import TimmSED
from src.train import train_fn, train_mixup_cutmix_fn
from src.valid import valid_fn

if __name__ == '__main__':
    config = read_config()
    if os.path.isdir(config["output_path"]):
        shutil.rmtree(config["output_path"])
    os.mkdir(config["output_path"])
    with open(Path(config["output_path"], "config.yaml"), 'w') as file_obj:
        yaml.dump(config, file_obj, default_flow_style=False)
    df = process_data(config["data_path"])#.sample(frac=0.02).reset_index()
    df = create_folds(df, **config["folds"])

    logger = init_logger(log_file=Path(config["output_path"], "train.log"))
    set_seed(config["seed"])
    device = get_device()

    for fold in config["train_folds"]:
        logger.info("=" * 100)
        logger.info(f"Fold {fold} Training")
        logger.info("=" * 100)

        trn_df = df[df.kfold != fold].reset_index(drop=True)
        val_df = df[df.kfold == fold].reset_index(drop=True)

        train_dataset = AllBirdsDataset(trn_df,
                                        config["AudioParams"],
                                        config["image_size"],
                                        mode='train')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["train_bs"],
            num_workers=config["workers"],
            pin_memory=False,
            shuffle=True)

        valid_dataset = AllBirdsDataset(val_df,
                                           config["AudioParams"],
                                           config["image_size"],
                                           mode='valid')
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config["valid_bs"],
            num_workers=config["workers"],
            pin_memory=False,
            shuffle=False)

        model = TimmSED(base_model_name=config["base_model_name"],
                        num_classes=len(ALL_BIRDS),
                        n_mels=config["n_mels"],
                        p_spec_augmenter=config["p_spec_augmenter"])

        optimizer = torch.optim.AdamW(model.parameters(),
                                      **config["optimizer"])
        if config["lr_scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **config["lr_scheduler"]["ReduceLROnPlateau"])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **config["lr_scheduler"]["CosineAnnealingLR"])

        model = model.to(device)

        best_score = -np.inf

        for epoch in range(config["epochs"]):
            logger.info("Starting {} epoch...".format(epoch + 1))

            start_time = time.time()

            if epoch < config["cutmix_and_mixup_epochs"]:
                train_avg, train_loss = train_mixup_cutmix_fn(
                    model,
                    train_dataloader,
                    device,
                    optimizer,
                    scheduler,
                    apex=config["apex"],
                    tqdm_disable=config["tqdm_disable"])
            else:
                train_avg, train_loss = train_fn(
                    model,
                    train_dataloader,
                    device,
                    optimizer,
                    scheduler,
                    apex=config["apex"],
                    tqdm_disable=config["tqdm_disable"])

            valid_avg, valid_loss = valid_fn(
                model,
                valid_dataloader,
                device,
                tqdm_disable=config["tqdm_disable"])

            if isinstance(scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)

            elapsed = time.time() - start_time

            logger.info(
                f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s'
            )
            logger.info(
                f"trn_02:{train_avg['m_0.2']: 0.5f}  val_02:{valid_avg['m_0.2']: 0.5f}"
            )
            logger.info(
                f"trn_03:{train_avg['m_0.3']: 0.5f}  val_03:{valid_avg['m_0.3']: 0.5f}"
            )
            logger.info(
                f"trn_035:{train_avg['m_0.35']: 0.5f}  val_035:{valid_avg['m_0.35']: 0.5f}"
            )
            logger.info(
                f"trn_04:{train_avg['m_0.4']: 0.5f}  val_04:{valid_avg['m_0.4']: 0.5f}"
            )
            logger.info(
                f"trn_045:{train_avg['m_0.45']: 0.5f}  val_045:{valid_avg['m_0.45']: 0.5f}"
            )
            logger.info(
                f"trn_05:{train_avg['m_0.5']: 0.5f}  val_05:{valid_avg['m_0.5']: 0.5f}"
            )

            new_score = valid_avg[config["target_metric"]]
            if new_score > best_score:
                logger.info(
                    f">>>>>>>> Model Improved From {best_score} ----> {new_score}"
                )
                torch.save(model.state_dict(), Path(config["output_path"], f'fold-{fold}.bin'))
                best_score = new_score
