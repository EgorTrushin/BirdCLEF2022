#!/usr/bin/env python3
"""Training script for BirdCLEF 2022."""

from argparse import Namespace
from pathlib import Path

import glob
import albumentations as A
import librosa
import numpy as np
import pandas as pd
import torch
from BirdCLEF_DataModule import ALL_BIRDS, compute_melspec, mono_to_color
from BirdCLEF_Model import BirdCLEFModel

config = Namespace(
    data_path="/home/egortrushin/datasets/birdclef-2022",
    folds=Namespace(n_splits=5, random_state=42),
    train_folds=[0, 1, 2, 3, 4],
    seed=71,
    data_module=Namespace(
        train_bs=24,
        valid_bs=128,
        workers=8,
        AudioParams={
            "sr": 32000,
            "duration": 5,
            "fmin": 20,
            "fmax": 16000,
        },
    ),
    trainer=Namespace(
        gpus=1,
        max_epochs=2,  # 50,
        precision=16,
        deterministic=True,
        stochastic_weight_avg=False,
        gradient_clip_val=None,
        progress_bar_refresh_rate=1,
    ),
    model=Namespace(
        p_spec_augmenter=1.0,
        n_mels=128,
        base_model=Namespace(
            model_name="tf_efficientnet_b0_ns", pretrained=True, in_chans=3
        ),
        optimizer_params={"lr": 1.0e-3, "weight_decay": 0.01},
        scheduler=Namespace(
            name="CosineAnnealingLR",
            scheduler_params={
                "CosineAnnealingLR": {"T_max": 500, "eta_min": 1.0e-6, "last_epoch": -1}
            },
        ),
    ),
    es_callback={"monitor": "val_loss", "mode": "min", "patience": 8},
    ckpt_callback={
        "monitor": "val_score",
        "dirpath": "ckpts",
        "mode": "max",
        "save_top_k": 2,
        "verbose": 1,
    },
)

config.data_module.AudioParams["n_mels"] = config.model.n_mels


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, clip, AudioParams):
        self.df = df
        self.clip = np.concatenate([clip, clip, clip])
        self.AudioParams = AudioParams

        mean = (0.485, 0.456, 0.406)  # RGB
        std = (0.229, 0.224, 0.225)  # RGB

        self.albu_transforms = {
            "valid": A.Compose(
                [
                    A.Normalize(mean, std),
                ]
            ),
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        image = self.clip[
            self.AudioParams["sr"]
            * start_seconds : self.AudioParams["sr"]
            * end_seconds
        ].astype(np.float32)
        image = np.nan_to_num(image)

        image = compute_melspec(image, self.AudioParams)
        image = mono_to_color(image)
        image = image.astype(np.uint8)

        image = self.albu_transforms["valid"](image=image)["image"].T

        return {
            "image": image,
            "row_id": row_id,
        }


def prediction_for_clip(
    test_df, clip, models, config, threshold=0.05, threshold_long=None
):

    dataset = TestDataset(
        df=test_df,
        clip=clip,
        AudioParams=config["AudioParams"],
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_dict = {}
    for data in loader:
        row_id = data["row_id"][0]
        image = data["image"].to(device)

        with torch.no_grad():
            probas = []
            probas_long = []
            for model in models:
                with torch.cuda.amp.autocast():
                    output = model(image)
                probas.append(
                    output["clipwise_output"].detach().cpu().numpy().reshape(-1)
                )
            probas = np.array(probas)
        if threshold_long is None:
            events = probas.mean(0) >= threshold
        else:
            events = (
                (probas.mean(0) >= threshold).astype(int)
                + (probas_long.mean(0) >= threshold_long).astype(int)
            ) >= 2
        labels = np.argwhere(events).reshape(-1).tolist()
        if len(labels) == 0:
            prediction_dict[str(row_id)] = "nocall"
        else:
            labels_str_list = list(map(lambda x: ALL_BIRDS[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[str(row_id)] = label_string
    return prediction_dict


def prediction(test_audios, models, config, threshold=0.05, threshold_long=None):

    prediction_dicts = {}
    for audio_path in test_audios:
        clip, _ = librosa.load(audio_path, sr=config["AudioParams"]["sr"])

        seconds = []
        row_ids = []
        for second in range(5, 65, 5):
            row_id = "_".join(audio_path.name.split(".")[:-1]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)
        test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
        prediction_dict = prediction_for_clip(
            test_df,
            clip=clip,
            models=models,
            threshold=threshold,
            threshold_long=threshold_long,
            config=config,
        )
        prediction_dicts.update(prediction_dict)
    return prediction_dicts


CHKPT_PATH = "ckpts/"
DATA_PATH = "/home/egortrushin/datasets/birdclef-2022"

threshold = 0.02
threshold_long = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_DIR = Path(DATA_PATH, "test_soundscapes")
all_audios = list(AUDIO_DIR.glob("*.ogg"))

sample_submission = pd.read_csv(Path(DATA_PATH, "sample_submission.csv"))

models = []
for path in glob.glob(CHKPT_PATH + "/*.ckpt"):
    print(path)
    model = BirdCLEFModel.load_from_checkpoint(path, config=config.model)
    model.to(device)
    model.eval()
    models.append(model)

pred_dicts = prediction(
    all_audios,
    models,
    config=vars(config.data_module),
    threshold=threshold,
    threshold_long=threshold_long,
)

for i in range(len(sample_submission)):
    sample = sample_submission.row_id[i]
    key = sample.split("_")[0] + "_" + sample.split("_")[1] + "_" + sample.split("_")[3]
    target_bird = sample.split("_")[2]
    if key in pred_dicts:
        sample_submission.iat[i, 1] = target_bird in pred_dicts[key]

sample_submission.to_csv("submission.csv", index=False)
