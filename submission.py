#!/usr/bin/env python3

import torch
import glob
import yaml
import pandas as pd
from pathlib import Path
from src.sub_utils import TimmSED, SCORED_BIRDS, prediction

CHKPT_PATH = "ckpt/"
DATA_PATH = "/home/egortrushin/datasets/birdclef-2022"

threshold = 0.02
threshold_long = None

with open(Path(CHKPT_PATH, "config.yaml"), "r") as file_obj:
    config = yaml.safe_load(file_obj)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_DIR = Path(DATA_PATH, "test_soundscapes")
all_audios = list(AUDIO_DIR.glob("*.ogg"))

sample_submission = pd.read_csv(Path(DATA_PATH, "sample_submission.csv"))

model_paths = glob.glob(CHKPT_PATH + 'fold-*.bin')

models = []
for p in model_paths:
    model = TimmSED(base_model_name=config["base_model_name"],
                    num_classes=len(SCORED_BIRDS) + 1,
                    n_mels=config["n_mels"],
                    pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(p))
    model.eval()
    models.append(model)

pred_dicts = prediction(all_audios,models,config,threshold=threshold,threshold_long=threshold_long)

for i in range(len(sample_submission)):
    sample = sample_submission.row_id[i]
    key = sample.split("_")[0] + "_" + sample.split("_")[1] + "_" + sample.split("_")[3]
    target_bird = sample.split("_")[2]
    if key in pred_dicts:
        sample_submission.iat[i, 1] = (target_bird in pred_dicts[key])

sample_submission.to_csv("submission.csv", index=False)
