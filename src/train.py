import random
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from .losses import loss_fn, BCEFocal2WayLoss
from .meter import AverageMeter, MetricMeter


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


def cutmix_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def mixup_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def train_fn(model, data_loader, device, optimizer, scheduler, apex=True, tqdm_disable=False):
    model.train()
    scaler = GradScaler(enabled=apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), disable=tqdm_disable)

    for data in tk0:
        optimizer.zero_grad()
        inputs = data["image"].to(device)
        targets = data["targets"].to(device)
        with autocast(enabled=apex):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            tk0.set_postfix(loss=losses.avg)
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            tk0.set_postfix(loss=losses.avg, lr=lr)
    return scores.avg, losses.avg


def train_mixup_cutmix_fn(model, data_loader, device, optimizer, scheduler, apex=True, tqdm_disable=False):
    model.train()
    scaler = GradScaler(enabled=apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), disable=tqdm_disable)

    for data in tk0:
        optimizer.zero_grad()
        inputs = data["image"].to(device)
        targets = data["targets"].to(device)

        if np.random.rand() < 0.5:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            with autocast(enabled=apex):
                outputs = model(inputs)
                loss = mixup_criterion(outputs, new_targets)
        else:
            inputs, new_targets = cutmix(inputs, targets, 0.4)
            with autocast(enabled=apex):
                outputs = model(inputs)
                loss = cutmix_criterion(outputs, new_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), inputs.size(0))
        scores.update(new_targets[0], outputs)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            tk0.set_postfix(loss=losses.avg)
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            tk0.set_postfix(loss=losses.avg, lr=lr)
    return scores.avg, losses.avg
