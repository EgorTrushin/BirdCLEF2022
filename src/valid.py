import torch
from tqdm import tqdm
from .meter import AverageMeter, MetricMeter
from .losses import loss_fn


def valid_fn(model, data_loader, device, tqdm_disable=False):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), disable=tqdm_disable)
    with torch.no_grad():
        for data in tk0:
            inputs = data["image"].to(device)
            targets = data["targets"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg
