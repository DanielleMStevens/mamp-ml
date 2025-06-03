import math
import sys
import os
import numpy as np
import pandas as pd
from functools import partial
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.cross_entropy import CrossEntropyLoss
from losses.supcon import SupConLoss

from transformers.tokenization_utils_base import BatchEncoding
import misc
import wandb
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
    top_k_accuracy_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt

def move_to_device(obj, device):
    """
    Recursively moves all PyTorch tensors in a nested dictionary to the specified device.
    
    Args:
    obj: The object to move (can be a dict, list, tuple, or tensor)
    device: The PyTorch device to move the tensors to

    Returns:
    The same object structure with all tensors moved to the specified device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, BatchEncoding):
        return BatchEncoding({k: move_to_device(v, device) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj

loss_dict = {
    "ce": CrossEntropyLoss(),
    "supcon": SupConLoss()
}

def train_one_epoch(
    model: torch.nn.Module,
    dl,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
):
    ## prepare training
    model.train()
    optimizer.zero_grad()

    ## prepare logging
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    lists = {"gt": [], "pr": [], "x": []}
    for batch_idx, batch in enumerate(metric_logger.log_every(dl, print_freq, header)):
        misc.adjust_learning_rate(optimizer, batch_idx / len(dl) + epoch, args)

        ## move inputs/outputs to cuda
        batch = move_to_device(batch, device)
        ## forward
        output = model(batch['x'])
        all_losses = {}
        model_with_losses = model.module if hasattr(model, "module") else model
        for loss_name in model_with_losses.losses:
            losses = loss_dict[loss_name](output, batch)
            all_losses.update(losses)

        total_loss = sum(all_losses.values())

        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)

        ## backward
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

        ## logging
        gt = batch['y']
        model_with_losses = model.module if hasattr(model, "module") else model
        preds = model_with_losses.get_pr(output)

        stats = model_with_losses.get_stats(gt, preds, train=True)
        lr = optimizer.param_groups[0]["lr"]
        losses_detach = {f"train_{k}": v.cpu().item() for k, v in all_losses.items()}
        metric_logger.update(lr=lr)
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(**losses_detach)
        metric_logger.update(**stats)
        if not args.disable_wandb and misc.is_main_process():
            wandb.log(
                {
                    "train_loss": total_loss.item(),
                    "lr": lr,
                    **losses_detach,
                    **stats,
                }
            )
            
        lists["gt"].append(batch['y'].cpu())
        lists["pr"].append(preds.cpu())
        model_with_losses = model.module if hasattr(model, "module") else model
        lists["x"].extend(model_with_losses.batch_decode(batch))

    gt_all = torch.cat(lists["gt"])
    prob_all = torch.cat(lists["pr"])

    torch.save(
    {
        "gt": gt_all,       
        "pr": prob_all,
        "x": lists["x"]
    },
    args.output_dir / "train_preds.pth",
    )
    
    ## gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, dl, device, args, output_dir):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    lists = {"gt": [], "pr": [], "x": [], "loss": []}
    for batch in metric_logger.log_every(dl, 10, header):
        ## move inputs/outputs to cuda
        batch = move_to_device(batch, device)
        output = model(batch['x'])
        all_losses = {}
        model_with_losses = model.module if hasattr(model, "module") else model
        for loss_name in model_with_losses.losses:
            losses = loss_dict[loss_name](output, batch)
            all_losses.update(losses)
        for loss_name, loss_val in all_losses.items():
            if loss_name not in lists.keys():
                lists[loss_name] = []
            lists[loss_name].append(loss_val.cpu())
        preds = model_with_losses.get_pr(output)
        lists["gt"].append(batch['y'].cpu())
        lists["pr"].append(preds.cpu())
        lists["x"].extend(model_with_losses.batch_decode(batch))

        total_loss = sum(all_losses.values())
        lists['loss'].append(total_loss.cpu())

    gt_all = torch.cat(lists["gt"])
    prob_all = torch.cat(lists["pr"])
    mean_loss = float(np.mean(lists['loss']))

    model_with_losses = model.module if hasattr(model, "module") else model
    stats = model_with_losses.get_stats(gt_all, prob_all, train=False)

    for loss_name, loss_val in all_losses.items():
        stats[f'test_{loss_name}'] = float(np.mean(lists[loss_name]))
    
    stats['test_loss'] = mean_loss

    # Generate and save ROC curves
    plt.figure(figsize=(10, 8))
    gt_onehot = np.eye(3)[gt_all.cpu()]
    pr_np = prob_all.cpu().numpy()
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(gt_onehot[:, i], pr_np[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {stats[f"test_auroc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(Path(output_dir) / 'roc_curve.png')
    plt.close()

    # Generate and save PR curves
    plt.figure(figsize=(10, 8))
    
    for i in range(3):
        precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
        plt.plot(recall, precision, label=f'Class {i} (AUC = {stats[f"test_auprc_class{i}"]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(Path(output_dir) / 'pr_curve.png')
    plt.close()

    if not args.disable_wandb and misc.is_main_process():
        wandb.log(stats)
        wandb.log({
            "roc_curve": wandb.Image(str(Path(output_dir) / 'roc_curve.png')),
            "pr_curve": wandb.Image(str(Path(output_dir) / 'pr_curve.png'))
        })

    torch.save(
        {
            "gt": gt_all,       
            "pr": prob_all,
            "x": lists["x"]
        },
        output_dir / "test_preds.pth",
    )

    ds_name = dl.dataset.name if hasattr(dl.dataset, 'name') else 'test'
    print(ds_name, stats)

    metric_logger.update(**stats)
    ret = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return ret
