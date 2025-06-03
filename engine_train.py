#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

"""
Training and evaluation engine for a deep learning model.
This module provides the core functionality for training and evaluating machine learning models,
particularly focused on multi-class classification tasks. It includes functionality for
loss computation, metric tracking, and visualization of results.
"""

######################################################################
# import packages and libraries
######################################################################

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
        obj: The object to move (can be a dict, list, tuple, tensor, or BatchEncoding)
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

# Dictionary mapping loss function names to their implementations
loss_dict = {
    "ce": CrossEntropyLoss(),     # Standard cross-entropy loss
    "supcon": SupConLoss()        # Supervised contrastive loss
}

def train_one_epoch(
    model: torch.nn.Module,
    dl,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
):
    """
    Trains the model for one epoch.
    
    Args:
        model: The neural network model to train
        dl: DataLoader containing the training data
        optimizer: The optimizer for updating model parameters
        device: The device (CPU/GPU) to use for training
        epoch: Current epoch number
        args: Additional arguments for training configuration
    
    Returns:
        dict: Dictionary containing averaged training metrics for the epoch
    """
    # Set model to training mode and reset optimizer
    model.train()  # Enables training-specific behaviors like dropout and batch norm
    optimizer.zero_grad()

    # Initialize metric logging
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    # Lists to store predictions and ground truth for the entire epoch
    lists = {"gt": [], "pr": [], "x": []}
    
    # Training loop over batches
    for batch_idx, batch in enumerate(metric_logger.log_every(dl, print_freq, header)):
        # Update learning rate according to schedule
        misc.adjust_learning_rate(optimizer, batch_idx / len(dl) + epoch, args)

        # Move batch to appropriate device
        batch = move_to_device(batch, device)
        
        # Forward pass
        output = model(batch['x'])
        all_losses = {}
        model_with_losses = model.module if hasattr(model, "module") else model
        
        # Calculate all specified losses
        for loss_name in model_with_losses.losses:
            losses = loss_dict[loss_name](output, batch)
            all_losses.update(losses)

        total_loss = sum(all_losses.values())

        # Check for invalid loss values
        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)

        # Backward pass and optimization - Training-specific step
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()
        optimizer.zero_grad()

        # Get predictions and calculate metrics
        gt = batch['y']
        model_with_losses = model.module if hasattr(model, "module") else model
        preds = model_with_losses.get_pr(output)

        # Calculate statistics and update loggers
        stats = model_with_losses.get_stats(gt, preds, train=True)  # Training-specific metrics
        lr = optimizer.param_groups[0]["lr"]
        losses_detach = {f"train_{k}": v.cpu().item() for k, v in all_losses.items()}
        
        # Update metric logger
        metric_logger.update(lr=lr)
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(**losses_detach)
        metric_logger.update(**stats)
        
        # Log to wandb if enabled
        if not args.disable_wandb and misc.is_main_process():
            wandb.log(
                {
                    "train_loss": total_loss.item(),
                    "lr": lr,
                    **losses_detach,
                    **stats,
                }
            )
            
        # Store batch results
        lists["gt"].append(batch['y'].cpu())
        lists["pr"].append(preds.cpu())
        model_with_losses = model.module if hasattr(model, "module") else model
        lists["x"].extend(model_with_losses.batch_decode(batch))

    # Concatenate all predictions and ground truth
    gt_all = torch.cat(lists["gt"])
    prob_all = torch.cat(lists["pr"])

    # Save epoch predictions
    torch.save(
        {
            "gt": gt_all,       
            "pr": prob_all,
            "x": lists["x"]
        },
        args.output_dir / "train_preds.pth",
    )
    
    # Synchronize metrics across processes and return averaged stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


######################################################################
# evaluate the model on a validation/test dataset
######################################################################

@torch.no_grad()  # Disables gradient computation for evaluation
def evaluate(model, dl, device, args, output_dir):
    """
    Evaluates the model on a validation/test dataset.
    
    Args:
        model: The neural network model to evaluate
        dl: DataLoader containing the evaluation data
        device: The device (CPU/GPU) to use for evaluation
        args: Additional arguments for evaluation configuration
        output_dir: Directory to save evaluation results and plots
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()  # Disables training-specific behaviors like dropout

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # Lists to store predictions, ground truth, and losses
    lists = {"gt": [], "pr": [], "x": [], "loss": []}
    
    # Evaluation loop - no gradient computation or parameter updates
    for batch in metric_logger.log_every(dl, 10, header):
        batch = move_to_device(batch, device)
        output = model(batch['x'])
        
        # Calculate losses (for monitoring only, no backprop)
        all_losses = {}
        model_with_losses = model.module if hasattr(model, "module") else model
        for loss_name in model_with_losses.losses:
            losses = loss_dict[loss_name](output, batch)
            all_losses.update(losses)
            
        # Store individual losses
        for loss_name, loss_val in all_losses.items():
            if loss_name not in lists.keys():
                lists[loss_name] = []
            lists[loss_name].append(loss_val.cpu())
            
        # Get predictions
        preds = model_with_losses.get_pr(output)
        lists["gt"].append(batch['y'].cpu())
        lists["pr"].append(preds.cpu())
        lists["x"].extend(model_with_losses.batch_decode(batch))

        total_loss = sum(all_losses.values())
        lists['loss'].append(total_loss.cpu())

    # Process all predictions and calculate metrics
    gt_all = torch.cat(lists["gt"])
    prob_all = torch.cat(lists["pr"])
    mean_loss = float(np.mean(lists['loss']))

    model_with_losses = model.module if hasattr(model, "module") else model
    stats = model_with_losses.get_stats(gt_all, prob_all, train=False)  # Testing-specific metrics

    # Calculate average losses
    for loss_name, loss_val in all_losses.items():
        stats[f'test_{loss_name}'] = float(np.mean(lists[loss_name]))
    stats['test_loss'] = mean_loss

    # Create plots directory
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Generate and save ROC curves
    plt.figure(figsize=(3, 3))
    gt_onehot = np.eye(3)[gt_all.cpu()]  # Convert to one-hot encoding for 3-class classification
    pr_np = prob_all.cpu().numpy()
    
    # Plot ROC curve for each class
    for i in range(3):
        fpr, tpr, _ = roc_curve(gt_onehot[:, i], pr_np[:, i])
        class_names = ['Immunogenic', 'Non-immunogenic', 'Weakly immunogenic']
        colors = ['#4A4A4A', '#8B0000', '#00008B']  # Dark grey, dark red, dark blue
        plt.plot(fpr, tpr, color=colors[i],label=f'{class_names[i]} (AUC = {stats[f"test_auroc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Add diagonal line for reference
    plt.xlabel('False Positive Rate', fontsize=10, fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=10, fontname='Arial')
    plt.title(f'ROC Curves (Epoch {getattr(args, "current_epoch", "final")})', fontsize=8, fontname='Arial')
    plt.legend(prop={'family': 'Arial', 'size': 7})
    plt.tick_params(axis='both', which='major', labelsize=9)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    plt.savefig(plots_dir / f'roc_curve_epoch_{getattr(args, "current_epoch", "final")}.pdf')
    plt.close()

    # Create and save Precision-Recall curves
    plt.figure(figsize=(3, 3))
    
    # Calculate mean precision and recall across all classes
    mean_precision = np.zeros_like(pr_np[:, 0])
    mean_recall = np.linspace(0, 1, len(mean_precision))
    
    for i in range(3):
        precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
        class_names = ['Immunogenic', 'Non-immunogenic', 'Weakly immunogenic']
        colors = ['#4A4A4A', '#8B0000', '#00008B']  # Dark grey, dark red, dark blue
        plt.plot(recall, precision, color=colors[i], label=f'{class_names[i]} (AUC = {stats[f"test_auprc_class{i}"]:.2f})')
        
        # Interpolate precision values for mean calculation
        mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])
    
    # Calculate and plot mean precision
    mean_precision /= 3
    mean_auprc = np.mean([stats[f"test_auprc_class{i}"] for i in range(3)])
    plt.plot(mean_recall, mean_precision, color='black', linestyle='--', 
             label=f'Mean Average Precision (AUC = {mean_auprc:.2f})')
    
    plt.xlabel('Recall', fontsize=10, fontname='Arial')
    plt.ylabel('Precision', fontsize=10, fontname='Arial')
    plt.title(f'Precision-Recall Curves (Epoch {getattr(args, "current_epoch", "final")})', fontsize=8, fontname='Arial')
    plt.legend(prop={'family': 'Arial', 'size': 7})
    plt.tick_params(axis='both', which='major', labelsize=9)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    plt.savefig(plots_dir / f'pr_curve_epoch_{getattr(args, "current_epoch", "final")}.pdf')
    plt.close()

    # Save evaluation metrics to CSV
    metrics = {
        'epoch': getattr(args, "current_epoch", "final"),
        'auroc': stats["test_auroc"],
#        'auprc_immunogenic': stats["test_auprc_class0"],
#        'auprc_non_immunogenic': stats["test_auprc_class1"],
#        'auprc_weakly_immunogenic': stats["test_auprc_class2"],
        'auprc_class0': stats["test_auprc_class0"],
        'auprc_class1': stats["test_auprc_class1"],
        'auprc_class2': stats["test_auprc_class2"],
        'accuracy': stats["test_acc"],
        'f1_macro': stats["test_f1_macro"],
        'f1_weighted': stats["test_f1_weighted"],
        'loss': stats["test_loss"]
    }
    
    # Load existing metrics if available, otherwise create new DataFrame
    metrics_file = plots_dir / 'test_metrics.csv'
    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file)
        df_new = pd.DataFrame([metrics])
        df_metrics = pd.concat([df_metrics, df_new], ignore_index=True)
    else:
        df_metrics = pd.DataFrame([metrics])
    
    df_metrics.to_csv(metrics_file, index=False)

    # Create and save training progress visualization
    plt.figure(figsize=(15, 10))
    
    # Convert epoch column to numeric, replacing 'final' with the last numeric value + 1
    numeric_epochs = pd.to_numeric(df_metrics['epoch'].replace('final', float('inf')), errors='coerce')
    if float('inf') in numeric_epochs.values:
        last_numeric = numeric_epochs[numeric_epochs != float('inf')].max()
        numeric_epochs = numeric_epochs.replace(float('inf'), last_numeric + 1 if not pd.isna(last_numeric) else 0)
    
    # Plot 1: AUROC over epochs
    plt.subplot(2, 2, 1)
    plt.plot(numeric_epochs, df_metrics['auroc'], marker='o', color='#9370DB')  # Using a purple color
    plt.title('AUROC over epochs', fontsize=7, fontname='Arial')
    plt.xlabel('Epoch', fontsize=8, fontname='Arial')
    plt.ylabel('AUROC', fontsize=8, fontname='Arial')
    plt.tick_params(axis='both', which='major', labelsize=7)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    plt.grid(True)

    # Plot 2: AUPRC for each class over epochs
    plt.subplot(2, 2, 2)
    purples = plt.cm.Purples(np.linspace(0.4, 0.9, 3))  # Get 3 shades of purple
    for i in range(3):
        plt.plot(numeric_epochs, df_metrics[f'auprc_class{i}'], marker='o', 
                color=purples[i], label=f'Class {i}')
    plt.title('AUPRC over epochs', fontsize=7, fontname='Arial')
    plt.xlabel('Epoch', fontsize=8, fontname='Arial')
    plt.ylabel('AUPRC', fontsize=8, fontname='Arial')
    plt.legend(prop={'family': 'Arial', 'size': 7})
    plt.tick_params(axis='both', which='major', labelsize=7)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    plt.grid(True)

    # Plot 3: Accuracy and F1 scores over epochs
    plt.subplot(2, 2, 3)
    purples = plt.cm.Purples(np.linspace(0.4, 0.9, 3))  # Get 3 shades of purple
    plt.plot(numeric_epochs, df_metrics['accuracy'], marker='o', color=purples[0], label='Accuracy')
    plt.plot(numeric_epochs, df_metrics['f1_macro'], marker='o', color=purples[1], label='F1 Macro')
    plt.plot(numeric_epochs, df_metrics['f1_weighted'], marker='o', color=purples[2], label='F1 Weighted')
    plt.title('Accuracy and F1 Scores over epochs', fontsize=7, fontname='Arial')
    plt.xlabel('Epoch', fontsize=8, fontname='Arial')
    plt.ylabel('Score', fontsize=8, fontname='Arial')
    plt.legend(prop={'family': 'Arial', 'size': 7})
    plt.tick_params(axis='both', which='major', labelsize=7)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    plt.grid(True)

    # Plot 4: Loss over epochs
    plt.subplot(2, 2, 4)
    plt.plot(numeric_epochs, df_metrics['loss'], marker='o', color='#9370DB')  # Using a purple color
    plt.title('Loss over epochs', fontsize=7, fontname='Arial')
    plt.xlabel('Epoch', fontsize=8, fontname='Arial')
    plt.ylabel('Loss', fontsize=8, fontname='Arial')
    plt.tick_params(axis='both', which='major', labelsize=7)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plots_dir / 'test_progress.pdf')
    plt.close()

    # Save predictions for later analysis
    torch.save(
        {
            "gt": gt_all,       # Ground truth labels
            "pr": prob_all,     # Model predictions
            "x": lists["x"]     # Input data
        },
        output_dir / "test_preds.pth",
    )

    # Log dataset name and metrics
    ds_name = dl.dataset.name if hasattr(dl.dataset, 'name') else 'test'
    print(ds_name, stats)

    # Update and return metrics
    metric_logger.update(**stats)
    ret = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return ret
