from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

class CrossEntropyLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, batch):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        labels = batch['y']
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        loss = F.cross_entropy(logits, labels, reduction='mean')

        return {"ce": loss}