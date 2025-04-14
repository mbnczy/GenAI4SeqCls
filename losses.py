import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', smoothing=0.1):
        """
        Combines Focal Loss with Label Smoothing and supports class weights.

        Args:
            alpha (Tensor or float): Class weights tensor of shape [num_classes].
            gamma (float): Focusing parameter for hard examples.
            reduction (str): 'none', 'mean', or 'sum'.
            smoothing (float): Smoothing factor for label smoothing.
        """
        super(FocalLossWithLabelSmoothing, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha  # Tensor of shape [num_classes] or None
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        device = logits.device

        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute probabilities
        probs = torch.exp(log_probs)

        # Compute focal weights per class
        focal_weight = (1 - probs) ** self.gamma

        # Compute the loss per class
        loss = -true_dist * focal_weight * log_probs

        # Apply alpha (class weights)
        if self.alpha is not None:
            if self.alpha.device != device:
                self.alpha = self.alpha.to(device)
            alpha_factor = self.alpha.unsqueeze(0)  # Shape [1, num_classes]
            loss = alpha_factor * loss

        # Sum over classes
        loss = loss.sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss