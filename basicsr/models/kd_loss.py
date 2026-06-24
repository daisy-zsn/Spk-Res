import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Knowledge Distillation Loss.

    Supports output-level distillation via soft targets (KL divergence)
    and can be extended to feature-level distillation.

    Reference:
        Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014.

    Args:
        temperature (float): Temperature for softening probability distributions.
                            Higher values produce softer distributions.
                            Default: 4.0
        alpha (float): Weight for KD loss vs standard loss. Default: 1.0 (only KD).
    """

    def __init__(self, temperature=4.0, alpha=1.0):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_output, teacher_output):
        """Compute KD loss between student and teacher outputs.

        Uses MSE between softened outputs as a simple and numerically stable
        alternative to KL divergence, suitable for regression tasks
        (signal reconstruction).

        Args:
            student_output (Tensor): Student model predictions.
            teacher_output (Tensor): Teacher model predictions (detached).

        Returns:
            torch.Tensor: Scalar KD loss.
        """
        # Soften the outputs using temperature scaling
        # For regression tasks, we use MSE on temperature-scaled outputs
        # which is equivalent to matching the softened distribution
        student_soft = student_output / self.temperature
        teacher_soft = teacher_output / self.temperature

        # MSE-based KD loss (simpler and more stable than KL for regression)
        kd_loss = F.mse_loss(student_soft, teacher_soft)

        return kd_loss * (self.temperature ** 2) * self.alpha


class FeatureKDLoss(nn.Module):
    """Feature-level Knowledge Distillation Loss.

    Aligns intermediate feature representations between teacher and student
    using MSE loss. This is useful for preserving the feature extraction
    capability learned during pretraining.

    Args:
        loss_weight (float): Weight for this feature KD loss.
    """

    def __init__(self, loss_weight=1.0):
        super(FeatureKDLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, student_features, teacher_features):
        """Compute MSE between student and teacher features.

        Args:
            student_features (Tensor): Student intermediate features.
            teacher_features (Tensor): Teacher intermediate features (detached).

        Returns:
            torch.Tensor: Scalar feature KD loss.
        """
        return self.loss_weight * F.mse_loss(student_features, teacher_features)