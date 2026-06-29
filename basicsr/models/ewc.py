import torch


class EWC:
    """Elastic Weight Consolidation (EWC).

    Computes diagonal Fisher Information Matrix to measure parameter importance
    on the pretrained task, then adds a quadratic penalty against deviating
    from pretrained weights, weighted by importance.

    Reference:
        Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017.

    Args:
        model (nn.Module): The pretrained model (already loaded with pretrained weights).
        lambda_ewc (float): Global scaling factor for EWC penalty.
        num_samples (int): Number of data samples used to estimate Fisher matrix.
                           If 0, falls back to uniform importance (= L2-SP).
    """

    def __init__(self, model, lambda_ewc, num_samples=0):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.old_params = {}

        # Save pretrained parameter snapshot
        for name, param in model.named_parameters():
            self.old_params[name] = param.detach().clone()

        if num_samples > 0:
            # Estimate Fisher diagonals from a batch
            self._estimate_fisher(model, num_samples)
        else:
            # Fallback: use uniform importance (same as L2-SP)
            for name, param in model.named_parameters():
                self.fisher[name] = torch.ones_like(param)

    def _estimate_fisher(self, model, num_samples):
        """Estimate diagonal of Fisher Information Matrix using empirical Fisher.

        Here we use a simplified version where Fisher diagonal is approximated
        by the squared gradient of the MSE loss on pretrained data.

        For a more complete implementation, you would iterate over a small subset
        of the pretraining dataset. Here we initialize with ones as a fallback.
        """
        for name, param in model.named_parameters():
            # Uniform importance as default
            self.fisher[name] = torch.ones_like(param)

    def penalty(self, model):
        """Compute EWC penalty: sum_i (F_i * (theta_i - theta_i*)^2).

        Args:
            model (nn.Module): Current model during training.

        Returns:
            torch.Tensor: Scalar EWC loss.
        """
        loss = 0.
        for name, param in model.named_parameters():
            if name in self.old_params and name in self.fisher:
                _fisher = self.fisher[name].to(param.device)
                _old = self.old_params[name].to(param.device)
                loss += torch.sum(_fisher * (param - _old) ** 2)
        return self.lambda_ewc * loss

    def update_fisher(self, model, dataloader, loss_fn, device, num_batches=10):
        """Update Fisher diagonals using a small number of batches.

        This should be called on the pretraining data before fine-tuning starts.

        Args:
            model (nn.Module): Pretrained model.
            dataloader (DataLoader): Pretraining data loader.
            loss_fn (callable): Loss function (e.g., MSELoss).
            device (torch.device): Device.
            num_batches (int): Number of batches to use for estimation.
        """
        # Reset fisher
        for name in self.fisher:
            self.fisher[name].zero_()

        model.eval()
        batch_count = 0
        for data in dataloader:
            if batch_count >= num_batches:
                break
            model.zero_grad()
            lq = data['lq'].to(device)
            gt = data['gt'].to(device)
            output = model(lq)
            if isinstance(output, list):
                output = output[-1]
            loss = loss_fn(output, gt)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2 / num_batches

            batch_count += 1

        # Clip and normalize
        for name in self.fisher:
            self.fisher[name] = torch.clamp(self.fisher[name], max=1e3)
            self.fisher[name] = self.fisher[name] / (self.fisher[name].max() + 1e-8)