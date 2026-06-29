"""
VAE (Variational Autoencoder) for MEA Signal Denoising/Reconstruction.

Architecture (Spatial Convolutional VAE):
    Encoder: Conv2d → ResBlocks → parallel Conv2d heads → spatial μ_map / σ_map
             → reparameterize → spatial z_map (preserves spatial locality)
    Decoder: z_map → fuse Conv2d → ResBlocks → Conv2d output + input (global residual)

Key design: SPATIAL latent space — each spatial position (channel×time) gets its own
latent vector, preserving local structure. NO global pooling — avoids posterior collapse.

Designed to serve as a deep learning baseline for comparison with Spk-Res.
Compatible with the existing ReconstructionModel training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs import common


def make_model(args, parent=False):
    return VAESR(args)


class VAESR(nn.Module):
    """Spatial Convolutional VAE for Signal Restoration.

    Unlike standard VAE which uses global pooling (compressing the entire
    feature map into a 1D vector), this architecture maintains the spatial
    structure throughout the latent space. Each (channel, time) position
    gets its own latent vector, allowing the model to learn local denoising
    patterns without the severe information bottleneck that causes
    posterior collapse.

    Args:
        num_channels (int): Input/output channels.
        factor (int): Scale factor (1 = same size).
        width (int): Number of feature channels in hidden layers.
        depth (int): Number of residual blocks (split between encoder/decoder).
        kernel_size (int): Convolution kernel size.
        latent_dim (int): Number of latent channels (= dimension per spatial position).
        beta (float): Weight for the KL divergence term (β-VAE).
        kl_anneal_start (float): Starting KL weight multiplier for annealing.
            Linearly increases from kl_anneal_start to 1.0 over training.
            Default 0.1 means KL starts at 10% weight. Used to prevent
            posterior collapse in early training. Set beta=1.0 and
            kl_anneal_start=1.0 to disable annealing.
    """

    def __init__(
        self,
        num_channels=1,
        width=64,
        depth=8,
        kernel_size=3,
        latent_dim=32,
        beta=1.0,
        kl_anneal_start=0.1,
    ):
        super(VAESR, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.width = width
        self.kl_anneal_start = kl_anneal_start
        n_feats = width
        conv = common.default_conv
        act = nn.ReLU(True)

        # ---- Encoder (preserves spatial resolution) ----
        enc_depth = depth // 2

        self.enc_head = conv(num_channels, n_feats, kernel_size)

        enc_body = []
        for _ in range(enc_depth):
            enc_body.append(
                common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1.0)
            )
        self.enc_body = nn.Sequential(*enc_body)

        # μ and log(σ²) as SPATIAL feature maps (same resolution as input)
        # Each spatial position gets its own latent distribution parameters
        self.conv_mu = conv(n_feats, latent_dim, kernel_size)
        self.conv_logvar = conv(n_feats, latent_dim, kernel_size)

        # ---- Decoder ----
        dec_depth = depth - enc_depth

        # Fuse: map spatial latent back to feature space
        self.dec_fuse = conv(latent_dim, n_feats, kernel_size)

        dec_body = []
        for _ in range(dec_depth):
            dec_body.append(
                common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1.0)
            )
        self.dec_body = nn.Sequential(*dec_body)

        self.dec_tail = conv(n_feats, num_channels, kernel_size)

        # KL loss (set during forward, read by training pipeline)
        self.kl_loss = None
        # Annealing factor (set externally by training loop if supported)
        self._kl_scale = 1.0

    def encode(self, x):
        """Encode input to spatial latent distribution parameters.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            mu: (B, latent_dim, H, W) spatial mean map.
            logvar: (B, latent_dim, H, W) spatial log-variance map.
        """
        h = self.enc_head(x)                    # (B, n_feats, H, W)
        h = self.enc_body(h)                     # (B, n_feats, H, W)
        mu = self.conv_mu(h)                     # (B, latent_dim, H, W)
        logvar = self.conv_logvar(h)             # (B, latent_dim, H, W)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε.

        Args:
            mu: (B, latent_dim, H, W) spatial mean.
            logvar: (B, latent_dim, H, W) spatial log-variance.

        Returns:
            z: (B, latent_dim, H, W) spatial latent.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode spatial latent to reconstructed signal.

        Args:
            z: (B, latent_dim, H, W) spatial latent.

        Returns:
            out: (B, C, H, W) reconstructed residual.
        """
        h = self.dec_fuse(z)                     # (B, n_feats, H, W)
        h = self.dec_body(h)                      # (B, n_feats, H, W)
        out = self.dec_tail(h)                    # (B, C, H, W)
        return out

    def forward(self, x):
        """Forward pass with global residual connection.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            out: (B, C, H, W) reconstructed tensor.
        """
        # Encode
        mu, logvar = self.encode(x)

        # KL divergence loss (per spatial position, summed over latent
        # channels, then averaged over batch and spatial positions).
        # KL(N(μ,σ²) || N(0,1)) = 0.5 * (σ² + μ² - 1 - ln(σ²))
        kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Sum over latent channels, mean over batch and spatial positions
        kl_per_position = kl_elementwise.sum(dim=1)  # (B, H, W)
        kl_batch = kl_per_position.mean()            # scalar
        self.kl_loss = self.beta * self._kl_scale * kl_batch

        # Reparameterize
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            # During inference, use μ directly (deterministic)
            z = mu

        # Decode
        out = self.decode(z)

        # Global residual connection
        out = out + x

        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('dec_tail') == -1 and name.find('enc_head') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size())
                        )
            elif strict:
                if name.find('dec_tail') == -1 and name.find('enc_head') == -1:
                    raise KeyError(
                        'unexpected key "{}" in state_dict'.format(name)
                    )

    def set_kl_scale(self, scale):
        """Update KL annealing scale (call from training loop).

        Args:
            scale (float): Current KL weight multiplier in [kl_anneal_start, 1.0].
        """
        self._kl_scale = max(0.0, min(1.0, scale))
