"""
This file contains the implementation of the attention module.

Reference: https://github.com/meta-llama/llama3/blob/main/llama/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, asdict, field
from omegaconf import OmegaConf
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange, repeat
from .mlp import ConditionedNorm
from .utils.dataclass import shallow_asdict


############
# Config
############
@dataclass
class AttentionConfig:
    num_heads: int = 8  # Number of attention heads (for multi-head attention)
    num_kv_heads: int = (
        8  # Number of attention heads for Key and Value (Grouped Query Attention)
    )
    use_conditional_norm: bool = False  # Whether to use time conditional normalization
    cond_norm_hidden_size: int = 4  # Hidden size for the time conditional normalization
    atten_dropout: float = 0.0  # Dropout probability in the attention module


@dataclass
class TransformerConfig:
    patch_size: int = 8  # Size of the patches for the structured latent tokens
    hidden_size: int = 256  # Hidden size of the transformer
    use_attn_norm: bool = True  # Whether to use normalization in the attention module
    use_ffn_norm: bool = True  # Whether to use normalization in the feedforward network
    norm_eps: float = 1e-6  # Epsilon value for layer normalization
    num_layers: int = 3  # Number of transformer blocks
    positional_embedding: str = "absolute"  # Positional embedding type, supports ['absolute', 'rope', 'continuous_rope', 'relative_bias', 'learnable']
    use_long_range_skip: bool = True  # Set it to True for UViT processor
    ffn_multiplier: int = (
        4  # FFN hidden size multiplier (ffn_hidden = hidden_size * ffn_multiplier)
    )
    attn_config: AttentionConfig = field(
        default_factory=AttentionConfig
    )  # Configuration for the attention sub-module


############
# Attention
############


class RelativeBias(nn.Module):
    def __init__(self, num_heads: int, hidden_size: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_scaling = nn.Parameter(torch.ones(num_heads), requires_grad=True)

        # Small MLP to map a scalar distance to a bias value per head
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size), nn.GELU(), nn.Linear(hidden_size, num_heads)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords: torch.Tensor, shape (batch, seq_len, coord_dim)
            The continuous coordinates (e.g., x, y)

        Returns
        -------
        bias: torch.Tensor, shape (batch, num_heads, seq_len, seq_len)
            The additive bias for the attention matrix
        """
        # 1. Compute pairwise Euclidean distances: [N, N]
        dist_mat = torch.cdist(coords, coords, p=2)

        # 2. Add feature dimension for the MLP: [N, N, 1]
        dist_mat = dist_mat.unsqueeze(-1)

        # 3. Project distance to head-specific biases: [N, N, num_heads]
        bias = self.mlp(dist_mat)

        # 4. Scale biases per head
        bias = bias * self.head_scaling

        # 5. [B, H, N, N]
        bias = bias.permute(2, 0, 1)

        return bias


class ContinuousRoPE(nn.Module):
    def __init__(self, dim, theta=10000):
        """
        dim: head_dim (total dimension per head)
        theta: base frequency (default 10000)
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

        # We split the head_dim into two halves (one for X, one for Y)
        # Each axis rotation needs dim // 2. Inside that, we take every 2nd
        # for the frequency bands (Standard RoPE math).
        half_dim = dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def rotate_half(self, x):
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    def forward(self, t, coords):
        """
        t: [batch, heads, seq_len, head_dim]
        coords: [batch, seq_len, 2] -> continuous (x, y)
        """
        # Split Q/K into two halves for X and Y rotations
        # t_x: [B, H, N, D/2], t_y: [B, H, N, D/2]
        t_x, t_y = t.chunk(2, dim=-1)

        # Calculate frequencies for X and Y coordinates
        # coords[..., 0:1] is [B, N, 1]. self.inv_freq is [D/4]
        # freqs_x/y shape: [B, 1, N, D/2] after repeat
        def get_freqs(c):
            # [N, D/4]
            freqs = torch.einsum("n, f -> n f", c, self.inv_freq)
            # repeat for sin/cos pairs -> [N, D/2]
            freqs = repeat(freqs, "n d -> n (d r)", r=2)
            return rearrange(freqs, "n d -> 1 n d")

        freqs_x = get_freqs(coords[..., 0])
        freqs_y = get_freqs(coords[..., 1])

        # Apply rotation to each half
        # x_rotated = x*cos + rotate_half(x)*sin
        t_x_rotated = (t_x * freqs_x.cos()) + (self.rotate_half(t_x) * freqs_x.sin())
        t_y_rotated = (t_y * freqs_y.cos()) + (self.rotate_half(t_y) * freqs_y.sin())

        # Recombine into [batch, heads, seq_len, head_dim]
        return torch.cat([t_x_rotated, t_y_rotated], dim=-1)


class GroupQueryFlashAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        use_conditional_norm: bool = False,
        cond_norm_hidden_size: int = 4,
        atten_dropout: float = 0.0,
        positional_embedding: str = "absolute",  # Positional embedding type, supports ['absolute', 'rope', 'continuous_rope', 'relative_bias', 'learnable']
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_repeat = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.atten_dropout = atten_dropout

        kv_hidden_size = self.head_dim * self.num_kv_heads

        self.q_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(
            hidden_size, input_size, bias=False
        )  # output back to input_size

        if use_conditional_norm:
            self.correction = ConditionedNorm(1, input_size, cond_norm_hidden_size)
        else:
            self.correction = None

        self.positional_embedding_type = positional_embedding
        if positional_embedding == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        elif positional_embedding == "continuous_rope":
            self.rotary_emb = ContinuousRoPE(dim=self.head_dim)
        elif positional_embedding == "relative_bias":
            self.relative_bias = RelativeBias(num_heads=num_heads, hidden_size=16)

    def forward(
        self,
        x,
        condition: Optional[float] = None,
        relative_positions: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        x: torch.Tensor, shape (..., seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (..., seq_len, input_size)
        """

        if self.correction is not None:
            x = self.correction(c=condition, x=x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size, seq_len, _ = q.size()

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_repeat, dim=1)
            v = v.repeat_interleave(self.num_repeat, dim=1)

        if relative_positions is not None and self.positional_embedding_type == "rope":
            # TODO: This is the point where we might have to add rope based on continoous relative coordinates
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        if (
            relative_positions is not None
            and self.positional_embedding_type == "continuous_rope"
        ):
            q = self.rotary_emb(q, relative_positions)
            k = self.rotary_emb(k, relative_positions)

        bias_mask = None
        if (
            relative_positions is not None
            and self.positional_embedding_type == "relative_bias"
        ):
            bias_mask = self.relative_bias(relative_positions)

        if self.training:
            dp = self.atten_dropout
        else:
            dp = 0.0
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias_mask,
            dropout_p=dp,
        )

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.o_proj(x)

        return x

    @classmethod
    def from_config(
        cls,
        input_size: int,
        hidden_size: int,
        config: AttentionConfig,
        positional_embedding: str = "absolute",
    ):
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            positional_embedding=positional_embedding,
            **shallow_asdict(config),
        )


############
# Feedforward Network
############
class FFN(nn.Module):
    def __init__(
        self,
        input_size: int,
        ffn_hidden_size: int,  # Directly specify FFN hidden size
        use_conditional_norm: bool = False,
        cond_norm_hidden_size: int = 4,
    ):
        super().__init__()
        self.w1 = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.w2 = nn.Linear(ffn_hidden_size, input_size, bias=False)
        self.w3 = nn.Linear(input_size, ffn_hidden_size, bias=False)

        if use_conditional_norm:
            self.correction = ConditionedNorm(1, input_size, cond_norm_hidden_size)
        else:
            self.correction = None

    def forward(self, x, condition: Optional[float] = None):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))

        if self.correction is not None:
            x = self.correction(c=condition, x=x)

        return x


############
# Normalization
############
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


############
# Transformer Block
############
class TransformerBlock(nn.Module):
    def __init__(
        self, input_size: int, config: TransformerConfig, skip_connection: bool = False
    ):
        super().__init__()
        hidden_size = config.hidden_size
        ffn_hidden_size = hidden_size * config.ffn_multiplier

        self.attn = GroupQueryFlashAttention.from_config(
            input_size=input_size,
            hidden_size=hidden_size,
            config=config.attn_config,
            positional_embedding=config.positional_embedding,
        )

        self.ffn = FFN(
            input_size=input_size,
            ffn_hidden_size=ffn_hidden_size,
            use_conditional_norm=config.attn_config.use_conditional_norm,
            cond_norm_hidden_size=config.attn_config.cond_norm_hidden_size,
        )

        self.attn_norm = (
            RMSNorm(input_size, eps=config.norm_eps) if config.use_attn_norm else None
        )
        self.ffn_norm = (
            RMSNorm(input_size, eps=config.norm_eps) if config.use_ffn_norm else None
        )

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = nn.Linear(input_size * 2, input_size)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[float] = None,
        relative_positions: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor, shape (..., seq_len, input_size)
        condition: Optional[float]

        Returns
        -------
        torch.Tensor, shape (..., seq_len, input_size)
        """
        if self.skip_connection and skip is not None:
            x = torch.cat([x, skip], dim=-1)
            x = self.skip_proj(x)

        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h, condition=condition, relative_positions=relative_positions)
        h = h if self.ffn_norm is None else self.ffn_norm(h)
        out = h + self.ffn(h, condition=condition)
        return out


############
# Transformer
############
class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: TransformerConfig = TransformerConfig(),
    ):
        super().__init__()
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        self.use_long_range_skip = config.use_long_range_skip

        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
            working_size = hidden_size
        else:
            self.input_proj = nn.Identity()
            working_size = input_size

        if working_size != output_size:
            self.output_proj = nn.Linear(working_size, output_size)
        else:
            self.output_proj = nn.Identity()

        num_encoder_layers = num_layers // 2
        num_decoder_layers = num_layers // 2
        middle_layer_exists = num_layers % 2 == 1

        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_size=working_size, config=config, skip_connection=False
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.middle_layer = None
        if middle_layer_exists:
            self.middle_layer = TransformerBlock(
                input_size=working_size, config=config, skip_connection=False
            )

        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_size=working_size, config=config, skip_connection=True
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[float] = None,
        relative_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            [..., seq_len, input_size]

        Returns
        -------
        torch.Tensor
            [..., seq_len, output_size]
        """
        x = self.input_proj(x)
        skips = []

        for layer in self.encoder_layers:
            x = layer(x, condition=condition, relative_positions=relative_positions)
            skips.append(x)

        if self.middle_layer is not None:
            x = self.middle_layer(
                x, condition=condition, relative_positions=relative_positions
            )

        for layer in self.decoder_layers:
            skip = skips.pop() if self.use_long_range_skip else None
            x = layer(
                x, condition=condition, relative_positions=relative_positions, skip=skip
            )

        x = self.output_proj(x)
        return x
