import torch
from torch import nn


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive')
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.visual_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_visual = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.syn_norm = nn.LayerNorm(embed_dim)
        self.visual_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        self.syn_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ):
        if visual_features.ndim != 3 or text_features.ndim != 3:
            raise ValueError('fusion inputs must have shape [B, L, D]')
        if visual_features.shape != text_features.shape:
            raise ValueError('visual and text features must have identical shapes')
        if key_padding_mask is not None:
            if key_padding_mask.shape != visual_features.shape[:2]:
                raise ValueError('key_padding_mask must have shape [B, L]')
            key_padding_mask = key_padding_mask.to(dtype=torch.bool, device=visual_features.device)

        visual_context, _ = self.visual_to_text(
            query=visual_features,
            key=text_features,
            value=text_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        text_context, _ = self.text_to_visual(
            query=text_features,
            key=visual_features,
            value=visual_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        aligned_visual = self.visual_norm(
            visual_features + self.visual_projection(visual_context)
        )
        aligned_text = self.text_norm(
            text_features + self.text_projection(text_context)
        )
        joint_features = self.syn_projection(
            torch.cat((aligned_visual, aligned_text), dim=-1)
        )
        joint_features = self.syn_norm(joint_features + aligned_visual + aligned_text)
        return aligned_visual, aligned_text, joint_features
