import torch
from torch import nn
import torch.nn.functional as F


class LookBackController(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        sie_module: nn.Module,
        fusion_module: nn.Module,
        tau_unc: float = 0.4,
        max_lookback: int = 3,
    ):
        super().__init__()
        if embed_dim < 2:
            raise ValueError('embed_dim must be at least 2')
        if not 0 <= tau_unc:
            raise ValueError('tau_unc must be non-negative')
        if max_lookback < 1:
            raise ValueError('max_lookback must be at least 1')

        hidden_dim = max(1, embed_dim // 2)
        self.tau_unc = tau_unc
        self.max_lookback = max_lookback
        self.sie_module = sie_module
        self.fusion_module = fusion_module
        self.visual_calibration = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.text_calibration = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        if lengths is None:
            return values.mean(dim=1)
        time_indices = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        valid = (time_indices < lengths.to(values.device).unsqueeze(1)).to(values.dtype)
        return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def _uncertainty(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        disagreement = 1.0 - F.cosine_similarity(visual_features, text_features, dim=-1, eps=1e-6)
        return self._masked_mean(disagreement, lengths)

    def forward(
        self,
        original_visual: torch.Tensor,
        original_text: torch.Tensor,
        audio_features: torch.Tensor,
        aligned_visual: torch.Tensor,
        aligned_text: torch.Tensor,
        joint_features: torch.Tensor,
        lengths: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        current_visual = original_visual
        current_text = original_text
        current_joint = joint_features
        current_aligned_visual = aligned_visual
        current_aligned_text = aligned_text
        fallback_count = torch.zeros(
            original_visual.shape[0], dtype=torch.long, device=original_visual.device
        )

        for _ in range(self.max_lookback):
            uncertainty = self._uncertainty(
                current_aligned_visual, current_aligned_text, lengths
            )
            fallback_mask = uncertainty > self.tau_unc
            if not torch.any(fallback_mask):
                break

            visual_gate = self.visual_calibration(current_joint)
            text_gate = self.text_calibration(current_joint)
            updated_visual = current_visual * visual_gate
            updated_text = current_text * text_gate
            next_visual, next_text = self.sie_module(
                updated_visual, updated_text, audio_features
            )
            next_aligned_visual, next_aligned_text, next_joint = self.fusion_module(
                next_visual, next_text, key_padding_mask
            )

            sample_mask = fallback_mask.view(-1, 1, 1)
            current_visual = torch.where(sample_mask, updated_visual, current_visual)
            current_text = torch.where(sample_mask, updated_text, current_text)
            current_aligned_visual = torch.where(
                sample_mask, next_aligned_visual, current_aligned_visual
            )
            current_aligned_text = torch.where(
                sample_mask, next_aligned_text, current_aligned_text
            )
            current_joint = torch.where(sample_mask, next_joint, current_joint)
            fallback_count = fallback_count + fallback_mask.long()

        uncertainty = self._uncertainty(
            current_aligned_visual, current_aligned_text, lengths
        )
        return current_joint, uncertainty, fallback_count
