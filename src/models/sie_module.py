import torch
from torch import nn
import torch.nn.functional as F


class SIEModule(nn.Module):
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        window_radius: int = 2,
        topk_ratio: float = 0.17,
        sigma: float = 3.0,
        alpha_init: float = 0.5,
        epsilon_audio_init: float = 0.1,
        lambda_audio_init: float = 0.1,
    ):
        super().__init__()
        if window_radius < 0:
            raise ValueError('window_radius must be non-negative')
        if not 0 < topk_ratio <= 1:
            raise ValueError('topk_ratio must be in (0, 1]')
        if sigma <= 0:
            raise ValueError('sigma must be positive')

        self.similarity_threshold = similarity_threshold
        self.window_radius = window_radius
        self.topk_ratio = topk_ratio
        self.sigma = sigma
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.epsilon_audio = nn.Parameter(torch.tensor(float(epsilon_audio_init)))
        self.lambda_audio = nn.Parameter(torch.tensor(float(lambda_audio_init)))

    def _local_mean(self, energy: torch.Tensor) -> torch.Tensor:
        if self.window_radius == 0:
            return energy
        kernel_size = 2 * self.window_radius + 1
        padded = F.pad(energy.unsqueeze(1), (self.window_radius, self.window_radius), mode='replicate')
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).squeeze(1)

    @staticmethod
    def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(left, right, dim=-1, eps=1e-6)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
    ):
        if visual_features.ndim != 3 or text_features.ndim != 3 or audio_features.ndim != 3:
            raise ValueError('SIE inputs must have shape [B, L, D]')
        if visual_features.shape != text_features.shape or visual_features.shape != audio_features.shape:
            raise ValueError('SIE inputs must have identical shapes')

        batch_size, sequence_length, _ = audio_features.shape
        audio_energy = torch.linalg.vector_norm(audio_features, dim=-1)
        local_energy = self._local_mean(audio_energy)
        background_mean = local_energy.mean(dim=1, keepdim=True)
        background_std = local_energy.std(dim=1, keepdim=True, unbiased=False)
        deviation = F.relu((local_energy - background_mean) / (background_std + 1e-6))

        top_k = max(1, min(sequence_length, int(sequence_length * self.topk_ratio + 0.999999)))
        _, candidate_indices = torch.topk(deviation, k=top_k, dim=1)

        visual_audio_similarity = F.relu(self._cosine_similarity(visual_features, audio_features))
        text_audio_similarity = F.relu(self._cosine_similarity(text_features, audio_features))
        consistency = visual_audio_similarity * text_audio_similarity
        candidate_consistency = consistency.gather(1, candidate_indices)
        candidate_deviation = deviation.gather(1, candidate_indices)
        valid_candidates = candidate_consistency > self.similarity_threshold

        positions = torch.arange(sequence_length, device=audio_features.device, dtype=audio_features.dtype)
        distances = positions.view(1, sequence_length, 1) - candidate_indices.unsqueeze(1).to(audio_features.dtype)
        gaussian_weights = torch.exp(-(distances.square()) / (2.0 * self.sigma ** 2))
        alpha = self.alpha.clamp(0.0, 1.0)
        candidate_scores = alpha * candidate_consistency + (
            1.0 - alpha
        ) * candidate_deviation
        candidate_scores = candidate_scores * valid_candidates.to(candidate_scores.dtype)
        enhancement = torch.sum(gaussian_weights * candidate_scores.unsqueeze(1), dim=-1)

        enhanced_energy = audio_energy + self.epsilon_audio.clamp_min(0.0) * enhancement
        modulation = 1.0 + self.lambda_audio.clamp_min(0.0) * enhanced_energy
        return visual_features * modulation.unsqueeze(-1), text_features * modulation.unsqueeze(-1)
