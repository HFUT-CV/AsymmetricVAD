from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models import CrossModalFusion, LookBackController, SIEModule
from utils.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class AsymmetricVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 device,
                 similarity_threshold: float = 0.3,
                 lookback_steps: int = 3,
                 sie_window_radius: int = 2,
                 sie_topk_ratio: float = 0.17,
                 sie_sigma: float = 3.0,
                 sie_alpha_init: float = 0.5,
                 sie_epsilon_audio_init: float = 0.1,
                 sie_lambda_audio_init: float = 0.1,
                 uncertainty_threshold: float = 0.4):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.lookback_steps = lookback_steps

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)
        self.sie = SIEModule(
            similarity_threshold=similarity_threshold,
            window_radius=sie_window_radius,
            topk_ratio=sie_topk_ratio,
            sigma=sie_sigma,
            alpha_init=sie_alpha_init,
            epsilon_audio_init=sie_epsilon_audio_init,
            lambda_audio_init=sie_lambda_audio_init,
        )
        self.cross_modal_fusion = CrossModalFusion(visual_width, num_heads=1)
        self.look_back = LookBackController(
            visual_width, self.sie, self.cross_modal_fusion,
            tau_unc=uncertainty_threshold, max_lookback=lookback_steps
        )

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=images.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, padding_mask))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1], x.device)
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def forward(self, visual, padding_mask, lengths, text_features=None, category_text_features=None, audio=None):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        if category_text_features is None:
            raise ValueError('category_text_features must be provided')
        category_text_features = category_text_features.to(visual_features.device, dtype=visual_features.dtype)
        batch_size, sequence_length, feature_dim = visual_features.shape

        if text_features is None:
            temporal_text = category_text_features.mean(dim=0).view(1, 1, -1)
            temporal_text = temporal_text.expand(batch_size, sequence_length, -1)
        else:
            temporal_text = text_features.to(visual_features.device, dtype=visual_features.dtype)
            if temporal_text.ndim != 3 or temporal_text.shape[:2] != (batch_size, sequence_length):
                raise ValueError('text_features must have shape [B, L, D] aligned with visual')
            if temporal_text.shape[-1] != feature_dim:
                raise ValueError('text_features dimension must match visual_width')

        if audio is None:
            audio_features = torch.zeros_like(visual_features)
        else:
            audio_features = audio.to(visual_features.device, dtype=visual_features.dtype)
            if audio_features.shape != visual_features.shape:
                raise ValueError('audio must have the same shape as visual features')

        sie_visual, sie_text = self.sie(visual_features, temporal_text, audio_features)
        aligned_visual, aligned_text, joint_features = self.cross_modal_fusion(
            sie_visual, sie_text, padding_mask
        )
        syn_features, _, _ = self.look_back(
            visual_features,
            temporal_text,
            audio_features,
            aligned_visual,
            aligned_text,
            joint_features,
            lengths,
            padding_mask,
        )

        logits1 = self.classifier(syn_features + self.mlp2(syn_features))
        visual_features_norm = F.normalize(syn_features, dim=-1)
        text_features_norm = F.normalize(category_text_features, dim=-1)
        logits2 = visual_features_norm @ text_features_norm.t().type(visual_features_norm.dtype) / 0.07

        return category_text_features, logits1, logits2
    