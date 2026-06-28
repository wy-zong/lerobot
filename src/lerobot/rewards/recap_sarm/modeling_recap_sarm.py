# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RECAP-style distributional value model using the SARM multimodal backbone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import OBS_STR

from ..pretrained import PreTrainedRewardModel
from ..sarm.sarm_utils import pad_state_to_max_dim
from .configuration_recap_sarm import RECAPSARMConfig


def estimate_advantage(
    value_t: Tensor | np.ndarray | float,
    rewards_t_to_t_plus_n_minus_1: Tensor | np.ndarray | Sequence[float] | float,
    value_t_plus_n: Tensor | np.ndarray | float,
) -> Tensor:
    """Compute a RECAP-style bootstrapped advantage."""

    value_t_tensor = torch.as_tensor(value_t, dtype=torch.float32)
    reward_tensor = torch.as_tensor(rewards_t_to_t_plus_n_minus_1, dtype=torch.float32)
    value_t_plus_n_tensor = torch.as_tensor(value_t_plus_n, dtype=torch.float32)

    if reward_tensor.ndim == 0:
        reward_tensor = reward_tensor.unsqueeze(0)

    return reward_tensor.sum(dim=-1) + value_t_plus_n_tensor - value_t_tensor


class RECAPSARMTemporalBackbone(nn.Module):
    """Multimodal temporal fusion backbone shared by the value heads."""

    def __init__(
        self,
        d_model: int = 512,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_cameras: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        fused_in = d_model * (num_cameras + 2)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

    def _prep_lang(self, lang_emb: torch.Tensor, batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
        if lang_emb.dim() == 3:
            return self.lang_proj(lang_emb).unsqueeze(1)
        return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_len, d_model)

    def forward(
        self,
        img_seq: torch.Tensor,
        lang_emb: torch.Tensor,
        state: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_cameras, seq_len, _ = img_seq.shape
        d_model = self.d_model
        device = img_seq.device

        vis_proj = self.visual_proj(img_seq)
        state_proj = self.state_proj(state).unsqueeze(1)
        lang_proj = self._prep_lang(lang_emb, batch_size, seq_len, d_model)

        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)
        x[:, :num_cameras, 0, :] = x[:, :num_cameras, 0, :] + self.first_pos

        x_tokens = x.view(batch_size, (num_cameras + 2) * seq_len, d_model)
        token_count = x_tokens.size(1)

        base_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        padding_mask = base_mask.unsqueeze(1).expand(batch_size, num_cameras + 2, seq_len).reshape(
            batch_size, (num_cameras + 2) * seq_len
        )
        causal_mask = torch.triu(torch.ones(token_count, token_count, device=device, dtype=torch.bool), diagonal=1)

        hidden = self.transformer(
            x_tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        hidden = hidden.view(batch_size, num_cameras + 2, seq_len, d_model)
        fused = hidden.permute(0, 2, 1, 3).reshape(batch_size, seq_len, (num_cameras + 2) * d_model)
        return self.fusion_backbone(fused)


class RECAPSARMRewardModel(PreTrainedRewardModel):
    """Distributional value model aligned with RECAP-style advantage estimation."""

    name = "recap_sarm"
    config_class = RECAPSARMConfig

    def __init__(self, config: RECAPSARMConfig, dataset_stats: dict | None = None, dataset_meta=None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = torch.device(
            config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.temporal_backbone = RECAPSARMTemporalBackbone(
            d_model=config.hidden_dim,
            vis_emb_dim=config.image_dim,
            text_emb_dim=config.text_dim,
            state_dim=config.max_state_dim,
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            dropout=config.dropout,
            num_cameras=1,
        )
        self.value_head = nn.Linear(config.hidden_dim, config.num_value_bins)
        self.stage_aux_head = (
            nn.Linear(config.hidden_dim, config.num_sparse_stages) if config.use_stage_aux_loss else None
        )

        self.register_buffer(
            "value_bin_centers",
            torch.linspace(config.value_min, config.value_max, config.num_value_bins, dtype=torch.float32),
        )
        self.register_buffer(
            "value_bin_edges",
            torch.linspace(config.value_min, config.value_max, config.num_value_bins + 1, dtype=torch.float32),
        )

        self.to(self.device)

    def to(self, device):
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def _prepare_inputs_from_batch(
        self, batch: dict[str, Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        observation = batch.get(OBS_STR, batch)
        text_features = observation.get("text_features")
        video_features = observation.get("video_features", observation.get("observation_features"))
        state_features = observation.get("state_features", observation.get("observation.state"))
        lengths = observation.get("lengths")

        if text_features is None:
            raise ValueError("text_features are required for RECAP-SARM inference")
        if video_features is None:
            raise ValueError("video_features are required for RECAP-SARM inference")

        return text_features, video_features, state_features, lengths

    def _coerce_inputs(
        self,
        text_embeddings: np.ndarray | torch.Tensor,
        video_embeddings: np.ndarray | torch.Tensor,
        state_features: np.ndarray | torch.Tensor | None = None,
        lengths: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
        if state_features is not None and isinstance(state_features, np.ndarray):
            state_features = torch.tensor(state_features, dtype=torch.float32)
        if lengths is not None and isinstance(lengths, np.ndarray):
            lengths = torch.tensor(lengths, dtype=torch.int32)

        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
            video_embeddings = video_embeddings.unsqueeze(0)
            if state_features is not None:
                state_features = state_features.unsqueeze(0)
            if lengths is not None and lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)

        return text_embeddings, video_embeddings, state_features, lengths

    def predict_value_distribution(
        self,
        batch_or_text_embeddings: dict[str, Tensor] | np.ndarray | torch.Tensor,
        video_embeddings: np.ndarray | torch.Tensor | None = None,
        state_features: np.ndarray | torch.Tensor | None = None,
        lengths: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict value logits, probabilities, and expectations for each timestep."""

        if isinstance(batch_or_text_embeddings, dict):
            text_embeddings, video_embeddings, state_features, lengths = self._prepare_inputs_from_batch(
                batch_or_text_embeddings
            )
        else:
            if video_embeddings is None:
                raise ValueError("video_embeddings are required when passing raw embeddings")
            text_embeddings = batch_or_text_embeddings

        text_embeddings, video_embeddings, state_features, lengths = self._coerce_inputs(
            text_embeddings, video_embeddings, state_features, lengths
        )

        batch_size = video_embeddings.shape[0]
        seq_len = video_embeddings.shape[1]
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)

        state = (
            state_features
            if state_features is not None
            else torch.zeros(batch_size, seq_len, self.config.max_state_dim, dtype=torch.float32)
        )
        state = pad_state_to_max_dim(state, self.config.max_state_dim)

        hidden_states = self.temporal_backbone(
            video_embeddings.unsqueeze(1).to(self.device),
            text_embeddings.to(self.device),
            state.to(self.device),
            lengths.to(self.device),
        )
        value_logits = self.value_head(hidden_states)
        value_probs = F.softmax(value_logits, dim=-1)
        value_expectation = torch.sum(value_probs * self.value_bin_centers, dim=-1)

        outputs = {
            "value_logits": value_logits,
            "value_probs": value_probs,
            "value_expectation": value_expectation,
            "hidden_states": hidden_states,
            "lengths": lengths.to(self.device),
        }
        if self.stage_aux_head is not None:
            outputs["stage_aux_logits"] = self.stage_aux_head(hidden_states)
        return outputs

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.predict_value_distribution(batch)
        frame_index = min(self.config.n_obs_steps, outputs["value_expectation"].shape[1] - 1)
        return outputs["value_expectation"][:, frame_index]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        observation = batch.get(OBS_STR, batch)
        value_targets_bin = observation.get("value_targets_bin")
        value_targets_continuous = observation.get("value_targets_continuous")

        if value_targets_bin is None:
            raise ValueError("value_targets_bin is required for RECAP-SARM training")
        if value_targets_continuous is None:
            raise ValueError("value_targets_continuous is required for RECAP-SARM training")

        outputs = self.predict_value_distribution(batch)
        value_logits = outputs["value_logits"]
        value_expectation = outputs["value_expectation"]
        lengths = outputs["lengths"]

        value_targets_bin = value_targets_bin.to(self.device)
        value_targets_continuous = value_targets_continuous.to(self.device)

        seq_len = value_logits.shape[1]
        valid_mask = torch.arange(seq_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)

        flat_logits = value_logits[valid_mask]
        flat_bin_targets = value_targets_bin[valid_mask]
        flat_cont_targets = value_targets_continuous[valid_mask]
        flat_expectation = value_expectation[valid_mask]

        value_loss = F.cross_entropy(flat_logits, flat_bin_targets, reduction="mean")
        value_mae = torch.abs(flat_expectation - flat_cont_targets).mean()
        total_loss = value_loss

        metrics: dict[str, Any] = {
            "value_loss": value_loss.item(),
            "value_mae": value_mae.item(),
            "value_expectation_mean": flat_expectation.mean().item(),
            "loss": total_loss.item(),
            "total_loss": total_loss.item(),
        }

        if self.stage_aux_head is not None:
            sparse_targets = observation.get("sparse_targets")
            if sparse_targets is not None:
                sparse_targets = sparse_targets.to(self.device)
                stage_targets = torch.floor(sparse_targets).long().clamp(0, self.config.num_sparse_stages - 1)
                stage_aux_logits = outputs["stage_aux_logits"]
                stage_aux_loss = F.cross_entropy(stage_aux_logits[valid_mask], stage_targets[valid_mask])
                total_loss = total_loss + self.config.stage_loss_weight * stage_aux_loss
                metrics["stage_aux_loss"] = stage_aux_loss.item()
                metrics["loss"] = total_loss.item()
                metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
