# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Intra-chunk action smoothing processor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import PolicyAction

from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("intra_chunk_smoothing_processor")
@dataclass
class IntraChunkSmoothingProcessorStep(PolicyActionProcessorStep):
    """Smooth action chunks by fitting a cubic polynomial along chunk time.

    The step operates independently for each batch element and action
    dimension.  It is intended for runtime rollout postprocessing of
    ``predict_action_chunk`` outputs and leaves feature metadata unchanged.
    """

    enabled: bool = True
    degree: int = 3

    def __post_init__(self) -> None:
        if self.degree != 3:
            raise ValueError(
                "IntraChunkSmoothingProcessorStep currently supports only degree=3 "
                f"(cubic smoothing), got degree={self.degree}."
            )

    def action(self, action: PolicyAction) -> PolicyAction:
        if not self.enabled or not action.is_floating_point():
            return action

        if action.ndim == 3:
            batched_action = action
            squeeze_batch = False
        elif action.ndim == 2:
            batched_action = action.unsqueeze(0)
            squeeze_batch = True
        else:
            return action

        if batched_action.shape[1] < self.degree + 1:
            return action

        smoothed = self._smooth_batched_chunk(batched_action)
        if squeeze_batch:
            return smoothed.squeeze(0)
        return smoothed

    def _smooth_batched_chunk(self, action: torch.Tensor) -> torch.Tensor:
        batch_size, chunk_length, action_dim = action.shape
        compute_dtype = torch.float64 if action.dtype == torch.float64 else torch.float32
        action_work = action.to(dtype=compute_dtype)

        # Normalized time improves conditioning without changing the fitted curve.
        time = torch.linspace(-1.0, 1.0, chunk_length, device=action.device, dtype=compute_dtype)
        design = torch.stack([time.pow(power) for power in range(self.degree + 1)], dim=-1)

        targets = action_work.permute(1, 0, 2).reshape(chunk_length, batch_size * action_dim)
        coefficients = torch.linalg.lstsq(design, targets).solution
        fitted = design @ coefficients
        smoothed = fitted.reshape(chunk_length, batch_size, action_dim).permute(1, 0, 2)
        return smoothed.to(dtype=action.dtype)

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "degree": self.degree}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
