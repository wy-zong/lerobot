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

"""SARM-based subtask predictor for runtime subtask switching during rollout."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torchvision.transforms.functional as F

logger = logging.getLogger(__name__)


class SARMSubtaskPredictor:
    """Encapsulates SARM model + CLIP encoder for runtime subtask prediction.

    During rollout, the VLA policy produces action chunks. This predictor
    runs SARM inference **only** when the VLA is about to produce a new
    action chunk (i.e. when its internal action cache is exhausted),
    predicting the current subtask so the VLA can use the appropriate
    task prompt.

    For the sync inference engine, the predictor uses an internal step
    counter aligned with ``n_action_steps`` to gate predictions.  For
    the RTC engine the caller already gates invocations via queue
    threshold checks, so ``n_action_steps`` should be set to ``1``.
    """

    def __init__(
        self,
        sarm_model: Any,
        clip_model: Any,
        clip_processor: Any,
        device: str,
        n_action_steps: int = 1,
    ) -> None:
        self._sarm_model = sarm_model
        self._clip_model = clip_model
        self._clip_processor = clip_processor
        self._device = device
        self._n_action_steps = max(1, n_action_steps)
        self._step_counter = 0
        self._current_subtask: str | None = None

        # Resolve subtask names from config
        cfg = sarm_model.config
        if cfg.annotation_mode in ("dense_only", "dual"):
            self._subtask_names: list[str] = list(cfg.dense_subtask_names or [])
            self._head_mode = "dense"
        else:
            # single_stage or any sparse-like mode
            self._subtask_names = list(cfg.sparse_subtask_names or [])
            self._head_mode = "sparse"

    def reset(self) -> None:
        """Reset the step counter and cached subtask (call on episode boundary)."""
        self._step_counter = 0
        self._current_subtask = None

    def predict_subtask(self, obs_batch: dict, current_task: str) -> str:
        """Predict the current subtask, gated by ``n_action_steps``.

        Only runs CLIP + SARM inference when the step counter is aligned
        with the VLA prediction boundary.  Otherwise returns the most
        recently predicted subtask (or *current_task* if none yet).

        Args:
            obs_batch: Observation dict that should contain an image tensor
                under the SARM image key.
            current_task: The task string currently used by the VLA.

        Returns:
            The (possibly updated) task string.
        """
        should_predict = self._step_counter % self._n_action_steps == 0
        self._step_counter += 1

        if not should_predict:
            return self._current_subtask or current_task

        new_task = self._run_inference(obs_batch, current_task)
        self._current_subtask = new_task
        return new_task

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_image_key(self, obs_batch: dict) -> str | None:
        """Resolve the image key in *obs_batch*, trying several conventions."""
        img_key = self._sarm_model.config.image_key
        if img_key in obs_batch:
            return img_key

        # build_dataset_frame may add 'observation.' prefix
        suffix = img_key.split(".")[-1]
        for candidate in (
            f"observation.images.{suffix}",
            suffix,
        ):
            if candidate in obs_batch:
                return candidate

        return None

    def _run_inference(self, obs_batch: dict, current_task: str) -> str:
        """Run CLIP encoding + SARM forward pass and return the predicted subtask."""
        img_key = self._find_image_key(obs_batch)
        if img_key is None:
            logger.warning(
                "SARM image key '%s' not found in observation keys %s. "
                "Skipping subtask prediction.",
                self._sarm_model.config.image_key,
                list(obs_batch.keys()),
            )
            return current_task

        with torch.inference_mode():
            img_tensor = obs_batch[img_key]  # [C, H, W]
            img_pil = F.to_pil_image(img_tensor)

            clip_inputs = self._clip_processor(
                images=img_pil, return_tensors="pt"
            ).to(self._device)
            video_emb = self._clip_model.get_image_features(**clip_inputs)  # [1, 512]
            video_emb = video_emb.unsqueeze(1)  # [1, 1, 512]

            text_emb = torch.zeros(
                (1, 512), dtype=torch.float32, device=self._device
            )

            outputs = self._sarm_model.calculate_rewards(
                text_embeddings=text_emb,
                video_embeddings=video_emb,
                return_stages=True,
                return_all_frames=True,
                head_mode=self._head_mode,
            )

            if len(outputs) >= 2:
                stage_probs = outputs[1]
                stage_idx = stage_probs[0, 0].argmax(-1).item()
                if stage_idx < len(self._subtask_names):
                    new_subtask = self._subtask_names[stage_idx]
                    if current_task != new_subtask:
                        logger.info(
                            "SARM switched subtask: '%s' -> '%s'",
                            current_task,
                            new_subtask,
                        )
                    return new_subtask

        return current_task
