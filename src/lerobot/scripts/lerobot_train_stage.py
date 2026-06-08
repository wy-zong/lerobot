#!/usr/bin/env python

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
"""Train SmolVLA with PI0.5-style subtask post-training."""

import logging
from collections import deque
from collections.abc import Callable
from types import MethodType
from typing import TYPE_CHECKING, Any

from lerobot.configs import parser
from lerobot.configs.train_stage import TrainStagePipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins

from . import lerobot_train

if TYPE_CHECKING:
    from accelerate import Accelerator


def validate_stage_dataset(dataset: Any, cfg: TrainStagePipelineConfig) -> None:
    """Ensure the dataset has the annotations required by lerobot-train-stage."""

    if not cfg.subtask_required:
        return

    meta = getattr(dataset, "meta", None)
    features = getattr(dataset, "features", None)
    if features is None and meta is not None:
        features = getattr(meta, "features", None)
    feature_keys = set(features or {})

    missing = []
    if "subtask_index" not in feature_keys:
        missing.append("data feature 'subtask_index'")

    subtasks = getattr(meta, "subtasks", None) if meta is not None else None
    if subtasks is None:
        missing.append("meta/subtasks.parquet")
    elif hasattr(subtasks, "__len__") and len(subtasks) == 0:
        missing.append("non-empty meta/subtasks.parquet")

    if missing:
        raise ValueError(
            "lerobot-train-stage requires a dataset exported with subtask annotations. "
            f"Missing: {', '.join(missing)}. "
            "Expected parquet rows to contain subtask_index and metadata at meta/subtasks.parquet."
        )


def _find_forward_stage(policy: Any) -> Callable[..., Any]:
    queue = deque([policy])
    seen: set[int] = set()
    while queue:
        candidate = queue.popleft()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        forward_stage = getattr(candidate, "forward_stage", None)
        if callable(forward_stage):
            return forward_stage

        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            queue.append(get_base_model())

        for attr in ("base_model", "model", "module"):
            child = getattr(candidate, attr, None)
            if child is not None:
                queue.append(child)

    raise ValueError("lerobot-train-stage requires a SmolVLA policy with forward_stage().")


def install_stage_forward(policy: Any, cfg: TrainStagePipelineConfig) -> Any:
    """Route the train loop's policy.forward() call to SmolVLA forward_stage()."""

    forward_stage = _find_forward_stage(policy)

    def _stage_forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        return forward_stage(
            batch,
            noise=noise,
            time=time,
            reduction=reduction,
            flow_loss_weight=cfg.flow_loss_weight,
        )

    policy.forward = MethodType(_stage_forward, policy)
    logging.info(
        "Using SmolVLA stage objective: loss = subtask_ce_loss + %s * action_flow_loss",
        cfg.flow_loss_weight,
    )
    return policy


@parser.wrap()
def train(cfg: TrainStagePipelineConfig, accelerator: "Accelerator | None" = None):
    return lerobot_train.train.__wrapped__(
        cfg,
        accelerator,
        dataset_validator=validate_stage_dataset,
        policy_setup_fn=install_stage_forward,
    )


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
