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

from contextlib import nullcontext
from types import SimpleNamespace

import datasets
import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.utils.task_conditioning import (
    SARMTaskConditioner,
    SARMTaskConditioningConfig,
    make_task_conditioner,
)


def _dataset(indices, episodes, interventions, *, include_intervention_feature=True):
    features = {"intervention": {"dtype": "bool", "shape": [1]}}
    if not include_intervention_feature:
        features = {}
    hf_dataset = datasets.Dataset.from_dict(
        {
            "index": indices,
            "episode_index": episodes,
            "intervention": [[value] for value in interventions],
        }
    )
    return SimpleNamespace(meta=SimpleNamespace(features=features), hf_dataset=hf_dataset)


def _write_progress(path, indices, episodes, progress, *, head_mode="sparse"):
    pd.DataFrame(
        {
            "index": indices,
            "episode_index": episodes,
            f"progress_{head_mode}": progress,
        }
    ).to_parquet(path)
    return path


def test_binary_labels_use_strict_threshold_and_human_chunk_priority(tmp_path):
    dataset = _dataset(
        indices=list(range(6)),
        episodes=[0] * 6,
        interventions=[False, False, False, False, True, False],
    )
    progress_path = _write_progress(
        tmp_path / "progress.parquet",
        indices=list(range(6)),
        episodes=[0] * 6,
        progress=[0.0, 0.1, 0.3, 0.3, -1.0, -2.0],
    )

    conditioner = SARMTaskConditioner(
        dataset=dataset,
        progress_path=progress_path,
        chunk_size=2,
        threshold=0.2,
    )

    assert conditioner.label_lookup[0] == "positive"  # delta 0.3 > threshold
    assert conditioner.label_lookup[1] == "negative"  # delta == threshold
    assert conditioner.label_lookup[2] == "negative"  # endpoint intervention is outside [t, t+2)
    assert conditioner.label_lookup[3] == "positive"  # intervention at the second action frame
    assert conditioner.label_lookup[4] == "positive"  # intervention at the first action frame


def test_mixed_autonomous_human_chunk_is_kept_and_positive_with_missing_progress(tmp_path):
    dataset = _dataset(
        indices=[0, 1, 2],
        episodes=[0, 0, 0],
        interventions=[False, True, False],
    )
    progress_path = _write_progress(
        tmp_path / "progress.parquet",
        indices=[0, 1, 2],
        episodes=[0, 0, 0],
        progress=[np.nan, np.nan, 0.0],
    )

    conditioner = SARMTaskConditioner(
        dataset=dataset,
        progress_path=progress_path,
        chunk_size=2,
    )

    assert conditioner.label_lookup == {0: "positive", 1: "positive", 2: "negative"}


def test_episode_tail_clamps_delta_endpoint_without_crossing_episode(tmp_path):
    dataset = _dataset(
        indices=[0, 1, 10, 11],
        episodes=[0, 0, 1, 1],
        interventions=[False] * 4,
    )
    progress_path = _write_progress(
        tmp_path / "progress.parquet",
        indices=[0, 1, 10, 11],
        episodes=[0, 0, 1, 1],
        progress=[0.0, 0.1, 100.0, 101.0],
    )

    conditioner = SARMTaskConditioner(
        dataset=dataset,
        progress_path=progress_path,
        chunk_size=3,
        threshold=0.5,
    )

    assert conditioner.label_lookup[0] == "negative"
    assert conditioner.label_lookup[1] == "negative"
    assert conditioner.label_lookup[10] == "positive"
    assert conditioner.label_lookup[11] == "negative"


def test_selected_episodes_keep_global_index_alignment(tmp_path):
    dataset = _dataset(
        indices=[10, 11, 12],
        episodes=[2, 2, 2],
        interventions=[False, False, False],
    )
    progress_path = _write_progress(
        tmp_path / "progress.parquet",
        indices=list(range(13)),
        episodes=[0] * 5 + [1] * 5 + [2] * 3,
        progress=[0.0] * 10 + [0.0, 0.1, 0.5],
    )

    conditioner = SARMTaskConditioner(
        dataset=dataset,
        progress_path=progress_path,
        chunk_size=2,
        threshold=0.2,
    )

    batch = {"index": torch.tensor([10, 11]), "task": ["pick cube\n", "place cube  \n"]}
    conditioner.condition_batch(batch)
    assert batch["task"] == ["pick cube task:positive", "place cube task:positive"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_progress_column", "missing required columns"),
        ("missing_index", "missing from the SARM progress"),
        ("episode_mismatch", "Episode mismatch"),
        ("autonomous_nan", "Autonomous chunk"),
    ],
)
def test_invalid_progress_data_fails_fast(tmp_path, mutation, match):
    dataset = _dataset(
        indices=[0, 1, 2],
        episodes=[0, 0, 0],
        interventions=[False, False, False],
    )
    head_mode = "dense" if mutation == "missing_progress_column" else "sparse"
    indices = [0, 1, 2]
    episodes = [0, 0, 0]
    progress = [0.0, 0.1, 0.2]
    if mutation == "missing_index":
        indices = [0, 2]
        episodes = [0, 0]
        progress = [0.0, 0.2]
    elif mutation == "episode_mismatch":
        episodes[1] = 1
    elif mutation == "autonomous_nan":
        progress[0] = np.nan
    progress_path = _write_progress(
        tmp_path / "progress.parquet",
        indices=indices,
        episodes=episodes,
        progress=progress,
    )

    with pytest.raises(ValueError, match=match):
        SARMTaskConditioner(
            dataset=dataset,
            progress_path=progress_path,
            chunk_size=1,
            head_mode=head_mode,
        )


def test_missing_intervention_feature_fails_fast(tmp_path):
    dataset = _dataset(
        indices=[0],
        episodes=[0],
        interventions=[False],
        include_intervention_feature=False,
    )
    progress_path = _write_progress(
        tmp_path / "progress.parquet",
        indices=[0],
        episodes=[0],
        progress=[0.0],
    )

    with pytest.raises(ValueError, match="requires an 'intervention' feature"):
        SARMTaskConditioner(dataset=dataset, progress_path=progress_path, chunk_size=1)


def test_factory_auto_detects_dataset_local_progress_and_requires_chunk_size(tmp_path):
    dataset = _dataset(indices=[0], episodes=[0], interventions=[False])
    _write_progress(
        tmp_path / "sarm_progress.parquet",
        indices=[0],
        episodes=[0],
        progress=[0.0],
    )
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=1))

    conditioner = make_task_conditioner(
        SARMTaskConditioningConfig(),
        dataset=dataset,
        policy=policy,
        dataset_root=tmp_path,
    )
    assert isinstance(conditioner, SARMTaskConditioner)

    policy.config.chunk_size = None
    with pytest.raises(ValueError, match="chunk_size"):
        make_task_conditioner(
            SARMTaskConditioningConfig(),
            dataset=dataset,
            policy=policy,
            dataset_root=tmp_path,
        )


def test_task_conditioning_config_validates_type_head_and_threshold():
    with pytest.raises(ValueError, match="Unknown task conditioning type"):
        SARMTaskConditioningConfig(type="other")
    with pytest.raises(ValueError, match="head_mode"):
        SARMTaskConditioningConfig(head_mode="other")
    with pytest.raises(ValueError, match="threshold"):
        SARMTaskConditioningConfig(threshold=float("nan"))


def test_conditioned_batch_uses_ordinary_policy_forward_without_sample_weights():
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(1.0))
            self.forward_kwargs = None

        def forward(self, batch, **kwargs):
            self.forward_kwargs = kwargs
            return (self.parameter - batch["target"]).square(), {}

    class Accelerator:
        def autocast(self):
            return nullcontext()

        def backward(self, loss):
            loss.backward()

        def clip_grad_norm_(self, parameters, max_norm):
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

        def unwrap_model(self, policy, **kwargs):
            return policy

    policy = Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    accelerator = Accelerator()
    metrics = {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s")}
    tracker = MetricsTracker(
        batch_size=1,
        num_frames=1,
        num_episodes=1,
        metrics=metrics,
        accelerator=SimpleNamespace(num_processes=1),
    )

    update_policy(
        tracker,
        policy,
        {"task": ["pick task:positive"], "target": torch.tensor(0.0)},
        optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
        sample_weighter=None,
    )

    assert policy.forward_kwargs == {}
