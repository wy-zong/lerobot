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
import draccus
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import ValidationConfig


def test_dataset_config_valid():
    DatasetConfig(repo_id="user/repo", episodes=[0, 1, 2])


def test_dataset_config_negative_episodes():
    with pytest.raises(ValueError, match="non-negative"):
        DatasetConfig(repo_id="user/repo", episodes=[0, -1, 2])


def test_dataset_config_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id="user/repo", episodes=[0, 1, 1, 2])


def test_dataset_config_none_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=None)


def test_dataset_config_empty_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=[])


def test_dataset_config_intervention_only_defaults_false():
    assert DatasetConfig(repo_id="user/repo").intervention_only is False


def test_dataset_config_intervention_only_true():
    assert DatasetConfig(repo_id="user/repo", intervention_only=True).intervention_only is True


def test_dataset_config_intervention_only_cli_true():
    config = draccus.parse(
        DatasetConfig,
        args=["--repo_id=user/repo", "--intervention_only=true"],
    )
    assert config.intervention_only is True


def test_validation_config_defaults_disabled():
    cfg = ValidationConfig()

    assert cfg.enable is False
    assert cfg.ratio == 0.2
    assert cfg.freq == 1000
    assert cfg.max_batches == 64
    assert cfg.episodes is None
    assert cfg.split == "tail"


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.0, 1.5])
def test_validation_config_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="validation.ratio"):
        ValidationConfig(ratio=ratio)


def test_validation_config_rejects_unsupported_split():
    with pytest.raises(ValueError, match="validation.split"):
        ValidationConfig(split="random")


def test_validation_config_rejects_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        ValidationConfig(episodes=[1, 1])
