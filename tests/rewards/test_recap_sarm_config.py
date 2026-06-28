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

from lerobot.rewards.recap_sarm.configuration_recap_sarm import RECAPSARMConfig


def test_recap_sarm_defaults():
    cfg = RECAPSARMConfig()

    assert cfg.num_value_bins == 201
    assert cfg.value_min == -1.0
    assert cfg.value_max == 0.0
    assert cfg.advantage_lookahead == 50
    assert set(cfg.output_features) == {"value_logits", "value"}
    assert "sparse_progress" not in cfg.output_features
