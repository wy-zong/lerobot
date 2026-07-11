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

import numpy as np
import pytest

from lerobot.rewards.recap_sarm.annotate_dataset import (
    _episode_success,
    compute_episode_advantages,
)


def test_compute_episode_advantages_matches_normalized_return():
    values = np.array([-1.0, -0.75, -0.5, -0.25], dtype=np.float32)

    advantages = compute_episode_advantages(
        values,
        success=True,
        normalization_length=4,
        failure_penalty=2.0,
        lookahead=2,
    )

    np.testing.assert_allclose(advantages, np.zeros(4), atol=1e-6)


def test_compute_episode_advantages_without_outcome_only_masks_terminal_window():
    values = np.linspace(-1.0, 0.0, 101, dtype=np.float32)

    advantages = compute_episode_advantages(
        values,
        success=None,
        normalization_length=100,
        failure_penalty=50.0,
        lookahead=10,
    )

    assert np.isfinite(advantages[:91]).all()
    assert np.isnan(advantages[91:]).all()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (None, None),
    ],
)
def test_episode_success_preserves_missing_values(value, expected):
    assert _episode_success({"episode_success": value}) is expected
