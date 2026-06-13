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

import pytest
import torch

from lerobot.processor import IntraChunkSmoothingProcessorStep


def _diff_std(action: torch.Tensor, order: int) -> torch.Tensor:
    return torch.diff(action, n=order, dim=-2).std()


def test_intra_chunk_smoothing_preserves_cubic_chunk():
    step = IntraChunkSmoothingProcessorStep()
    t = torch.linspace(-1.0, 1.0, 16)
    chunk = torch.stack(
        [
            0.5 + 2.0 * t,
            1.0 - 0.25 * t + 0.75 * t**2 - 0.4 * t**3,
        ],
        dim=-1,
    )
    action = torch.stack([chunk, chunk + torch.tensor([0.25, -0.5])], dim=0)

    smoothed = step.action(action)

    torch.testing.assert_close(smoothed, action, atol=1e-5, rtol=1e-5)


def test_intra_chunk_smoothing_reduces_high_frequency_noise():
    step = IntraChunkSmoothingProcessorStep()
    t = torch.linspace(-1.0, 1.0, 32)
    base = torch.stack([t, 0.5 * t - 0.25], dim=-1)
    alternating = torch.where(torch.arange(t.numel()) % 2 == 0, 1.0, -1.0).unsqueeze(-1)
    action = (base + 0.15 * alternating).unsqueeze(0)

    smoothed = step.action(action)

    assert _diff_std(smoothed, order=1) < _diff_std(action, order=1)
    assert _diff_std(smoothed, order=2) < _diff_std(action, order=2)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_intra_chunk_smoothing_preserves_shape_dtype_and_device(device: str):
    step = IntraChunkSmoothingProcessorStep()
    action = torch.randn(2, 8, 3, dtype=torch.float32, device=device)

    smoothed = step.action(action)

    assert smoothed.shape == action.shape
    assert smoothed.dtype == action.dtype
    assert smoothed.device == action.device


def test_intra_chunk_smoothing_preserves_float64_dtype():
    step = IntraChunkSmoothingProcessorStep()
    action = torch.randn(2, 8, 3, dtype=torch.float64)

    smoothed = step.action(action)

    assert smoothed.dtype == torch.float64


def test_intra_chunk_smoothing_accepts_unbatched_chunk():
    step = IntraChunkSmoothingProcessorStep()
    t = torch.linspace(-1.0, 1.0, 12)
    action = torch.stack([t, torch.where(torch.arange(t.numel()) % 2 == 0, 1.0, -1.0)], dim=-1)

    smoothed = step.action(action)

    assert smoothed.shape == action.shape
    assert _diff_std(smoothed, order=2) < _diff_std(action, order=2)


def test_intra_chunk_smoothing_short_chunk_is_noop():
    step = IntraChunkSmoothingProcessorStep()
    action = torch.randn(1, 3, 2)

    smoothed = step.action(action)

    assert smoothed is action


def test_intra_chunk_smoothing_disabled_is_noop():
    step = IntraChunkSmoothingProcessorStep(enabled=False)
    action = torch.randn(1, 8, 2)

    smoothed = step.action(action)

    assert smoothed is action


def test_intra_chunk_smoothing_rejects_non_cubic_degree():
    with pytest.raises(ValueError, match="degree=3"):
        IntraChunkSmoothingProcessorStep(degree=2)
