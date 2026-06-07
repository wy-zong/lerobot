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

"""Inspect live SmolVLA VLM/expert features from real robot observations."""

import json
import logging
import time
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import is_headless
from lerobot.configs import FeatureType, PreTrainedConfig, parser
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OBS_STR,
)
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class VLMInspectConfig:
    robot: RobotConfig | None = None
    policy: PreTrainedConfig | None = None
    device: str | None = "cuda"
    mode: str = "feature"
    fps: int = 10
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    output_dir: Path = Path("outputs/vlm_inspect")
    rename_map: dict[str, str] = field(default_factory=dict)
    capture_once: bool = False
    question: str | None = None
    answer_max_new_tokens: int = 16
    camera_only_answer: bool = False
    answer_score_only: bool = False

    def __post_init__(self) -> None:
        if self.robot is None:
            raise ValueError("--robot.type is required for lerobot-vlm-inspect")
        if self.mode not in {"feature", "answer"}:
            raise ValueError(f"--mode must be either 'feature' or 'answer', got '{self.mode}'.")
        if self.capture_once and self.question is None:
            raise ValueError("--question is required when --capture_once=true.")
        if self.camera_only_answer and (self.mode != "answer" or not self.capture_once):
            raise ValueError("--camera_only_answer=true requires --mode=answer and --capture_once=true.")

        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = _resolve_pretrained_name_or_path(policy_path)
        if self.policy is None:
            raise ValueError("--policy.path is required for lerobot-vlm-inspect")

        ensure_smolvla_policy_config(self.policy)
        _disable_compile_for_inspection(self.policy)
        if self.device is None or not is_torch_device_available(self.device):
            resolved = self.policy.device or auto_select_torch_device().type
            logger.info("Resolved unavailable device '%s' to '%s'", self.device, resolved)
            self.device = resolved
        self.policy.device = self.device

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@dataclass
class InspectEvents:
    exit: bool = False
    capture: bool = False
    toggle_mode: bool = False
    set_baseline: bool = False
    selected_camera_index: int = 0


@dataclass
class FeatureSnapshotReference:
    tensors: dict[str, torch.Tensor]
    action_chunk: torch.Tensor

    def clone(self) -> "FeatureSnapshotReference":
        return FeatureSnapshotReference(
            tensors={key: value.detach().cpu().clone() for key, value in self.tensors.items()},
            action_chunk=self.action_chunk.detach().cpu().clone(),
        )


def ensure_smolvla_policy_config(policy_cfg: Any) -> None:
    if not isinstance(policy_cfg, SmolVLAConfig):
        policy_type = getattr(policy_cfg, "type", type(policy_cfg).__name__)
        raise ValueError(
            f"lerobot-vlm-inspect currently supports only SmolVLA policies; got policy type '{policy_type}'."
        )


def _disable_compile_for_inspection(policy_cfg: SmolVLAConfig) -> None:
    if getattr(policy_cfg, "compile_model", False):
        logger.info("Disabling policy.compile_model for lerobot-vlm-inspect.")
        policy_cfg.compile_model = False


def _resolve_pretrained_name_or_path(pretrained_name_or_path: str | Path) -> str | Path:
    path = Path(pretrained_name_or_path).expanduser()
    if path.exists():
        return path
    return str(pretrained_name_or_path)


def _policy_pretrained_name_or_path(policy_cfg: SmolVLAConfig) -> str | Path:
    if policy_cfg.pretrained_path is None:
        raise ValueError("--policy.path is required for lerobot-vlm-inspect")
    return _resolve_pretrained_name_or_path(policy_cfg.pretrained_path)


def ensure_required_state(policy: Any, observation: dict[str, Any]) -> None:
    if getattr(policy.config, "use_state", False) and OBS_STATE not in observation:
        raise ValueError(
            f"SmolVLA policy has use_state=True, but the processed observation does not contain `{OBS_STATE}`. "
            "The inspector does not synthesize zero or fake state."
        )


def _load_policy(policy_cfg: SmolVLAConfig) -> SmolVLAPolicy:
    policy_class = get_policy_class(policy_cfg.type)
    pretrained_name_or_path = _policy_pretrained_name_or_path(policy_cfg)
    if policy_cfg.use_peft:
        from peft import PeftConfig, PeftModel

        peft_path = pretrained_name_or_path
        peft_config = PeftConfig.from_pretrained(peft_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=policy_cfg
        )
        policy = PeftModel.from_pretrained(policy, peft_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(pretrained_name_or_path, config=policy_cfg)

    policy = policy.to(policy_cfg.device)
    policy.eval()
    if not isinstance(policy, SmolVLAPolicy):
        raise ValueError(f"Expected SmolVLAPolicy after loading; got {type(policy).__name__}.")
    return policy


def _robot_policy_observation_features(robot: Robot) -> dict[str, type | tuple]:
    return {
        key: value
        for key, value in robot.observation_features.items()
        if isinstance(value, tuple) or (value is float and key.endswith(".pos"))
    }


def _build_observation_features(
    robot: Robot, robot_observation_processor: RobotProcessorPipeline
) -> dict[str, dict]:
    return aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=_robot_policy_observation_features(robot)),
        use_videos=True,
    )


def _validate_visual_features(policy_cfg: SmolVLAConfig, robot: Robot, rename_map: dict[str, str]) -> None:
    if rename_map or not policy_cfg.input_features:
        return
    expected_visuals = {
        key for key, feature in policy_cfg.input_features.items() if feature.type == FeatureType.VISUAL
    }
    provided_visuals = {
        f"{OBS_IMAGES}.{key}" for key, value in robot.observation_features.items() if isinstance(value, tuple)
    }
    if expected_visuals and not (
        expected_visuals.issubset(provided_visuals) or provided_visuals.issubset(expected_visuals)
    ):
        raise ValueError(
            "Visual feature mismatch between policy and robot hardware.\n"
            f"Policy expects: {expected_visuals}\n"
            f"Robot provides: {provided_visuals}\n"
            "Use --rename_map when camera names differ but semantics match."
        )


def camera_keys_from_observation(observation: dict[str, Any]) -> list[str]:
    return [key for key in observation if key.startswith(f"{OBS_IMAGES}.")]


def _tensor_summary(value: Any) -> dict[str, Any]:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu()
    tensor_float = tensor.to(dtype=torch.float32)
    if tensor_float.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "l2": 0.0,
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "mean": float(tensor_float.mean().item()),
        "std": float(tensor_float.std(unbiased=False).item()) if tensor_float.numel() > 1 else 0.0,
        "min": float(tensor_float.min().item()),
        "max": float(tensor_float.max().item()),
        "l2": float(torch.linalg.vector_norm(tensor_float).item()),
    }


def _tensor_values(value: Any, max_items: int = 128) -> list[float]:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    flat = tensor.detach().cpu().flatten()[:max_items].to(dtype=torch.float32)
    return [float(v) for v in flat.tolist()]


def _tensor_delta(current: torch.Tensor, reference: torch.Tensor | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    current_cpu = current.detach().cpu().to(dtype=torch.float32)
    reference_cpu = reference.detach().cpu().to(dtype=torch.float32)
    current_dtype = str(current.detach().dtype).removeprefix("torch.")
    reference_dtype = str(reference.detach().dtype).removeprefix("torch.")
    if list(current_cpu.shape) != list(reference_cpu.shape):
        return {
            "available": False,
            "shape_compatible": False,
            "dtype_compatible": current_dtype == reference_dtype,
            "current_shape": list(current_cpu.shape),
            "reference_shape": list(reference_cpu.shape),
            "current_dtype": current_dtype,
            "reference_dtype": reference_dtype,
            "reason": f"shape mismatch current={list(current_cpu.shape)} reference={list(reference_cpu.shape)}",
        }
    delta = current_cpu - reference_cpu
    current_flat = current_cpu.flatten()
    reference_flat = reference_cpu.flatten()
    denominator = torch.linalg.vector_norm(current_flat) * torch.linalg.vector_norm(reference_flat)
    if denominator.item() == 0:
        cosine_distance = 0.0 if torch.equal(current_flat, reference_flat) else 1.0
    else:
        cosine_distance = float(1.0 - torch.dot(current_flat, reference_flat).item() / denominator.item())
    return {
        "available": True,
        "shape_compatible": True,
        "dtype_compatible": current_dtype == reference_dtype,
        "current_shape": list(current_cpu.shape),
        "reference_shape": list(reference_cpu.shape),
        "current_dtype": current_dtype,
        "reference_dtype": reference_dtype,
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "mean_abs": float(delta.abs().mean().item()) if delta.numel() else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "cosine_distance": cosine_distance,
    }


def _representation_delta(
    current: dict[str, torch.Tensor],
    reference: FeatureSnapshotReference | None,
) -> dict[str, Any]:
    if reference is None:
        return {}

    deltas = {}
    for key, current_tensor in current.items():
        reference_tensor = reference.tensors.get(key)
        if reference_tensor is None:
            deltas[key] = {
                "available": False,
                "shape_compatible": False,
                "dtype_compatible": False,
                "current_shape": list(current_tensor.shape),
                "reference_shape": None,
                "current_dtype": str(current_tensor.dtype).removeprefix("torch."),
                "reference_dtype": None,
                "reason": "reference tensor is missing",
            }
            continue
        deltas[key] = _tensor_delta(current_tensor, reference_tensor)
    return deltas


def _action_delta(current: torch.Tensor, reference: FeatureSnapshotReference | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    return _tensor_delta(current, reference.action_chunk)


def _safe_npz_array_name(name: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in name)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "tensor"


def _tensor_to_storage_array(tensor: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    tensor_cpu = tensor.detach().cpu().contiguous()
    original_dtype = str(tensor_cpu.dtype).removeprefix("torch.")
    if tensor_cpu.dtype == torch.bfloat16:
        try:
            storage = tensor_cpu.view(torch.uint16).numpy()
            return storage, {
                "dtype": original_dtype,
                "storage_dtype": str(storage.dtype),
                "encoding": "torch.bfloat16_bits",
            }
        except RuntimeError:
            storage = tensor_cpu.to(dtype=torch.float32).numpy()
            return storage, {
                "dtype": original_dtype,
                "storage_dtype": str(storage.dtype),
                "encoding": "float32_upcast_from_bfloat16",
            }

    storage = tensor_cpu.numpy()
    return storage, {
        "dtype": original_dtype,
        "storage_dtype": str(storage.dtype),
        "encoding": "native",
    }


def _artifact_display_path(path: Path, tensor_dir: Path) -> str:
    try:
        display_path = path.relative_to(tensor_dir.parent)
    except ValueError:
        display_path = path
    return display_path.as_posix()


def _save_tensor_artifact(
    tensor_dir: Path | None,
    *,
    filename: str,
    arrays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = {"path": None, "arrays": {}}
    storage_arrays = {}
    for array_name, array_info in arrays.items():
        tensor = array_info["tensor"]
        summary = _tensor_summary(tensor)
        storage_array, storage_info = _tensor_to_storage_array(tensor)
        storage_arrays[array_name] = storage_array
        metadata["arrays"][array_name] = {key: value for key, value in array_info.items() if key != "tensor"}
        metadata["arrays"][array_name].update(
            {
                "shape": summary["shape"],
                "dtype": summary["dtype"],
                "summary": summary,
                "storage": storage_info,
            }
        )

    if tensor_dir is not None:
        tensor_dir.mkdir(parents=True, exist_ok=True)
        path = tensor_dir / filename
        np.savez_compressed(path, **storage_arrays)
        metadata["path"] = _artifact_display_path(path, tensor_dir)
    return metadata


def _clone_reference_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in tensors.items()}


def _prepare_snapshot_for_policy(
    observation_frame: dict[str, Any],
    *,
    task: str,
    device: str,
    robot_type: str,
    preprocessor,
) -> dict[str, Any]:
    observation = copy(observation_frame)
    observation = prepare_observation_for_inference(observation, torch.device(device), task, robot_type)
    return preprocessor(observation)


def _denoise_step_with_suffix_out(
    flow: Any,
    *,
    prefix_pad_masks: torch.Tensor,
    past_key_values: dict[int, dict[str, torch.Tensor]],
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    trace: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    suffix_embs, suffix_pad_masks, suffix_att_masks = flow.embed_suffix(x_t, timestep)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    outputs_embeds, _ = flow.vlm_with_expert.forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=flow.config.use_cache,
        fill_kv_cache=False,
        trace=trace,
    )
    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -flow.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = flow.action_out_proj(suffix_out)
    return v_t, suffix_out


def _image_to_uint8_hwc(image: Any) -> np.ndarray:
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        max_value = 1.0 if array.max(initial=0) <= 1.0 else 255.0
        array = np.clip(array * (255.0 / max_value), 0, 255).astype(np.uint8)
    return array


def _aggregate_attention_by_group(attention_entries: list[dict[str, Any]]) -> dict[str, Any]:
    expert_entries = [
        entry
        for entry in attention_entries
        if entry.get("kind") == "expert_cross"
        or (entry.get("kind") == "self" and not entry.get("fill_kv_cache", False))
    ]
    accum: dict[str, list[float]] = {}
    for entry in expert_entries:
        for group, mass in entry.get("group_mass", {}).items():
            accum.setdefault(group, []).append(float(mass))
    return {
        group: {
            "mean_mass": float(np.mean(values)),
            "max_mass": float(np.max(values)),
            "num_layers": len(values),
        }
        for group, values in accum.items()
    }


def trace_smolvla_features(
    policy: SmolVLAPolicy,
    batch: dict[str, Any],
    *,
    tensor_dir: Path | None = None,
    baseline_reference: FeatureSnapshotReference | None = None,
    previous_reference: FeatureSnapshotReference | None = None,
) -> tuple[dict[str, Any], FeatureSnapshotReference]:
    ensure_required_state(policy, batch)
    policy.eval()
    batch = policy._prepare_batch(dict(batch))

    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        flow = policy.model
        image_keys = [key for key in policy.config.image_features if key in batch]
        image_summaries = []
        image_connector_arrays: dict[str, dict[str, Any]] = {}
        reference_tensors: dict[str, torch.Tensor] = {}
        token_groups: dict[str, tuple[int, int]] = {}
        offset = 0
        for image_index, (image, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            camera_key = (
                image_keys[image_index] if image_index < len(image_keys) else f"empty_camera_{image_index}"
            )
            group_start = offset
            if flow.add_image_special_tokens:
                offset += int(flow.global_image_start_token.numel())
            image_tokens = flow.vlm_with_expert.embed_image(image)
            token_start = offset
            offset += image_tokens.shape[1]
            if flow.add_image_special_tokens:
                offset += int(flow.image_end_token.numel())
            token_groups[f"image:{camera_key}"] = (group_start, offset)
            reference_key = f"image_connector_tokens:{camera_key}"
            array_name = _safe_npz_array_name(reference_key)
            reference_tensors[reference_key] = image_tokens.detach().cpu().clone()
            image_connector_arrays[array_name] = {
                "tensor": image_tokens,
                "key": reference_key,
                "camera_key": camera_key,
                "token_group": [group_start, offset],
                "connector_token_range": [token_start, token_start + image_tokens.shape[1]],
                "mask": [bool(v) for v in img_mask.detach().cpu().flatten().tolist()],
            }
            image_summaries.append(
                {
                    "key": camera_key,
                    "token_start": token_start,
                    "token_end": token_start + image_tokens.shape[1],
                    "mask": [bool(v) for v in img_mask.detach().cpu().flatten().tolist()],
                    "connector_tokens": _tensor_summary(image_tokens),
                }
            )

        language_start = offset
        offset += lang_tokens.shape[1]
        token_groups["language"] = (language_start, offset)

        state_summary = None
        state_embedding_summary = None
        if state is not None:
            state_emb = flow.state_proj(state)
            if state_emb.ndim == 2:
                state_emb = state_emb[:, None, :]
            state_start = offset
            offset += state_emb.shape[1]
            token_groups["state"] = (state_start, offset)
            state_summary = {"values": _tensor_values(state), **_tensor_summary(state)}
            state_embedding_summary = _tensor_summary(state_emb)

        trace_runtime: dict[str, Any] = {"token_groups": token_groups}
        prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_outputs, past_key_values = flow.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy.config.use_cache,
            fill_kv_cache=True,
            trace=trace_runtime,
        )
        prefix_output = prefix_outputs[0]
        reference_tensors["prefix_output"] = prefix_output.detach().cpu().clone()

        prefix_kv_arrays: dict[str, dict[str, Any]] = {}
        if past_key_values is not None:
            for layer_idx in sorted(past_key_values):
                layer_cache = past_key_values[layer_idx]
                for cache_name in ("key_states", "value_states"):
                    cache_tensor = layer_cache[cache_name]
                    reference_key = f"prefix_kv_cache:layer_{int(layer_idx):04d}:{cache_name}"
                    array_name = _safe_npz_array_name(reference_key)
                    reference_tensors[reference_key] = cache_tensor.detach().cpu().clone()
                    prefix_kv_arrays[array_name] = {
                        "tensor": cache_tensor,
                        "key": reference_key,
                        "layer": int(layer_idx),
                        "cache": cache_name,
                        "token_groups": {key: list(value) for key, value in token_groups.items()},
                    }

        if state is not None:
            bsize = state.shape[0]
            device = state.device
        else:
            bsize = lang_tokens.shape[0]
            device = lang_tokens.device

        action_shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
        noise = flow.sample_noise(action_shape, device)
        num_steps = policy.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        suffix_out_steps = []
        for step in range(num_steps):
            time_value = 1.0 + step * dt
            time_tensor = torch.tensor(time_value, dtype=torch.float32, device=device).expand(bsize)
            step_suffix_out = None

            def denoise_step_partial_call(
                input_x_t: torch.Tensor, current_timestep: torch.Tensor = time_tensor
            ) -> torch.Tensor:
                nonlocal step_suffix_out
                v_t, suffix_out = _denoise_step_with_suffix_out(
                    flow,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    trace=trace_runtime,
                )
                step_suffix_out = suffix_out
                return v_t

            if flow._rtc_enabled():
                v_t = flow.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=None,
                    inference_delay=None,
                    time=time_value,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=None,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t
            if step_suffix_out is not None:
                suffix_out_steps.append(step_suffix_out.detach().cpu().clone())

            if flow.rtc_processor is not None and flow.rtc_processor.is_debug_enabled():
                flow.rtc_processor.track(time=time_value, x_t=x_t, v_t=v_t)

        if suffix_out_steps:
            suffix_out_steps_tensor = torch.stack(suffix_out_steps, dim=0)
        else:
            suffix_out_steps_tensor = torch.empty(0, device=device)
        reference_tensors["suffix_out_steps"] = suffix_out_steps_tensor.detach().cpu().clone()

        action_chunk = x_t[:, :, : policy.config.action_feature.shape[0]]
        if policy.config.adapt_to_pi_aloha:
            action_chunk = policy._pi_aloha_encode_actions(action_chunk)
        action_chunk = action_chunk.detach().cpu()

    language_mask = lang_masks.detach().cpu().to(dtype=torch.bool)
    language_tokens = lang_tokens.detach().cpu()
    token_groups_json = {key: list(value) for key, value in token_groups.items()}
    tensor_artifacts = {
        "image_connector_tokens": _save_tensor_artifact(
            tensor_dir,
            filename="image_connector_tokens.npz",
            arrays=image_connector_arrays,
        ),
        "prefix_output": _save_tensor_artifact(
            tensor_dir,
            filename="prefix_output.npz",
            arrays={
                "prefix_output": {
                    "tensor": prefix_output,
                    "key": "prefix_output",
                    "token_groups": token_groups_json,
                }
            },
        ),
        "prefix_kv_cache": _save_tensor_artifact(
            tensor_dir,
            filename="prefix_kv_cache.npz",
            arrays=prefix_kv_arrays,
        ),
        "suffix_out_steps": _save_tensor_artifact(
            tensor_dir,
            filename="suffix_out_steps.npz",
            arrays={
                "suffix_out_steps": {
                    "tensor": suffix_out_steps_tensor,
                    "key": "suffix_out_steps",
                    "step_count": len(suffix_out_steps),
                    "action_token_range": [0, policy.config.chunk_size],
                }
            },
        ),
    }
    reference_tensors = _clone_reference_tensors(reference_tensors)
    reference = FeatureSnapshotReference(tensors=reference_tensors, action_chunk=action_chunk.clone())
    trace = {
        "schema_version": 1,
        "mode": "feature",
        "tensor_artifacts": tensor_artifacts,
        "image_tokens": image_summaries,
        "language": {
            "token_count": int(language_mask.sum().item()),
            "sequence_length": int(language_tokens.shape[1]),
            "token_ids": [int(v) for v in language_tokens[0, : min(language_tokens.shape[1], 128)].tolist()],
        },
        "state": state_summary,
        "state_embedding": state_embedding_summary,
        "token_groups": token_groups_json,
        "prefix_hidden_states": trace_runtime.get("prefix_hidden_states", [])[:16],
        "expert_attention": _aggregate_attention_by_group(trace_runtime.get("attention", [])),
        "attention_calls": trace_runtime.get("attention", []),
        "action_chunk": _tensor_summary(action_chunk),
        "action_delta": {
            "baseline": _action_delta(action_chunk, baseline_reference),
            "previous": _action_delta(action_chunk, previous_reference),
        },
        "representation_delta": {
            "baseline": _representation_delta(reference_tensors, baseline_reference),
            "previous": _representation_delta(reference_tensors, previous_reference),
        },
    }
    return trace, reference


def answer_with_smolvlm(
    policy: SmolVLAPolicy,
    image: Any,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int = 64,
    generate_answer: bool = True,
) -> dict[str, Any]:
    vlm_with_expert = getattr(policy.model, "vlm_with_expert", None)
    if vlm_with_expert is None:
        return {"available": False, "answer": None, "reason": "SmolVLA VLM module is unavailable."}

    vlm = getattr(vlm_with_expert, "vlm", None)
    processor = getattr(vlm_with_expert, "processor", None)
    if vlm is None or processor is None or not hasattr(vlm, "generate"):
        return {"available": False, "answer": None, "reason": "Loaded VLM is not generation-capable."}

    text_model = getattr(vlm_with_expert.get_vlm_model(), "text_model", None)
    loaded_layers = len(getattr(text_model, "layers", [])) if text_model is not None else None
    configured_layers = getattr(getattr(vlm, "config", None), "text_config", None)
    full_layers = getattr(configured_layers, "num_hidden_layers", None)
    if full_layers is not None and loaded_layers is not None and loaded_layers < full_layers:
        return {
            "available": False,
            "answer": None,
            "reason": f"VLM text layers are truncated ({loaded_layers}/{full_layers}); generation disabled.",
        }

    try:
        image_hwc = _image_to_uint8_hwc(image)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            text = prompt
        inputs = processor(text=text, images=[image_hwc], return_tensors="pt")
        inputs = inputs.to(device) if hasattr(inputs, "to") else {k: v.to(device) for k, v in inputs.items()}
        input_len = _model_input_token_count(inputs)
        yes_no = _score_yes_no_from_next_token(vlm, processor, inputs)
        if generate_answer:
            with torch.inference_mode():
                output_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens)
            answer_ids, prompt_echo_removed = _generated_answer_ids(output_ids, input_len)
            answer = processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
            generated_token_count = int(answer_ids.shape[-1]) if hasattr(answer_ids, "shape") else None
        else:
            answer = yes_no.get("label") if yes_no.get("available") else None
            generated_token_count = 0
            prompt_echo_removed = False
        return {
            "available": bool(answer is not None or generate_answer),
            "answer": answer,
            "reason": None if answer is not None or generate_answer else yes_no.get("reason"),
            "input_token_count": input_len,
            "generated_token_count": generated_token_count,
            "prompt_echo_removed": prompt_echo_removed,
            "generation_skipped": not generate_answer,
            "yes_no": yes_no,
        }
    except Exception as exc:
        return {"available": False, "answer": None, "reason": str(exc)}


def _model_input_value(inputs: Any, key: str) -> Any | None:
    try:
        return inputs[key]
    except (AttributeError, KeyError, TypeError):
        return getattr(inputs, key, None)


def _model_input_token_count(inputs: Any) -> int:
    input_ids = _model_input_value(inputs, "input_ids")
    if input_ids is None or not hasattr(input_ids, "shape") or len(input_ids.shape) == 0:
        return 0
    return int(input_ids.shape[-1])


def _generated_answer_ids(output_ids: torch.Tensor, input_len: int) -> tuple[torch.Tensor, bool]:
    if input_len <= 0 or not hasattr(output_ids, "shape") or output_ids.shape[-1] < input_len:
        return output_ids, False
    return output_ids[:, input_len:], True


def _tokenizer_input_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
    except Exception:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return []

    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _candidate_first_token_ids(tokenizer: Any, candidates: tuple[str, ...]) -> list[int]:
    token_ids = []
    for candidate in candidates:
        ids = _tokenizer_input_ids(tokenizer, candidate)
        if ids:
            token_ids.append(ids[0])
    return sorted(set(token_ids))


def _score_yes_no_from_next_token(vlm: Any, processor: Any, inputs: Any) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return {"available": False, "reason": "Processor has no tokenizer."}

    yes_ids = _candidate_first_token_ids(tokenizer, ("yes", "Yes", " YES", " yes"))
    no_ids = _candidate_first_token_ids(tokenizer, ("no", "No", " NO", " no"))
    if not yes_ids or not no_ids:
        return {"available": False, "reason": "Could not encode yes/no candidate tokens."}

    try:
        with torch.inference_mode():
            outputs = vlm(**inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            return {"available": False, "reason": "VLM forward output has no logits."}

        next_logits = logits[:, -1, :].float()
        vocab_size = next_logits.shape[-1]
        yes_ids = [token_id for token_id in yes_ids if token_id < vocab_size]
        no_ids = [token_id for token_id in no_ids if token_id < vocab_size]
        if not yes_ids or not no_ids:
            return {"available": False, "reason": "Encoded yes/no token ids exceed logits vocabulary."}

        yes_score = torch.logsumexp(next_logits[0, yes_ids], dim=0)
        no_score = torch.logsumexp(next_logits[0, no_ids], dim=0)
        yes_no_probs = torch.softmax(torch.stack([yes_score, no_score]), dim=0)
        yes_probability = float(yes_no_probs[0].item())
        no_probability = float(yes_no_probs[1].item())
        margin = float((yes_score - no_score).item())
        return {
            "available": True,
            "label": "yes" if margin >= 0 else "no",
            "yes_probability": yes_probability,
            "no_probability": no_probability,
            "yes_logit_score": float(yes_score.item()),
            "no_logit_score": float(no_score.item()),
            "margin": margin,
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
            "note": "Probability is normalized only over yes/no first-token candidates.",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def _save_snapshot_arrays(snapshot_dir: Path, observation: dict[str, Any]) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for key in camera_keys_from_observation(observation):
        filename = key.removeprefix(f"{OBS_IMAGES}.").replace(".", "_")
        np.save(snapshot_dir / f"{filename}.npy", np.asarray(observation[key]))


def _log_trace_to_rerun(trace: dict[str, Any], snapshot_index: int) -> None:
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    action = trace.get("action_chunk", {})
    if "l2" in action:
        rr.log("vlm_inspect/action_chunk_l2", rr.Scalars(float(action["l2"])))
    for group, summary in trace.get("expert_attention", {}).items():
        rr.log(
            f"vlm_inspect/attention/{group.replace(':', '_')}",
            rr.Scalars(float(summary["mean_mass"])),
        )
    for delta_name, delta in trace.get("action_delta", {}).items():
        if delta and delta.get("available") and "l2" in delta:
            rr.log(f"vlm_inspect/action_delta/{delta_name}_l2", rr.Scalars(float(delta["l2"])))
    for reference_name, deltas in trace.get("representation_delta", {}).items():
        for tensor_name, delta in deltas.items():
            if delta and delta.get("available") and "l2" in delta:
                safe_tensor_name = _safe_npz_array_name(tensor_name)
                rr.log(
                    f"vlm_inspect/representation_delta/{reference_name}/{safe_tensor_name}_l2",
                    rr.Scalars(float(delta["l2"])),
                )
    rr.log("vlm_inspect/snapshot_index", rr.Scalars(float(snapshot_index)))


def _start_keyboard_listener(events: InspectEvents):
    if is_headless():
        logger.warning("Headless environment detected; keyboard shortcuts are unavailable.")
        return None

    from pynput import keyboard

    def on_press(key):
        if key == keyboard.Key.space:
            events.capture = True
            return
        try:
            char = key.char
        except AttributeError:
            return
        if char == "q":
            events.exit = True
        elif char == "m":
            events.toggle_mode = True
        elif char == "b":
            events.set_baseline = True
        elif char and char.isdigit() and char != "0":
            events.selected_camera_index = int(char) - 1

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _print_camera_selection(camera_keys: list[str], selected_index: int) -> None:
    if not camera_keys:
        print("No camera keys found in the current observation.")
        return
    selected = camera_keys[min(selected_index, len(camera_keys) - 1)]
    print("Cameras: " + ", ".join(f"{i + 1}:{key}" for i, key in enumerate(camera_keys)))
    print(f"Selected camera: {selected}")


def _camera_only_entries(robot: Robot) -> list[tuple[str, Any]]:
    if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
        entries = []
        for prefix, arm in (("left_", robot.left_arm), ("right_", robot.right_arm)):
            entries.extend((f"{OBS_IMAGES}.{prefix}{key}", camera) for key, camera in arm.cameras.items())
        return entries
    return [(f"{OBS_IMAGES}.{key}", camera) for key, camera in getattr(robot, "cameras", {}).items()]


def _run_camera_only_answer_capture(
    *,
    cfg: VLMInspectConfig,
    policy: SmolVLAPolicy,
    robot: Robot,
    run_dir: Path,
) -> None:
    if cfg.question is None:
        raise ValueError("--question is required for camera-only answer capture.")

    camera_entries = _camera_only_entries(robot)
    if not camera_entries:
        raise ValueError("No cameras configured on robot.")

    connected_cameras = []
    snapshot_index = 1
    snapshot_dir = run_dir / f"snapshot_{snapshot_index:04d}"
    try:
        observation_frame = {}
        selected_key, selected_camera = camera_entries[0]
        _print_camera_selection([key for key, _camera in camera_entries], 0)
        selected_camera.connect()
        connected_cameras.append(selected_camera)
        observation_frame[selected_key] = selected_camera.read()

        _save_snapshot_arrays(snapshot_dir, observation_frame)
        answer = answer_with_smolvlm(
            policy,
            observation_frame[selected_key],
            cfg.question,
            device=cfg.device,
            max_new_tokens=cfg.answer_max_new_tokens,
            generate_answer=not cfg.answer_score_only,
        )
        answer["camera"] = selected_key
        _write_json(snapshot_dir / "answer.json", answer)
        if answer.get("available"):
            print(f"Answer: {answer['answer']}")
            yes_no = answer.get("yes_no", {})
            if yes_no.get("available"):
                print(
                    "Yes/no: "
                    f"{yes_no['label']} "
                    f"(yes={yes_no['yes_probability']:.3f}, "
                    f"no={yes_no['no_probability']:.3f}, "
                    f"margin={yes_no['margin']:.3f})"
                )
        else:
            print(f"Answer unavailable: {answer.get('reason')}")
    finally:
        for camera in reversed(connected_cameras):
            camera.disconnect()


def _run_feature_snapshot(
    *,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    observation_frame: dict[str, Any],
    task: str,
    device: str,
    robot_type: str,
    tensor_dir: Path | None,
    baseline_reference: FeatureSnapshotReference | None,
    previous_reference: FeatureSnapshotReference | None,
) -> tuple[dict[str, Any], FeatureSnapshotReference, torch.Tensor | None]:
    preprocessed = _prepare_snapshot_for_policy(
        observation_frame,
        task=task,
        device=device,
        robot_type=robot_type,
        preprocessor=preprocessor,
    )
    trace, reference = trace_smolvla_features(
        policy,
        preprocessed,
        tensor_dir=tensor_dir,
        baseline_reference=baseline_reference,
        previous_reference=previous_reference,
    )
    try:
        postprocessed_action = postprocessor(reference.action_chunk.to(device))
        trace["postprocessed_action_chunk"] = _tensor_summary(postprocessed_action)
    except Exception as exc:
        trace["postprocessed_action_chunk"] = {"available": False, "reason": str(exc)}
        postprocessed_action = None
    trace["task"] = task
    return trace, reference, postprocessed_action


@parser.wrap()
def vlm_inspect(cfg: VLMInspectConfig) -> None:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    policy_cfg = cfg.policy
    ensure_smolvla_policy_config(policy_cfg)
    pretrained_name_or_path = _policy_pretrained_name_or_path(policy_cfg)
    policy = _load_policy(policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_name_or_path),
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    if cfg.display_data:
        init_rerun(session_name="vlm_inspect", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    _teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    robot = make_robot_from_config(cfg.robot)
    _validate_visual_features(policy_cfg, robot, cfg.rename_map)

    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(cfg))

    if cfg.camera_only_answer:
        _run_camera_only_answer_capture(cfg=cfg, policy=policy, robot=robot, run_dir=run_dir)
        logger.info("VLM inspection finished. Outputs: %s", run_dir)
        return

    robot.connect()
    observation_features = _build_observation_features(robot, robot_observation_processor)

    events = InspectEvents()
    listener = _start_keyboard_listener(events)
    mode = cfg.mode
    baseline_reference = None
    baseline_next_capture = False
    previous_reference = None
    snapshot_index = 0
    control_interval = 1 / cfg.fps

    print("Controls: 1..9 select camera, space capture, m mode, b baseline, q quit.")
    try:
        while not events.exit:
            loop_start = time.perf_counter()
            raw_observation = robot.get_observation()
            processed_observation = robot_observation_processor(raw_observation)
            observation_frame = build_dataset_frame(
                observation_features, processed_observation, prefix=OBS_STR
            )
            camera_keys = camera_keys_from_observation(observation_frame)
            if events.selected_camera_index >= len(camera_keys):
                events.selected_camera_index = max(0, len(camera_keys) - 1)

            if cfg.display_data:
                log_rerun_data(
                    observation=observation_frame,
                    action=None,
                    compress_images=display_compressed_images,
                )

            if events.toggle_mode:
                mode = "answer" if mode == "feature" else "feature"
                events.toggle_mode = False
                print(f"Mode: {mode}")

            if events.set_baseline:
                if previous_reference is None:
                    print("No feature snapshot yet; next feature capture will become the baseline.")
                    baseline_next_capture = True
                else:
                    baseline_reference = previous_reference.clone()
                    print("Baseline set to previous feature snapshot.")
                events.set_baseline = False

            if cfg.capture_once and snapshot_index == 0:
                events.capture = True

            if events.capture:
                events.capture = False
                snapshot = copy(observation_frame)
                _print_camera_selection(camera_keys, events.selected_camera_index)
                task = cfg.question if cfg.capture_once else input("Task / question: ")
                if task is None:
                    raise ValueError("Task / question is required.")
                snapshot_index += 1
                snapshot_dir = run_dir / f"snapshot_{snapshot_index:04d}"
                _save_snapshot_arrays(snapshot_dir, snapshot)

                if mode == "feature":
                    autocast_ctx = (
                        torch.autocast(device_type=torch.device(cfg.device).type)
                        if torch.device(cfg.device).type == "cuda" and policy.config.use_amp
                        else nullcontext()
                    )
                    with autocast_ctx:
                        trace, reference, _postprocessed_action = _run_feature_snapshot(
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            observation_frame=snapshot,
                            task=task,
                            device=cfg.device,
                            robot_type=robot.name,
                            tensor_dir=snapshot_dir / "tensors",
                            baseline_reference=baseline_reference,
                            previous_reference=previous_reference,
                        )
                    if baseline_reference is None and previous_reference is None:
                        print(
                            "First feature snapshot captured. Press b after a later capture to set a baseline."
                        )
                    previous_reference = reference.clone()
                    if baseline_next_capture:
                        baseline_reference = reference.clone()
                        baseline_next_capture = False
                        print("Baseline set to current feature snapshot.")
                    _write_json(snapshot_dir / "trace.json", trace)
                    if cfg.display_data:
                        _log_trace_to_rerun(trace, snapshot_index)
                    print(f"Feature trace saved: {snapshot_dir / 'trace.json'}")
                else:
                    if not camera_keys:
                        answer = {"available": False, "answer": None, "reason": "No camera image available."}
                    else:
                        selected_key = camera_keys[events.selected_camera_index]
                        answer = answer_with_smolvlm(
                            policy,
                            snapshot[selected_key],
                            task,
                            device=cfg.device,
                            max_new_tokens=cfg.answer_max_new_tokens,
                            generate_answer=not cfg.answer_score_only,
                        )
                        answer["camera"] = selected_key
                    _write_json(snapshot_dir / "answer.json", answer)
                    if answer.get("available"):
                        print(f"Answer: {answer['answer']}")
                    else:
                        print(f"Answer unavailable: {answer.get('reason')}")

                if cfg.capture_once:
                    events.exit = True

            dt = time.perf_counter() - loop_start
            precise_sleep(max(control_interval - dt, 0.0))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if listener:
            listener.stop()
        if robot.is_connected:
            robot.disconnect()
        logger.info("VLM inspection finished. Outputs: %s", run_dir)


def main() -> None:
    register_third_party_plugins()
    vlm_inspect()


if __name__ == "__main__":
    main()
