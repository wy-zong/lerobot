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

    def __post_init__(self) -> None:
        if self.robot is None:
            raise ValueError("--robot.type is required for lerobot-vlm-inspect")
        if self.mode not in {"feature", "answer"}:
            raise ValueError(f"--mode must be either 'feature' or 'answer', got '{self.mode}'.")

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


def _action_delta(current: torch.Tensor, reference: torch.Tensor | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    current_cpu = current.detach().cpu().to(dtype=torch.float32)
    reference_cpu = reference.detach().cpu().to(dtype=torch.float32)
    if list(current_cpu.shape) != list(reference_cpu.shape):
        return {
            "available": False,
            "reason": f"shape mismatch current={list(current_cpu.shape)} reference={list(reference_cpu.shape)}",
        }
    delta = current_cpu - reference_cpu
    return {"available": True, **_tensor_summary(delta), "mean_abs": float(delta.abs().mean().item())}


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
    baseline_action: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    ensure_required_state(policy, batch)
    policy.eval()
    batch = policy._prepare_batch(dict(batch))

    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        flow = policy.model
        image_keys = [key for key in policy.config.image_features if key in batch]
        image_summaries = []
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
        _, past_key_values = flow.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy.config.use_cache,
            fill_kv_cache=True,
            trace=trace_runtime,
        )

        action_shape = (
            lang_tokens.shape[0],
            policy.config.chunk_size,
            policy.config.max_action_dim,
        )
        noise = flow.sample_noise(action_shape, lang_tokens.device)
        timestep = torch.ones(lang_tokens.shape[0], dtype=torch.float32, device=lang_tokens.device)
        _ = flow.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=noise,
            timestep=timestep,
            trace=trace_runtime,
        )

        action_chunk = policy.predict_action_chunk(batch)

    language_mask = lang_masks.detach().cpu().to(dtype=torch.bool)
    language_tokens = lang_tokens.detach().cpu()
    trace = {
        "schema_version": 1,
        "mode": "feature",
        "image_tokens": image_summaries,
        "language": {
            "token_count": int(language_mask.sum().item()),
            "sequence_length": int(language_tokens.shape[1]),
            "token_ids": [int(v) for v in language_tokens[0, : min(language_tokens.shape[1], 128)].tolist()],
        },
        "state": state_summary,
        "state_embedding": state_embedding_summary,
        "token_groups": {key: list(value) for key, value in token_groups.items()},
        "prefix_hidden_states": trace_runtime.get("prefix_hidden_states", [])[:16],
        "expert_attention": _aggregate_attention_by_group(trace_runtime.get("attention", [])),
        "attention_calls": trace_runtime.get("attention", []),
        "action_chunk": _tensor_summary(action_chunk),
        "action_delta": {
            "baseline": _action_delta(action_chunk, baseline_action),
            "previous": _action_delta(action_chunk, previous_action),
        },
    }
    return trace, action_chunk.detach().cpu()


def answer_with_smolvlm(
    policy: SmolVLAPolicy,
    image: Any,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int = 64,
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
        with torch.inference_mode():
            output_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[-1] if isinstance(inputs, dict) and "input_ids" in inputs else 0
        answer_ids = output_ids[:, input_len:] if input_len else output_ids
        answer = processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        return {"available": True, "answer": answer, "reason": None}
    except Exception as exc:
        return {"available": False, "answer": None, "reason": str(exc)}


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


def _run_feature_snapshot(
    *,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    observation_frame: dict[str, Any],
    task: str,
    device: str,
    robot_type: str,
    baseline_action: torch.Tensor | None,
    previous_action: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:
    preprocessed = _prepare_snapshot_for_policy(
        observation_frame,
        task=task,
        device=device,
        robot_type=robot_type,
        preprocessor=preprocessor,
    )
    trace, action_chunk = trace_smolvla_features(
        policy,
        preprocessed,
        baseline_action=baseline_action,
        previous_action=previous_action,
    )
    try:
        postprocessed_action = postprocessor(action_chunk.to(device))
        trace["postprocessed_action_chunk"] = _tensor_summary(postprocessed_action)
    except Exception as exc:
        trace["postprocessed_action_chunk"] = {"available": False, "reason": str(exc)}
        postprocessed_action = None
    trace["task"] = task
    return trace, action_chunk, postprocessed_action


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
    robot.connect()
    _validate_visual_features(policy_cfg, robot, cfg.rename_map)
    observation_features = _build_observation_features(robot, robot_observation_processor)

    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(cfg))

    events = InspectEvents()
    listener = _start_keyboard_listener(events)
    mode = cfg.mode
    baseline_action = None
    baseline_next_capture = False
    previous_action = None
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
                if previous_action is None:
                    print("No feature snapshot yet; next feature capture will become the baseline.")
                    baseline_next_capture = True
                else:
                    baseline_action = previous_action.clone()
                    print("Baseline set to previous feature snapshot.")
                events.set_baseline = False

            if events.capture:
                events.capture = False
                snapshot = copy(observation_frame)
                _print_camera_selection(camera_keys, events.selected_camera_index)
                task = input("Task / question: ")
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
                        trace, action_chunk, _postprocessed_action = _run_feature_snapshot(
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            observation_frame=snapshot,
                            task=task,
                            device=cfg.device,
                            robot_type=robot.name,
                            baseline_action=baseline_action,
                            previous_action=previous_action,
                        )
                    if baseline_action is None and previous_action is None:
                        print(
                            "First feature snapshot captured. Press b after a later capture to set a baseline."
                        )
                    previous_action = action_chunk.clone()
                    if baseline_next_capture:
                        baseline_action = action_chunk.clone()
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
                        )
                        answer["camera"] = selected_key
                    _write_json(snapshot_dir / "answer.json", answer)
                    if answer.get("available"):
                        print(f"Answer: {answer['answer']}")
                    else:
                        print(f"Answer unavailable: {answer.get('reason')}")

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
