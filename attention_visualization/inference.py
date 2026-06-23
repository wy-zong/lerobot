"""Offline sampled SmolVLA inference with expert cross-attention tracing."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from .core import (
    aggregate_attention_trace,
    load_jsonl_records,
    map_model_cameras,
)


def model_image_keys(config: Any) -> list[str]:
    return list(config.image_features)


def _language_token_groups(batch: dict[str, Any]):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK

    attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().tolist()
    return {"language": [[0, int(sum(attention_mask))]]}


def prepare_model_observation(
    item: dict[str, Any],
    *,
    camera_mapping: dict[str, str],
    max_state_dim: int | None,
) -> dict[str, Any]:
    import torch

    observation: dict[str, Any] = {"task": item["task"]}
    for model_key, source_key in camera_mapping.items():
        observation[model_key] = item[source_key]
    if max_state_dim is not None:
        state = item["observation.state"]
        if state.shape[-1] > max_state_dim:
            raise ValueError(
                f"Dataset state dimension {state.shape[-1]} exceeds model maximum {max_state_dim}"
            )
        # Some historical checkpoints declare a six-dimensional input feature
        # while their saved normalizer and state projection were trained with
        # the full 12-dimensional bi-arm state. Preserve the observed state and
        # let SmolVLA pad it to max_state_dim after normalization.
        observation["observation.state"] = state.clone()
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in observation.items()
    }


def load_local_model(
    model_path: Path,
    vlm_path: Path,
    device: str,
):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(model_path)
    config.device = device
    config.compile_model = False
    config.vlm_model_name = str(vlm_path)
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": str(vlm_path)},
        },
    )
    return config, policy, preprocessor


def trace_one_frame(
    *,
    policy: Any,
    preprocessor: Any,
    config: Any,
    item: dict[str, Any],
    camera_mapping: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    import torch

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    observation = prepare_model_observation(
        item,
        camera_mapping=camera_mapping,
        max_state_dim=config.max_state_dim if config.use_state else None,
    )
    batch = preprocessor(observation)
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    language_tokens = batch[OBS_LANGUAGE_TOKENS]
    language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    source_cameras = [camera_mapping[key] for key in model_image_keys(config)]
    trace = {
        "image_group_names": source_cameras,
        # For discrete-state models the complete `Task: ..., State: ...; Action:`
        # prompt is language input. There is no independent state token to
        # attribute, so the State span must not be split into a synthetic group.
        "language_token_groups": _language_token_groups(batch),
    }

    generator = torch.Generator(device=config.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        (1, config.chunk_size, config.max_action_dim),
        dtype=torch.float32,
        device=config.device,
        generator=generator,
    )
    with torch.inference_mode():
        policy.model.sample_actions(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            noise=noise,
            trace=trace,
        )
    aggregate = aggregate_attention_trace(trace, source_cameras)
    denoise_steps = sorted({int(entry["denoise_step"]) for entry in aggregate["layer_step_details"]})
    if denoise_steps != list(range(config.num_steps)):
        raise ValueError(f"Trace captured denoise steps {denoise_steps}, expected 0..{config.num_steps - 1}")
    return {
        "token_ranges": trace["token_groups"],
        "state_attention_available": "state" in trace["token_groups"],
        "state_input_mode": (
            "discrete_values_in_language" if config.discrete_state_in_language else "independent_token"
        ),
        "prefix_length": trace["prefix_length"],
        "valid_prefix_tokens": trace["valid_prefix_tokens"],
        **aggregate,
    }


def sample_model_attention(
    *,
    model_id: str,
    model_path: Path,
    model_revision: str | None,
    vlm_path: Path,
    dataset: Any,
    dataset_camera_keys: list[str],
    sample_frames: list[int],
    episode_index: int,
    source_fps: float,
    output_path: Path,
    device: str,
    seed: int,
    resume: bool,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Load one model, append missing samples, then release all GPU allocations."""

    existing = load_jsonl_records(output_path)
    if existing and not resume:
        raise FileExistsError(f"Attention samples already exist: {output_path}; use --resume or --force")
    completed = {int(record["frame_index"]) for record in existing}
    missing_frames = [frame for frame in sample_frames if frame not in completed]
    if max_samples is not None:
        missing_frames = missing_frames[:max_samples]
    if not missing_frames:
        return {"camera_mapping": existing[0].get("camera_mapping", {}) if existing else {}}

    config = policy = preprocessor = None
    try:
        config, policy, preprocessor = load_local_model(model_path, vlm_path, device)
        model_keys = model_image_keys(config)
        camera_mapping = map_model_cameras(dataset_camera_keys, model_keys)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", buffering=1) as stream:
            for sample_number, frame_index in enumerate(missing_frames, start=1):
                item = dataset[frame_index]
                traced = trace_one_frame(
                    policy=policy,
                    preprocessor=preprocessor,
                    config=config,
                    item=item,
                    camera_mapping=camera_mapping,
                    seed=seed + frame_index,
                )
                raw_state = item["observation.state"].detach().cpu().reshape(-1).tolist()
                record = {
                    "schema_version": 1,
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame_index / source_fps,
                    "task": item["task"],
                    "state_values": raw_state,
                    "camera_mapping": camera_mapping,
                    "inference_seed": seed + frame_index,
                    **traced,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(
                    f"{model_id}: sampled {sample_number}/{len(missing_frames)} (frame {frame_index})",
                    flush=True,
                )
        return {"camera_mapping": camera_mapping}
    finally:
        del preprocessor, policy, config
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
