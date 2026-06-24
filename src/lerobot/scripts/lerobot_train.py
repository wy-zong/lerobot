#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import logging
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import (
    EpisodeAwareSampler,
    LeRobotDatasetMetadata,
    apply_intervention_only_data,
    compute_intervention_only_data,
    make_dataset,
    selected_episode_boundaries,
)
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all

SARM_VALIDATION_METRIC_KEYS = (
    "total_loss",
    "sparse_stage_loss",
    "sparse_subtask_loss",
    "dense_stage_loss",
    "dense_subtask_loss",
)


def _is_sarm_validation_requested(cfg: TrainPipelineConfig) -> bool:
    return (
        cfg.validation.enable
        and cfg.is_reward_model_training
        and cfg.reward_model is not None
        and cfg.reward_model.type == "sarm"
    )


def resolve_sarm_validation_episodes(
    candidate_episodes: list[int],
    *,
    ratio: float,
    split: str,
    validation_episodes: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Resolve train/validation episode lists without reindexing the dataset."""
    if split != "tail":
        raise ValueError(f"validation.split must be 'tail', got {split!r}")

    if len(candidate_episodes) < 2:
        return list(candidate_episodes), []

    if validation_episodes is not None:
        val_episodes = list(validation_episodes)
        val_set = set(val_episodes)
        train_episodes = [ep for ep in candidate_episodes if ep not in val_set]
        if not val_episodes:
            return list(candidate_episodes), []
        if not train_episodes:
            raise ValueError("validation.episodes cannot consume all candidate training episodes.")
        return train_episodes, val_episodes

    val_count = min(max(1, math.ceil(len(candidate_episodes) * ratio)), len(candidate_episodes) - 1)
    split_at = len(candidate_episodes) - val_count
    return candidate_episodes[:split_at], candidate_episodes[split_at:]


def _validate_episode_range(episodes: list[int], total_episodes: int, name: str) -> None:
    out_of_range = [ep for ep in episodes if ep >= total_episodes]
    if out_of_range:
        raise ValueError(
            f"{name} contains episode indices outside the dataset range [0, {total_episodes - 1}]: "
            f"{out_of_range}"
        )


def _resolve_sarm_validation_split(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator",
    is_main_process: bool,
) -> tuple[bool, list[int]]:
    """Resolve and broadcast SARM validation episodes before the training dataset is created."""
    if not cfg.validation.enable:
        return False, []

    if not _is_sarm_validation_requested(cfg):
        if is_main_process:
            logging.warning("validation.enable=true is only supported for SARM reward model training; skipping.")
        return False, []

    from accelerate.utils import broadcast_object_list

    payload = [None]
    if is_main_process:
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
        )
        candidate_episodes = (
            list(cfg.dataset.episodes)
            if cfg.dataset.episodes is not None
            else list(range(ds_meta.total_episodes))
        )
        _validate_episode_range(candidate_episodes, ds_meta.total_episodes, "dataset.episodes")
        if cfg.validation.episodes is not None:
            _validate_episode_range(cfg.validation.episodes, ds_meta.total_episodes, "validation.episodes")

        train_episodes, val_episodes = resolve_sarm_validation_episodes(
            candidate_episodes,
            ratio=cfg.validation.ratio,
            split=cfg.validation.split,
            validation_episodes=cfg.validation.episodes,
        )
        active = len(val_episodes) > 0
        payload[0] = {
            "active": active,
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "candidate_count": len(candidate_episodes),
        }

    split = broadcast_object_list(payload, from_process=0)[0]
    train_episodes = split["train_episodes"]
    val_episodes = split["val_episodes"]
    cfg.validation.resolved_train_episodes = train_episodes
    cfg.validation.resolved_val_episodes = val_episodes

    if not split["active"]:
        if is_main_process and split["candidate_count"] < 2:
            logging.warning(
                "Skipping SARM validation because at least 2 candidate episodes are required; got %d.",
                split["candidate_count"],
            )
        return False, []

    cfg.dataset.episodes = train_episodes
    if is_main_process:
        logging.info(
            "SARM validation split resolved: train episodes=%s, validation episodes=%s",
            train_episodes,
            val_episodes,
        )
    return True, val_episodes


def _make_dataset_for_episodes(cfg: TrainPipelineConfig, episodes: list[int]):
    val_cfg = deepcopy(cfg)
    val_cfg.dataset.episodes = episodes
    return make_dataset(val_cfg)


def _make_offline_sampler(dataset, cfg: TrainPipelineConfig, shuffle: bool):
    active_cfg = cfg.trainable_config
    if cfg.dataset.intervention_only and not cfg.dataset.streaming:
        dataset_from_indices, dataset_to_indices = selected_episode_boundaries(dataset)
        return (
            EpisodeAwareSampler(
                dataset_from_indices,
                dataset_to_indices,
                drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
                shuffle=shuffle,
                eligible_indices=dataset.intervention_indices,
            ),
            False,
        )

    if hasattr(active_cfg, "drop_n_last_frames") and not cfg.dataset.streaming:
        if dataset.episodes is None:
            dataset_from_indices = dataset.meta.episodes["dataset_from_index"]
            dataset_to_indices = dataset.meta.episodes["dataset_to_index"]
        else:
            dataset_from_indices, dataset_to_indices = selected_episode_boundaries(dataset)
        return (
            EpisodeAwareSampler(
                dataset_from_indices,
                dataset_to_indices,
                drop_n_last_frames=active_cfg.drop_n_last_frames,
                shuffle=shuffle,
            ),
            False,
        )

    return None, shuffle


def _set_processor_training_mode(processor, mode: bool) -> None:
    for step in getattr(processor, "steps", ()):
        if hasattr(step, "train"):
            step.train(mode)


def _as_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().item()
    if isinstance(value, int | float):
        return float(value)
    return None


def run_sarm_validation(
    *,
    policy,
    preprocessor,
    val_dataloader,
    val_dataset,
    accelerator: "Accelerator",
    max_batches: int | None,
) -> dict[str, float]:
    """Run offline SARM validation on the main process and return averaged metrics."""
    unwrapped_policy = accelerator.unwrap_model(policy)
    was_training = policy.training
    policy.eval()
    _set_processor_training_mode(preprocessor, False)

    totals: dict[str, float] = {"loss": 0.0}
    num_batches = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                for cam_key in val_dataset.meta.camera_keys:
                    if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                        batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

                batch = preprocessor(batch)
                with accelerator.autocast():
                    loss, output_dict = unwrapped_policy.forward(batch)

                totals["loss"] += loss.detach().float().item()
                if output_dict:
                    for key in SARM_VALIDATION_METRIC_KEYS:
                        value = _as_float(output_dict.get(key))
                        if value is not None:
                            totals[key] = totals.get(key, 0.0) + value
                num_batches += 1
    finally:
        if was_training:
            policy.train()
        _set_processor_training_mode(preprocessor, True)

    if num_batches == 0:
        logging.warning("SARM validation produced no batches; skipping validation metrics.")
        return {}

    return {key: value / num_batches for key, value in totals.items()}


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:
            # Use per-sample loss for weighted training
            # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Weighted loss: each sample's contribution is scaled by its weight.
            # We divide by weight sum (not batch size) so that if some weights are zero,
            # the remaining samples contribute proportionally more, preserving gradient scale.
            # Weights are pre-normalized to sum to batch_size for stable training dynamics.
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

            # Log weighting statistics
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator | None" = None,
    update_policy_fn: Callable[..., tuple[MetricsTracker, dict | None]] = update_policy,
    dataset_validator: Callable[[Any, TrainPipelineConfig], None] | None = None,
    policy_setup_fn: Callable[[PreTrainedPolicy, TrainPipelineConfig], PreTrainedPolicy] | None = None,
):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator

    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    validation_active, validation_episodes = _resolve_sarm_validation_split(cfg, accelerator, is_main_process)

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        intervention_data = compute_intervention_only_data(dataset) if cfg.dataset.intervention_only else None
        if dataset_validator is not None:
            dataset_validator(dataset, cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)
        intervention_data = None
        if dataset_validator is not None:
            dataset_validator(dataset, cfg)

    if cfg.dataset.intervention_only:
        # The projected parquet scan and true-only statistics are computed once. Broadcast both the
        # runtime stats and eligible row indices so every rank uses identical normalization/sampling.
        from accelerate.utils import broadcast_object_list

        payload = broadcast_object_list([intervention_data], from_process=0)
        intervention_data = payload[0]
        apply_intervention_only_data(dataset, intervention_data)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    if policy_setup_fn is not None:
        policy = policy_setup_fn(policy, cfg)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    if (
        getattr(active_cfg, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms if "
            "the checkpoint processors do not define them. Building processors from current policy config."
        )
        processor_pretrained_path = None
    if (
        getattr(active_cfg, "discrete_state_in_language", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "discrete_state_in_language=true with pretrained SmolVLA processors requires the current "
            "state-prompt preprocessor. Building processors from current policy config."
        )
        processor_pretrained_path = None

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            **processor_kwargs,
            **postprocessor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        if cfg.dataset.intervention_only:
            intervention_ratio = dataset.intervention_true_frames / dataset.intervention_total_frames
            logging.info(
                "Intervention-only dataset: %d/%d frames (%.2f%%) have intervention=true",
                dataset.intervention_true_frames,
                dataset.intervention_total_frames,
                intervention_ratio * 100,
            )
            for feature_key in ("observation.state", "action"):
                feature_stats = dataset.meta.stats[feature_key]
                logging.info(
                    "True-only %s stats: mean=%s std=%s",
                    feature_key,
                    feature_stats["mean"],
                    feature_stats["std"],
                )
            logging.info("Streaming intervention filter enabled: %s", cfg.dataset.streaming)
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    sampler, shuffle = _make_offline_sampler(dataset, cfg, shuffle=True)

    if is_main_process and cfg.dataset.intervention_only:
        effective_samples = len(sampler) if sampler is not None else dataset.intervention_true_frames
        logging.info("Effective intervention-only training samples: %d", effective_samples)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    val_dataset = None
    val_dataloader = None
    if validation_active and is_main_process:
        logging.info("Creating SARM validation dataset")
        val_dataset = _make_dataset_for_episodes(cfg, validation_episodes)
        if cfg.dataset.intervention_only:
            val_intervention_data = compute_intervention_only_data(val_dataset)
            apply_intervention_only_data(val_dataset, val_intervention_data)
        val_sampler, val_shuffle = _make_offline_sampler(val_dataset, cfg, shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=val_shuffle and not cfg.dataset.streaming,
            sampler=val_sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )
        logging.info(
            "SARM validation dataset: num_frames=%s, num_episodes=%s, max_batches=%s",
            val_dataset.num_frames,
            val_dataset.num_episodes,
            cfg.validation.max_batches,
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    has_warned_missing_subtask = False
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)

        if getattr(cfg, "use_subtasks", False):
            if "subtask" in batch:
                batch["task"] = batch["subtask"]
            elif not has_warned_missing_subtask:
                logging.warning(
                    "cfg.use_subtasks is True, but 'subtask' key is missing from the dataset batch. "
                    "Make sure your dataset provides subtask annotations. Falling back to default 'task' key."
                )
                has_warned_missing_subtask = True

        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy_fn(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_validation_step = validation_active and (step % cfg.validation.freq == 0 or step == cfg.steps)

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log sample weighting statistics if enabled
                if sample_weighter is not None:
                    weighter_stats = sample_weighter.get_stats()
                    wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_validation_step:
            if is_main_process:
                val_metrics = run_sarm_validation(
                    policy=policy,
                    preprocessor=preprocessor,
                    val_dataloader=val_dataloader,
                    val_dataset=val_dataset,
                    accelerator=accelerator,
                    max_batches=cfg.validation.max_batches,
                )
                if val_metrics:
                    logging.info(
                        "Validation step %s: %s",
                        step,
                        " ".join(f"val/{key}:{value:.4f}" for key, value in val_metrics.items()),
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(val_metrics, step, mode="val")
            accelerator.wait_for_everyone()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
