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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32
    # Whether to condition SmolVLA on observation.state.
    use_state: bool = True
    # Encode normalized observation.state in the language prompt instead of using a continuous state token.
    discrete_state_in_language: bool = False
    # Optional training-time input dropout. Currently only supports observation.state.
    input_dropout_prob: float = 0.0
    input_dropout_features: list[str] = field(default_factory=list)
    # Optional training-time hard-prefix RTC regularization. Inference-time RTC remains configured
    # separately through rtc_config.
    training_time_rtc_enabled: bool = False
    training_time_rtc_max_delay_steps: int = 12

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    # Number of image tokens produced by the SmolVLM connector per camera.
    # Leave unset to use the VLM backbone default. For SmolVLM2-500M,
    # 256 corresponds to scale_factor=2 instead of the default scale_factor=4.
    image_seq_len: int | None = None

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.image_seq_len is not None and self.image_seq_len <= 0:
            raise ValueError(f"`image_seq_len` must be positive when set. Got {self.image_seq_len}.")
        if self.discrete_state_in_language:
            if not self.use_state:
                raise ValueError(
                    "`discrete_state_in_language=True` requires `use_state=True` because "
                    "`observation.state` is needed to build the language prompt."
                )
            self._set_state_normalization(NormalizationMode.QUANTILES)
            if self.tokenizer_max_length == 48:
                self.tokenizer_max_length = 200
        if not 0.0 <= self.input_dropout_prob <= 1.0:
            raise ValueError(
                f"`input_dropout_prob` must be in the range [0.0, 1.0]. Got {self.input_dropout_prob}."
            )
        supported_input_dropout_features = {OBS_STATE}
        unsupported_input_dropout_features = sorted(
            set(self.input_dropout_features) - supported_input_dropout_features
        )
        if unsupported_input_dropout_features:
            raise ValueError(
                "`input_dropout_features` contains unsupported feature(s): "
                f"{unsupported_input_dropout_features}. Supported features: "
                f"{sorted(supported_input_dropout_features)}."
            )
        if OBS_STATE in self.input_dropout_features and not self.use_state:
            raise ValueError(
                f"`input_dropout_features` includes `{OBS_STATE}`, but `use_state=False`. "
                "Set `policy.use_state=true` or remove state input dropout."
            )
        if OBS_STATE in self.input_dropout_features and self.discrete_state_in_language:
            raise ValueError(
                f"`input_dropout_features` cannot include `{OBS_STATE}` when "
                "`discrete_state_in_language=True` because state is encoded in the language prompt."
            )
        if self.training_time_rtc_max_delay_steps < 0:
            raise ValueError(
                "`training_time_rtc_max_delay_steps` must be greater than or equal to 0. "
                f"Got {self.training_time_rtc_max_delay_steps}."
            )
        if self.training_time_rtc_enabled and self.training_time_rtc_max_delay_steps >= self.chunk_size:
            raise ValueError(
                "`training_time_rtc_max_delay_steps` must be smaller than `chunk_size` when "
                "`training_time_rtc_enabled=True`. Got "
                f"{self.training_time_rtc_max_delay_steps} for `training_time_rtc_max_delay_steps` "
                f"and {self.chunk_size} for `chunk_size`."
            )

    def _set_state_normalization(self, normalization_mode: NormalizationMode) -> None:
        for key in list(self.normalization_mapping):
            key_value = key.value if isinstance(key, FeatureType) else key
            if key_value == FeatureType.STATE.value:
                self.normalization_mapping[key] = normalization_mode
                return
        self.normalization_mapping[FeatureType.STATE.value] = normalization_mode

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if not self.use_state:
            self.input_features = {
                key: feature
                for key, feature in self.input_features.items()
                if key != OBS_STATE and feature.type is not FeatureType.STATE
            }

        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
