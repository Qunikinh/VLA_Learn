"""
Minimal config registration for Qwen policy.
必须在 train_model.py 中于 draccus.parse() 之前导入。
"""
from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("qwen")
@dataclass
class QwenVLPolicyConfig(PreTrainedConfig):
    type: str = "qwen"
    pretrained_path: str = '/root/autodl-tmp/models/Qwen2-VL-2B-Instruct'
    chunk_size: int = 5
    n_action_steps: int = 5
    n_obs_steps: int = 1
    freeze_qwen: bool = False
    use_gradient_checkpointing: bool = True
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("QwenVLPolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("QwenVLPolicy requires 'action' in output_features.")

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None