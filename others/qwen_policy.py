"""
Qwen-based Policy for LeRobot
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Tuple, Any
from PIL import Image
import numpy as np

from lerobot.policies.pretrained import PreTrainedPolicy
from register_qwen import QwenVLPolicyConfig

# 兼容性导入 Normalize/Unnormalize
try:
    from lerobot.policies.normalize import Normalize, Unnormalize
except ImportError:
    class Normalize(nn.Module):
        def __init__(self, stats=None):
            super().__init__()
            self.stats = stats
        def forward(self, batch):
            if self.stats is None:
                return batch
            for key in ['observation.image', 'observation.state']:
                if key in batch and key in self.stats:
                    mean = torch.tensor(self.stats[key]['mean'], device=batch[key].device)
                    std = torch.tensor(self.stats[key]['std'], device=batch[key].device)
                    batch[key] = (batch[key] - mean) / std
            return batch

    class Unnormalize(nn.Module):
        def __init__(self, stats=None):
            super().__init__()
            self.stats = stats
        def forward(self, batch):
            if self.stats is None or 'action' not in batch:
                return batch
            if 'action' in self.stats:
                mean = torch.tensor(self.stats['action']['mean'], device=batch['action'].device)
                std = torch.tensor(self.stats['action']['std'], device=batch['action'].device)
                batch['action'] = batch['action'] * std + mean
            return batch


def _process_image_tensor(img_tensor: torch.Tensor) -> Image.Image:
    """
    将 LeRobot batch 中的图像 tensor 转换为 PIL.Image。
    支持 (C,H,W) 和 (n_obs,C,H,W) 输入。
    """
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[-1]
    
    img_np = img_tensor.detach().cpu().numpy()
    
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
        img_np = img_np.transpose(1, 2, 0)
    
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    
    return Image.fromarray(img_np)


class QwenVLPolicy(PreTrainedPolicy):
    config_class = QwenVLPolicyConfig
    name = "qwen"

    def __init__(self, config, ds_meta=None):
        super().__init__(config, ds_meta.stats if ds_meta else None)
        
        self.config = config
        self.ds_meta = ds_meta
        self.device = torch.device(config.device if hasattr(config, 'device') else 'cuda')
        self.action_dim = ds_meta.features['action']['shape'][0] if ds_meta else 7
        self.chunk_size = getattr(config, 'chunk_size', 5)
        self.n_action_steps = getattr(config, 'n_action_steps', 5)

        print(f"Initializing QwenVLPolicy:")
        print(f"  - Action dimension: {self.action_dim}")
        print(f"  - Chunk size: {self.chunk_size}")
        print(f"  - Device: {self.device}")

        model_path = getattr(config, 'pretrained_path', '/root/autodl-tmp/models/Qwen2-VL-2B-Instruct')
        print(f"Loading Qwen2-VL from {model_path}...")

        use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)

        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto' if self.device.type == 'cuda' else None,
            low_cpu_mem_usage=True,
        )

        if use_gradient_checkpointing and self.device.type == 'cuda':
            self.qwen_model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

        freeze_qwen = getattr(config, 'freeze_qwen', False)
        if freeze_qwen:
            for param in self.qwen_model.parameters():
                param.requires_grad = False
            print("✓ Qwen parameters frozen")
        else:
            print("✓ Qwen parameters trainable")

        hidden_size = self.qwen_model.config.hidden_size

        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim * self.chunk_size)
        ).to(self.device)

        self.normalize_inputs = Normalize(ds_meta.stats).to(self.device) if ds_meta else nn.Identity().to(self.device)
        self.unnormalize_outputs = Unnormalize(ds_meta.stats).to(self.device) if ds_meta else nn.Identity().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)

        print("✓ QwenVLPolicy initialized successfully!")

    def get_optim_params(self):
        """Return trainable parameters for the optimizer."""
        return self.parameters()

    def _build_messages(self, batch: Dict[str, torch.Tensor], batch_size: int) -> list:
        """从 batch 构建 Qwen 的 messages 列表。"""
        wrapped_messages = []
        for i in range(batch_size):
            pil_img = _process_image_tensor(batch['observation.image'][i])

            task_list = batch.get('task', [''] * batch_size)
            task_str = task_list[i] if i < len(task_list) else "pick and place"
            if isinstance(task_str, (list, tuple)) and len(task_str) > 0:
                task_str = str(task_str[0])
            elif not isinstance(task_str, str):
                task_str = str(task_str)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": f"Task: {task_str}. Predict the next {self.chunk_size} actions."}
                ]
            }]
            wrapped_messages.append(messages)
        return wrapped_messages

    def _qwen_forward(self, wrapped_messages: list, device: torch.device):
        """调用 Qwen 模型获取 pooled hidden state。"""
        text_prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in wrapped_messages
        ]
        images = [msg[0]["content"][0]["image"] for msg in wrapped_messages]

        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(device)

        outputs = self.qwen_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = outputs.hidden_states[-1]
        pooled_hidden = last_hidden_state.mean(dim=1)
        return pooled_hidden.to(torch.float32)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        batch_size = batch['action'].shape[0]
        device = batch['action'].device
        actions = batch['action']

        wrapped_messages = self._build_messages(batch, batch_size)
        pooled_hidden = self._qwen_forward(wrapped_messages, device)

        predicted_actions_flat = self.action_head(pooled_hidden)
        predicted_actions = predicted_actions_flat.view(batch_size, self.chunk_size, self.action_dim)

        target_actions = actions[:, -self.chunk_size:, :]
        loss = nn.functional.mse_loss(predicted_actions, target_actions)

        predicted_actions_unnorm = self.unnormalize_outputs({'action': predicted_actions})['action']
        output_dict = {
            'predicted_actions': predicted_actions_unnorm,
            'loss': loss.item(),
        }

        return loss, output_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions given environment observations (inference only)."""
        self.eval()
        batch_size = batch['observation.image'].shape[0]
        device = batch['observation.image'].device

        wrapped_messages = self._build_messages(batch, batch_size)
        pooled_hidden = self._qwen_forward(wrapped_messages, device)

        predicted_actions_flat = self.action_head(pooled_hidden)
        predicted_actions = predicted_actions_flat.view(batch_size, self.chunk_size, self.action_dim)

        predicted_actions = self.unnormalize_outputs({'action': predicted_actions})['action']

        return predicted_actions

    def select_action(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """Select a single action given environment observations."""
        with torch.no_grad():
            predicted_actions = self.predict_action_chunk(batch)
        first_action = predicted_actions[0, 0, :].cpu().numpy()
        return first_action

    def reset(self):
        pass


def make_qwen_policy(cfg, ds_meta):
    return QwenVLPolicy(cfg, ds_meta)