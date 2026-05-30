#!/usr/bin/env python
import argparse
import importlib
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from mujoco_env.mug_env import SimpleEnv2

# ---------------------------------------------------------------------------
# Compatibility shim for the broken top-level `lerobot.policies` package import.
# We bypass `lerobot.policies.__init__` and import the SmolVLA modules directly.
# ---------------------------------------------------------------------------
#todo使用环境变量数据集和检查点目录,如果没有则从仓库拉取
try:
    lerobot_spec = importlib.util.find_spec("lerobot")
    if lerobot_spec is None or lerobot_spec.origin is None:
        raise RuntimeError("Cannot locate installed lerobot package")
    lerobot_package_dir = Path(lerobot_spec.origin).resolve().parent
    policies_package = types.ModuleType("lerobot.policies")
    policies_package.__path__ = [str((lerobot_package_dir / "policies").resolve())]
    sys.modules.setdefault("lerobot.policies", policies_package)
except Exception as exc:
    raise RuntimeError(f"Failed to initialize LeRobot compatibility shim: {exc}") from exc

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser(description="Deploy a trained SmolVLA checkpoint in the MuJoCo environment")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(os.getenv("CHECKPOINT_DIR", os.getenv("OUTPUT_DIR", "./ckpt/smolvla"))) / "last",
        help="Path to a saved SmolVLA checkpoint directory.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("./asset/mug_scene/scene.xml"),
        help="MuJoCo scene XML file to load.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum rollout steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("CUDA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run inference on (cuda or cpu).",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available")

    scene_path = Path(args.scene).resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    print(f"[deploy] checkpoint={checkpoint_path}")
    print(f"[deploy] scene={scene_path}")
    print(f"[deploy] device={device}")

    # If the local checkpoint is missing, try to pull it from the remote hub.
    if not checkpoint_path.exists() or not any(checkpoint_path.iterdir()):
        print("[deploy] checkpoint not found locally; trying pull from hub ...")
        try:
            from hubservice import pull_model
            import yaml

            train_cfg_path = Path(os.getenv("TRAIN_CONFIG", "train_config.yaml"))
            if train_cfg_path.exists():
                data = yaml.safe_load(train_cfg_path.read_text())

                class Cfg:
                    pass

                cfg = Cfg()
                for k, v in (data or {}).items():
                    setattr(cfg, k, v)
                # Ensure OUTPUT_DIR env is set for backward-compat YAML expansion
                if not os.getenv("OUTPUT_DIR"):
                    os.environ["OUTPUT_DIR"] = os.environ.get("CHECKPOINT_DIR", "./ckpt/smolvla")
                pulled = pull_model(cfg)
                if pulled:
                    print(f"[deploy] pulled model from hub -> {checkpoint_path}")
            else:
                print("[deploy] no train_config.yaml found; cannot pull from hub")
        except Exception as exc:
            print(f"[deploy] pull from hub failed: {exc}")

    if not checkpoint_path.exists() or not any(checkpoint_path.iterdir()):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_root = Path(os.getenv("DATA_ROOT", "./demo_data")).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_root.name, root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    dataset_stats = None
    if dataset_metadata.stats is not None:
        dataset_stats = {
            feature_name: {
                stat_name: torch.as_tensor(stat_value)
                for stat_name, stat_value in feature_stats.items()
            }
            for feature_name, feature_stats in dataset_metadata.stats.items()
        }

    policy_config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=5,
        n_action_steps=5,
        device=str(device),
    )

    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path), config=policy_config)
    policy.to(device)
    policy.eval()

    smolvla_processor_module = importlib.import_module("lerobot.policies.smolvla.processor_smolvla")
    make_smolvla_pre_post_processors = smolvla_processor_module.make_smolvla_pre_post_processors
    preprocessor, _ = make_smolvla_pre_post_processors(
        policy.config,
        dataset_stats,
    )

    env = SimpleEnv2(str(scene_path), action_type="delta_joint_angle")
    env.reset(seed=0)

    step = 0
    while step < args.max_steps and env.env.is_viewer_alive():
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue

        state = np.asarray(env.get_joint_state()[:6], dtype=np.float32)
        image, wrist_image = env.grab_image()

        image_pil = Image.fromarray(image).resize((256, 256))
        wrist_pil = Image.fromarray(wrist_image).resize((256, 256))
        image_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)
        wrist_tensor = transforms.ToTensor()(wrist_pil).unsqueeze(0).to(device)

        batch = {
            "observation.state": torch.from_numpy(state)[None, :].to(device),
            "observation.image": image_tensor,
            "observation.wrist_image": wrist_tensor,
            "task": env.instruction,
        }
        batch = preprocessor(batch)

        with torch.inference_mode():
            actions = policy.select_action(batch)

        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim == 2:
            actions = actions[0]

        action = actions[:7].detach().cpu().numpy()
        action[-1] = 1.0 if action[-1] >= 0.5 else 0.0  # gripper: round to binary
        env.step(action)
        env.render()
        step += 1

    print(f"[deploy] finished rollout after {step} steps")


if __name__ == "__main__":
    main()
