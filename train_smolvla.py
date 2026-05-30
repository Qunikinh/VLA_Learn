#!/usr/bin/env python
import argparse
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import draccus
import torch
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Compatibility shim for the broken top-level `lerobot.policies` package import.
# We bypass `lerobot.policies.__init__` and import the SmolVLA modules directly.
# ---------------------------------------------------------------------------

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

# Import SmolVLA classes after the shim is registered.
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def apply_runtime_defaults(config_path: Path) -> None:
    config_path = config_path.resolve()
    base_dir = config_path.parent.resolve()

    default_data_root = (base_dir / "demo_data").resolve()
    default_output_dir = (base_dir / "ckpt" / "smolvla").resolve()

    data_root = os.getenv("DATA_ROOT")
    if not data_root:
        os.environ["DATA_ROOT"] = str(default_data_root)
        print(f"[env] DATA_ROOT not set, using default: {default_data_root}")

    checkpoint_dir = os.getenv("CHECKPOINT_DIR")
    if not checkpoint_dir:
        os.environ["CHECKPOINT_DIR"] = str(default_output_dir)
        print(f"[env] CHECKPOINT_DIR not set, using default: {default_output_dir}")

    # Keep backward compatibility: also set OUTPUT_DIR for cfg YAML expansion
    if not os.getenv("OUTPUT_DIR"):
        os.environ["OUTPUT_DIR"] = os.environ["CHECKPOINT_DIR"]


def expand_env_in_yaml(config_path: Path) -> Path:
    expanded = os.path.expandvars(config_path.read_text())
    tmp_dir = Path(tempfile.mkdtemp(prefix="smolvla_train_"))
    expanded_path = tmp_dir / config_path.name
    expanded_path.write_text(expanded)
    return expanded_path


def build_config(config_path: Path) -> TrainPipelineConfig:
    expanded_path = expand_env_in_yaml(config_path)
    expanded_data = yaml.safe_load(expanded_path.read_text())
    # draccus.decode doesn't accept unknown fields; strip customize before decoding
    customize_data = expanded_data.pop("customize", None) if isinstance(expanded_data, dict) else None
    cfg = draccus.decode(TrainPipelineConfig, expanded_data)
    if customize_data is not None:
        cfg.customize = customize_data  # type: ignore[attr-defined]

    resolved_data_root = Path(os.path.expandvars(cfg.dataset.root)).resolve()
    resolved_output_dir = Path(os.path.expandvars(cfg.output_dir)).resolve()
    cfg.dataset.root = str(resolved_data_root)
    cfg.output_dir = str(resolved_output_dir)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="SmolVLA local offline training")
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("train_config.yaml"),
        help="Path to the YAML training config. Environment variables like ${DATA_ROOT} are expanded.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("CUDA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        help="Training device, e.g. cuda or cpu.",
    )
    args = parser.parse_args()

    apply_runtime_defaults(args.config_path)
    cfg = build_config(args.config_path)
    cfg.validate()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={device}")
    print(f"[train] output_dir={output_dir}")
    print(f"[train] dataset_root={cfg.dataset.root}")

    # Dataset loading uses LeRobot's train config directly.
    dataset = make_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dataloader = cycle(dataloader)

    # Infer feature shapes from the local dataset metadata so SmolVLA knows
    # which observation keys are visual/state/action tensors before instantiation.
    features = dataset_to_policy_features(dataset.meta.features)
    cfg.policy.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.policy.input_features:
        cfg.policy.input_features = {key: ft for key, ft in features.items() if key not in cfg.policy.output_features}
    cfg.policy.device = str(device)

    policy = SmolVLAPolicy(cfg.policy)
    preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset.meta.stats)
    policy.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.policy.optimizer_lr,
        betas=cfg.policy.optimizer_betas,
        eps=cfg.policy.optimizer_eps,
        weight_decay=cfg.policy.optimizer_weight_decay,
    )

    step = 0
    running_loss = 0.0
    last_logged = 0
    for step in range(1, cfg.steps + 1):
        batch = next(dataloader)
        batch = preprocessor(batch)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=cfg.policy.use_amp):
            loss, output_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.policy.optimizer_grad_clip_norm)
        optimizer.step()

        running_loss += float(loss.item())
        if step % cfg.log_freq == 0:
            avg_loss = running_loss / max(1, step - last_logged)
            print(f"[step {step}] loss={avg_loss:.6f} grad_norm={grad_norm:.4f}")
            running_loss = 0.0
            last_logged = step

        if cfg.save_checkpoint and step % cfg.save_freq == 0:
            checkpoint_path = checkpoints_dir / f"step_{step}"
            policy.save_pretrained(checkpoint_path)
            print(f"[checkpoint] saved to {checkpoint_path}")

    # Keep the latest checkpoint as `last` for easy deployment.
    policy.save_pretrained(output_dir / "last")
    print(f"[train] finished. Latest checkpoint saved to {output_dir / 'last'}")

    # Optionally push model to a hub according to training config customize section
    try:
        from hubservice import push_model

        try:
            push_model(cfg)
        except Exception as exc:
            print(f"[push] push_model failed: {exc}")
    except Exception as exc:
        print(f"[push] hubservice module unavailable: {exc}")


if __name__ == "__main__":
    main()
