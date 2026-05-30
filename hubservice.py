#!/usr/bin/env python
"""
Hub service: push trained models to / pull models from remote hubs.

Controls are read from the training config (cfg.customize section):

  customize:
    push_to_hub: true                # bool – enable upload after training
    provider: modelscope             # "modelscope" or "huggingface"
    access_token: ms-xxx             # API token
    repo: qunikin/smolvla_qunikin    # username/repo
    download_dir: ./ckpt/download    # local dir for pullfromhub

Environment variables CHECKPOINT_DIR control the local model path.
This module is tolerant: if a provider SDK is missing it
prints a message and returns instead of raising.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_model_dir(cfg) -> Path:
    """Local folder that holds the trained model to push."""
    out = os.getenv("CHECKPOINT_DIR") or os.getenv("OUTPUT_DIR") or getattr(cfg, "output_dir", None)
    if out is None:
        raise RuntimeError("No checkpoint directory (CHECKPOINT_DIR / cfg.output_dir)")
    return (Path(out) / "last").resolve()


def _get_download_dir(cfg) -> Path:
    """Local folder where a pulled model should land (same as checkpoint dir)."""
    out = os.getenv("CHECKPOINT_DIR") or os.getenv("OUTPUT_DIR") or getattr(cfg, "output_dir", None)
    if out is None:
        raise RuntimeError("No checkpoint directory (CHECKPOINT_DIR / cfg.output_dir)")
    return (Path(out) / "last").resolve()


def _customize_section(cfg) -> dict:
    customize = getattr(cfg, "customize", None)
    if isinstance(customize, dict):
        return customize
    return {}


# ---------------------------------------------------------------------------
# HuggingFace provider
# ---------------------------------------------------------------------------

def _hf_push(local_dir: Path, repo_id: str, token: Optional[str] = None) -> None:
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except Exception:
        print("[hubservice] huggingface_hub not installed; skip HF push")
        return

    token = token or os.getenv("HF_TOKEN")
    if not token:
        print("[hubservice] HF_TOKEN not set; skip HF push")
        return

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, private=False, exist_ok=True, token=token)
    except Exception:
        pass

    for p in sorted(local_dir.rglob("*")):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local_dir)).replace("\\", "/")
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=repo_id, token=token)
            print(f"[hubservice/hf] uploaded {rel}")
        except Exception as exc:
            print(f"[hubservice/hf] FAIL {rel}: {exc}")


def _hf_pull(local_dir: Path, repo_id: str, token: Optional[str] = None) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except Exception:
        print("[hubservice] huggingface_hub not installed; skip HF pull")
        return

    token = token or os.getenv("HF_TOKEN")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=token)
    print(f"[hubservice/hf] pulled {repo_id} -> {local_dir}")


# ---------------------------------------------------------------------------
# ModelScope provider
# ---------------------------------------------------------------------------

def _ms_push(local_dir: Path, repo_id: str, token: str) -> None:
    try:
        from modelscope.hub.api import HubApi  # type: ignore[import-untyped]
    except Exception:
        print("[hubservice] modelscope SDK not installed; skip MS push")
        return

    if not token:
        print("[hubservice] access_token not set; skip MS push")
        return

    api = HubApi()
    api.login(token)

    # Ensure repo exists
    try:
        api.create_model(repo_id, token=token, exist_ok=True)
    except Exception:
        pass

    # Upload files one by one using the upload_file API
    for p in sorted(local_dir.rglob("*")):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local_dir))
        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=rel,
                repo_id=repo_id,
                revision="master",
            )
            print(f"[hubservice/ms] uploaded {rel}")
        except Exception as exc:
            print(f"[hubservice/ms] FAIL {rel}: {exc}")

    print(f"[hubservice/ms] pushed {local_dir} -> {repo_id}")


def _ms_pull(local_dir: Path, repo_id: str, token: str) -> None:
    """Download a repo from ModelScope to *local_dir* (overwrites if exists)."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore[import-untyped]
    except Exception:
        print("[hubservice] modelscope SDK not installed; skip MS pull")
        return

    if not token:
        print("[hubservice] access_token not set; skip MS pull")
        return

    # Remove existing download dir for a clean copy
    if local_dir.exists():
        shutil.rmtree(str(local_dir))

    try:
        # snapshot_download downloads into a cache_dir. We use a temp directory
        # and then copy the actual repo files to the target local_dir.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="ms_pull_") as tmp:
            snapshot_download(repo_id, cache_dir=tmp, revision="master")
            tmp_path = Path(tmp)
            # Walk the temp dir and find the deepest directory that contains
            # the actual model content (not just modelscope metadata).
            all_dirs = [d for d in tmp_path.rglob("*") if d.is_dir()]
            repo_dir = None
            if all_dirs:
                # Prefer the deepest directory with non-hidden files
                for d in sorted(all_dirs, key=lambda x: len(x.parts), reverse=True):
                    real_files = [f for f in d.iterdir() if not f.name.startswith(".")]
                    if real_files:
                        repo_dir = d
                        break
            if repo_dir is None:
                print(f"[hubservice/ms] no files found in download for {repo_id}")
                return
            # Copy the *contents* of repo_dir into local_dir
            local_dir.mkdir(parents=True, exist_ok=True)
            for item in repo_dir.iterdir():
                dst = local_dir / item.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(str(dst))
                    else:
                        dst.unlink()
                if item.is_dir():
                    shutil.copytree(str(item), str(dst))
                else:
                    shutil.copy2(str(item), str(dst))
        print(f"[hubservice/ms] pulled {repo_id} -> {local_dir}")
    except Exception as exc:
        print(f"[hubservice/ms] pull failed: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def push_model(cfg) -> None:
    """Upload the latest trained model to a remote hub.

    Reads from cfg.customize: provider, access_token, repo.
    Model source: CHECKPOINT_DIR/last  (env var or cfg.output_dir).
    """
    customize = _customize_section(cfg)
    push_enabled = customize.get("push_to_hub", False)
    if not push_enabled:
        print("[hubservice] push_to_hub disabled; skipping")
        return

    provider = (customize.get("provider") or "modelscope").lower()
    token = customize.get("access_token") or ""
    repo = customize.get("repo") or ""
    if not repo:
        print("[hubservice] no repo configured; skip push")
        return

    model_dir = _get_model_dir(cfg)
    if not model_dir.exists():
        print(f"[hubservice] model dir not found: {model_dir}; skip push")
        return

    print(f"[hubservice] push {model_dir} -> {provider}:{repo}")
    if provider == "modelscope":
        _ms_push(model_dir, repo, token)
    elif provider == "huggingface":
        _hf_push(model_dir, repo, token)
    else:
        print(f"[hubservice] unknown provider '{provider}'")


def pull_model(cfg) -> bool:
    """Download a model from a remote hub to the local download_dir.

    Returns True if files were downloaded, False otherwise.
    Reads from cfg.customize: provider, access_token, repo, download_dir.
    """
    customize = _customize_section(cfg)
    provider = (customize.get("provider") or "modelscope").lower()
    token = customize.get("access_token") or ""
    repo = customize.get("repo") or ""
    if not repo:
        print("[hubservice] no repo configured; skip pull")
        return False

    download_dir = _get_download_dir(cfg)

    # If the target dir already has model files, skip download
    if download_dir.exists() and any(download_dir.iterdir()):
        print(f"[hubservice] model already exists at {download_dir}; skip pull")
        return True

    print(f"[hubservice] pull {provider}:{repo} -> {download_dir}")
    if provider == "modelscope":
        _ms_pull(download_dir, repo, token)
    elif provider == "huggingface":
        _hf_pull(download_dir, repo, token)
    else:
        print(f"[hubservice] unknown provider '{provider}'")
        return False

    return download_dir.exists() and any(download_dir.iterdir())


# ---------------------------------------------------------------------------
# CLI (for standalone testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    cfg_path = Path(os.getenv("TRAIN_CONFIG", "train_config.yaml"))
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}")
        raise SystemExit(1)

    data = yaml.safe_load(cfg_path.read_text())

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in (data or {}).items():
        setattr(cfg, k, v)

    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "push"
    if cmd == "pull":
        pull_model(cfg)
    else:
        push_model(cfg)
