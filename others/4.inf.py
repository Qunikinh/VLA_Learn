import os
import sys
import ctypes
import re
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_VISIBLE_DEVICES"] = "0"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libEGL.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

from unittest.mock import MagicMock
mock_glfw = MagicMock()
mock_glfw.init.return_value = True
mock_glfw.create_window.return_value = MagicMock()
sys.modules["glfw"] = mock_glfw
sys.modules["pyautogui"] = MagicMock()
sys.modules["mouseinfo"] = MagicMock()
sys.modules["tkinter"] = MagicMock()

import torch
import numpy as np
import cv2
import mujoco
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file

from mujoco_env.mug_env import SimpleEnv2
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# import register_qwen
# from qwen_policy import QwenVLPolicy, QwenVLPolicyConfig

SimpleEnv2.init_viewer = lambda self: print("✅ 检测到服务器环境：已跳过 GLFW 窗口。")
SimpleEnv2.render = lambda self, **kwargs: None


def find_checkpoint_recursive(base_dir: str = "./ckpt/qwen_finetune") -> tuple[str, str]:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Checkpoint 根目录不存在: {base}")
    
    best_path = None
    best_weights = None
    best_step = -1
    weight_extensions = [".safetensors", ".pt", ".bin"]
    
    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        for ext in weight_extensions:
            weight_files = [f for f in files if f.endswith(ext) and "model" in f.lower()]
            for wf in weight_files:
                if any(x in wf.lower() for x in ["optimizer", "rng", "scheduler"]):
                    continue
                step = 0
                for part in root_path.parts:
                    m = re.search(r'(\d+)', part)
                    if m:
                        step = max(step, int(m.group(1)))
                    if part == "last":
                        step = 9999999
                if step > best_step:
                    best_step = step
                    best_path = root_path
                    best_weights = str(root_path / wf)
    
    if best_path is None:
        raise FileNotFoundError(f"在 {base} 下递归搜索，找不到任何有效的模型权重文件。")
    
    print(f"✅ 找到 checkpoint: {best_path} (step≈{best_step})")
    print(f"   权重文件: {best_weights}")
    return str(best_path), best_weights


device = 'cuda'
xml_path = './asset/example_scene_y2.xml'
video_output_path = 'eval_videos/qwen_eval.mp4'
os.makedirs('eval_videos', exist_ok=True)

print("正在初始化环境 (EGL 模式)...")
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

print("正在配置 MuJoCo 官方离线渲染器 (Native EGL)...")
headless_renderer = mujoco.Renderer(PnPEnv.env.model, 256, 256)

def custom_grab_image():
    headless_renderer.update_scene(PnPEnv.env.data, camera="agentview")
    img_main = headless_renderer.render().copy()
    headless_renderer.update_scene(PnPEnv.env.data, camera="egocentric")
    img_wrist = headless_renderer.render().copy()
    return img_main, img_wrist

PnPEnv.grab_image = custom_grab_image
print("✅ 官方 EGL 渲染器挂载成功！")

# ✅ 修改：和训练时一样用 seed=0 重置，确保物体初始分布一致
PnPEnv.reset(seed=44)
print(f"当前指令: {PnPEnv.instruction}")

dataset_metadata = LeRobotDatasetMetadata("datawhale_eai_pnp_language", root='./demo_data_language')

checkpoint_dir, weights_path = find_checkpoint_recursive()
print(f"加载配置从: {checkpoint_dir}")
print(f"加载权重从: {weights_path}")

config = QwenVLPolicyConfig.from_pretrained(checkpoint_dir)
policy = QwenVLPolicy(config, dataset_metadata)

if weights_path.endswith(".safetensors"):
    state_dict = load_file(weights_path, device=str(device))
else:
    state_dict = torch.load(weights_path, map_location=device)

policy.load_state_dict(state_dict, strict=False)
policy.to(device)
policy.eval()
print("✅ 模型加载完成")

fps = 240
video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (512, 256))
transform = transforms.Compose([transforms.ToTensor()])

# ✅ 修改：和训练时一样用 seed=0 重置，开始仿真
PnPEnv.reset(seed=1)
policy.reset()
policy.eval()

max_steps = 100000
print(f"🚀 开始录制长视频... 预设总步数: {max_steps}")

for i in range(max_steps):
    PnPEnv.step_env() 
    
    if PnPEnv.env.loop_every(HZ=20):
        state = PnPEnv.get_joint_state()[:6]
        image_raw, wrist_raw = PnPEnv.grab_image() 
        
        image_pil = Image.fromarray(image_raw).resize((256, 256))
        wrist_pil = Image.fromarray(wrist_raw).resize((256, 256))
        
        data = {
            'observation.state': torch.from_numpy(np.array([state], dtype=np.float32)).to(device),
            'observation.image': transform(image_pil).unsqueeze(0).to(device),
            'observation.wrist_image': transform(wrist_pil).unsqueeze(0).to(device),
            'task': [PnPEnv.instruction],
        }

        with torch.no_grad():
            action = policy.select_action(data)
        
        action_np = action[:7]
        _ = PnPEnv.step(action_np)

    if i % 25 == 0:
        img_for_video, wrist_for_video = PnPEnv.grab_image()
        combined_frame = np.hstack([img_for_video, wrist_for_video])
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(combined_frame)

    if (i+1) % 100 == 0:
        print(f"进度: {i+1}/{max_steps} 步...")

video_writer.release()
print(f"🎉 完整视频已保存：{video_output_path}")

try:
    headless_renderer.close()
except:
    pass