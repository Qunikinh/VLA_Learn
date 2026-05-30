from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.datasets.utils import write_json, serialize_dict
from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index


ROOT = "./demo_data" # The root directory to save the demonstrations 
# If you have downloaded the dataset from Hugging Face, you can set the root to the directory where the dataset is stored
# ROOT = './datawhale_eai_pnp_language' # if you want to use the example data provided, root = './datawhale_eai_pnp_language' instead!
dataset = LeRobotDataset('qunikin_language', root=ROOT) # if youu want to use the example data provided, root = './datawhale_eai_pnp_language' instead!

# Compute episode -> data index mapping for samplers
episode_data_index = calculate_episode_data_index(dataset.hf_dataset)

# If you want to use the collected dataset, please download it from Hugging Face.
# dataset = LeRobotDataset('datawhale_eai_pnp_language', root='datawhale_eai_pnp_language')

import torch

class EpisodeSampler(torch.utils.data.Sampler):  # *取特定episode
    """Sampler for a single episode using an episode_data_index mapping."""
    def __init__(self, episode_data_index: dict, episode_index: int):
        from_idx = int(episode_data_index["from"][episode_index].item())
        to_idx = int(episode_data_index["to"][episode_index].item())
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)
    
    # Select an episode index that you want to visualize
episode_index = 0

episode_sampler = EpisodeSampler(episode_data_index, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)

from mujoco_env.mug_env import SimpleEnv2  # !修改自定义场景
xml_path = './asset/mug_scene/scene.xml'#!场景文件
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

import torch
from tqdm import tqdm # 可选，用于显示进度

# 1. 获取数据集中所有的 episode 索引
total_episodes = dataset.num_episodes
print(f"Total episodes to visualize: {total_episodes}")

# 2. 外层循环：遍历每一个 episode
for ep_idx in range(total_episodes):
    print(f"--- Visualizing Episode {ep_idx} ---")
    
    # 为当前 episode 创建专门的 Sampler 和 DataLoader
    current_sampler = EpisodeSampler(episode_data_index, ep_idx)
    current_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=current_sampler,
    )
    
    iter_dataloader = iter(current_dataloader)
    PnPEnv.reset()
    step = 0
    
    # 内层循环：播放当前 episode 的每一帧
    while PnPEnv.env.is_viewer_alive():
        PnPEnv.step_env()
        
        # 控制回放频率 (例如 20Hz)
        if PnPEnv.env.loop_every(HZ=20):
            try:
                data = next(iter_dataloader)
            except StopIteration:
                # 当前 episode 播放完毕，跳出内层循环，进入下一个 episode
                break
            
            if step == 0:
                # 仅在每一集开始时设置指令和物体初始位置
                instruction = data['task'][0]
                PnPEnv.set_instruction(instruction)
                # 注意：确保数据集中有 'obj_init' 字段，如果没有请注释掉下面这行或做异常处理
                if 'obj_init' in data:
                    # obj_init can be stored as (1, N) or (N,), ensure we flatten and slice robustly
                    obj_vec = data['obj_init'][0].numpy().ravel() if hasattr(data['obj_init'][0], 'numpy') else np.asarray(data['obj_init'][0]).ravel()
                    PnPEnv.set_obj_pose(obj_vec[0:3], obj_vec[3:6], obj_vec[6:9])

            
            # 获取动作并执行
            action = data['action'].numpy()
            obs = PnPEnv.step(action[0])

            # 可视化图像叠加
            # 注意：确保图像数据范围是 0-1，如果是 0-255 则不需要 *255
            img_agent = data['observation.image'][0].numpy()
            img_ego = data['observation.wrist_image'][0].numpy()
            
            # 处理图像格式 (C,H,W) -> (H,W,C) 并转换为 uint8
            if img_agent.max() <= 1.0:
                img_agent = (img_agent * 255).astype(np.uint8)
                img_ego = (img_ego * 255).astype(np.uint8)
            else:
                img_agent = img_agent.astype(np.uint8)
                img_ego = img_ego.astype(np.uint8)
                
            PnPEnv.rgb_agent = np.transpose(img_agent, (1,2,0))
            PnPEnv.rgb_ego = np.transpose(img_ego, (1,2,0))
            PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
            
            PnPEnv.render(teleop=True, idx=ep_idx) 
            step += 1

    # 当前 episode 结束后，可以选择是否暂停或等待用户按键，或者直接进入下一个
    # input("Press Enter to continue to next episode...") 

print("All episodes visualized.")
PnPEnv.env.close_viewer()
# dataset.push_to_hub(upload_large_folder=True)
