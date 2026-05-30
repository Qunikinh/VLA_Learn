import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.Automatic_env import SimpleEnv2 #!修改自定义场景
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from yolo import process_episode_frames

#todo基础设置
#* 数据集配置
REPO_NAME = 'qunikin_language' #仓库名称
NUM_DEMO = 3 #数据集数量
ROOT = "./demo_data" #数据集文件
#* 场景配置
SEED = 0 #固定种子
# SEED = None #随机种子
xml_path = './asset/Automation_scene/scene.xml' #!场景文件
PnPEnv = SimpleEnv2(xml_path, seed = SEED, state_type = 'joint_angle') #创建环境

#todo数据集特征
create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False
if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                fps=20,
                root = ROOT, 
                robot_type="omy",
                tolerance_s = 1e-4,
                
                image_writer_threads=6,
                image_writer_processes=4,
                use_videos = True,
                video_backend = None,
                batch_encoding_size = 1,
                vcodec = "h264",
                metadata_buffer_size = 10,
                streaming_encoding = False,
                encoder_queue_maxsize=60,
                encoder_threads=6,
                
                features={#!数据集特征
                    "observation.image": {
                        "dtype": "video",#!自动编码视频
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 joint angles and 1 gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (9,),
                        "names": ["obj_init"], # just the initial position of the object. Not used in training.
                    },
                },
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)
    
#todo采集数据
action = np.zeros(7)
episode_id = 0
record_flag = False
episode_frames = []
try:
    while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
        PnPEnv.step_env()
        if PnPEnv.env.loop_every(HZ=20):
            # check if the episode is done
            done = PnPEnv.check_success()
            if done: 
                if episode_frames:
                    processed_frames = process_episode_frames(
                        episode_frames,
                        episode_id,
                        output_dir=os.path.join(ROOT, f"episode_{episode_id}_yolo"),
                        object_class_name='cup',
                        alpha=0.5,
                        save_overlay=False,
                        save_disk=False,
                    )
                    for frame in processed_frames:
                        dataset.add_frame({
                                "observation.image": frame["agent_image"],
                                "observation.wrist_image": frame["wrist_image"],
                                "observation.state": frame["state"],
                                "action": frame["action"],
                                "obj_init": frame["obj_init"],
                                "task": frame["task"],
                            }
                        )
                    episode_frames = []
                dataset.save_episode()
                PnPEnv.reset()
                episode_id += 1
            # Teleoperate the robot and get delta end-effector pose with gripper
            action, reset  = PnPEnv.teleop_robot()
            if not record_flag and sum(action) != 0:
                record_flag = True
                print("Start recording")
            if reset:
                # Reset the environment and clear the episode buffer
                # This can be done by pressing 'z' key
                # PnPEnv.reset(seed=SEED)
                PnPEnv.reset()
                dataset.clear_episode_buffer()
                episode_frames = []
                record_flag = False
            # Step the environment
            # Get the end-effector pose and images
            agent_image,wrist_image = PnPEnv.grab_image()
            # # resize to 256x256
            agent_image = Image.fromarray(agent_image)
            wrist_image = Image.fromarray(wrist_image)
            agent_image = agent_image.resize((256, 256))
            wrist_image = wrist_image.resize((256, 256))
            agent_image = np.array(agent_image)
            wrist_image = np.array(wrist_image)
            joint_q = PnPEnv.step(action)
            action = PnPEnv.q[:7] # 6 joint angles and 1 gripper
            action = action.astype(np.float32)
            if record_flag:
                raw_frame = {
                    "agent_image": agent_image.copy(),
                    "wrist_image": wrist_image.copy(),
                    "state": joint_q[:7],
                    "action": action,
                    "obj_init": PnPEnv.obj_init_pose,
                    "task": PnPEnv.instruction,
                    "frame_index": len(episode_frames),
                }
                episode_frames.append(raw_frame)
            PnPEnv.render(teleop=True, idx=episode_id)  
finally:
    PnPEnv.env.close_viewer()
    dataset.finalize()
