import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv2:
    #todo初始化
    def __init__(self, 
                 xml_path,
                action_type='ee_pose', 
                state_type='joint_angle',
                seed = None):
        """
        args:
            xml_path: str, path to the xml file #*  XML文件的路径
            action_type: str, type of action space, 'ee_pose','delta_joint_angle' or 'joint_angle' #*  动作空间类型，可以是'末端执行器位姿'、'关节角变化'或'关节角'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose' #*  状态空间类型，可以是'关节角'或'末端执行器位姿'
            seed: int, seed for random number generator #*随即化初始位置，0利于复现，NONE随机利于泛化
        """
        #* 加载 xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        #*初始化机械臂关节
        self.joint_names = ['joint1',
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint6',]
        #*初始化查看器，物体和场景
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    
    #todo重置
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        #* 随机化初始位置
        if seed != None: np.random.seed(seed=0) 
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'tcp_link',
            q_init       = q_init, # ik from zero pose
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)
        
        #* 盘子位置和旋转
        plate_xyz = np.array([0.3, -0.25, 0.82])
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        #*设置物体位置 Set object positions
        # #红色杯子位置和旋转
        obj_xyzs = sample_xyzs(
            1,
            x_range   = [+0.32,+0.33],
            y_range   = [-0.00,+0.02],
            z_range   = [0.83,0.83],
            min_dist  = 0.16,
            xy_margin = 0.0
        )
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=obj_xyzs[0,:])
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        # 蓝色杯子位置和旋转
        obj_xyzs = sample_xyzs(
            1,
            x_range   = [+0.29,+0.3],
            y_range   = [0.19,+0.21],
            z_range   = [0.83,0.83],
            min_dist  = 0.16,
            xy_margin = 0.0
        )
        self.env.set_p_base_body(body_name='body_obj_mug_6',p=obj_xyzs[0,:])
        self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        #* 初始化机器人位置和旋转
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0]*4)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_red_init_pose, mug_blue_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []
        
    #todo生成语言指令
    def set_instruction(self, given = None):#*如果给定指令则使用给定指令，否则随机生成指令
        """
        Set the instruction for the task
        """
        if given is None:
            obj_candidates = ['red', 'blue']
            obj1 = random.choice(obj_candidates)
            self.instruction = f'Place the {obj1} mug on the plate.' #*  根据对象名称生成指令，将特定颜色的杯子放在盘子上
            if obj1 == 'red':
                self.obj_target = 'body_obj_mug_5'#*设置目标物体
            else:
                self.obj_target = 'body_obj_mug_6'
        else:
            self.instruction = given
            if 'red' in self.instruction:
                self.obj_target = 'body_obj_mug_5'
            elif 'blue' in self.instruction:
                self.obj_target = 'body_obj_mug_6'
            else:
                raise ValueError('Instruction does not contain a valid object color (red or blue).')

    #todo关节运动方式
    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]#*末端执行器
                - joint_angle: [j1,j2,j3,j4,j5,j6]#*关节绝对角度
                - delta_joint_angle: [dj1,dj2,dj3,dj4,dj5,dj6]#*关节相对角度

        '''
        if self.action_type == 'ee_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'tcp_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        gripper_cmd = np.array([action[-1]]*4)
        gripper_cmd[[1,3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False, idx = 0):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        self.env.plot_T(p = np.array([0.1,0.0,1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        if getattr(self, 'instruction', None) is not None:
            language_instructions = self.instruction
            self.env.viewer_text_overlay(text1='Language Instructions',text2=language_instructions)
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    #todo 按键控制
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)
    
    #todo检查是否放置成功
    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body(self.obj_target)#*根据前面设定的目标物体
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint('rh_r1') < 0.1:
            p = self.env.get_p_body('tcp_link')[2]
            if p > 0.9:
                return True
        return False
    #todo获取位置
    def get_obj_pose(self):
        '''
        returns: 
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        p_mug_red = self.env.get_p_body('body_obj_mug_5')
        p_mug_blue = self.env.get_p_body('body_obj_mug_6')
        p_plate = self.env.get_p_body('body_obj_plate_11')

        return p_mug_red, p_mug_blue, p_plate
    
    def set_obj_pose(self, p_mug_red, p_mug_blue, p_plate):
        '''
        Set the object poses
        args:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug_red)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_mug_6',p=p_mug_blue)
        self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)