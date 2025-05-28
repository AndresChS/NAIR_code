#=================================================
#	This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
#reference.
#	Model	  :: MyoLeg 1 Dof 40 Musc Exo (MuJoCoV2.0)
#	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
#	source	:: https://github.com/AndresChS/NAIR_Code
#	====================================================== -->
import collections
from myosuite.utils import gym
import numpy as np
import mujoco
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs import env_base
import torch.nn
import random
import pickle

class Myoleg_env_v0(BaseV0,env_base.MujocoEnv):

    MYO_CREDIT = """\
    This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close reference.
        Model	:: MyoLeg 1 Dof 40 Musc Exo (MuJoCoV2.0)
        Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
        Source	:: https://github.com/AndresChS/NAIR_Code
    """
    
    DEFAULT_OBS_KEYS = ['time', 'qpos', 'qvel', 'qacc', 'act', 'pose_err', 'inter_force']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 5,
        "inter_force": 0,
        "smooth_act": 0,
        "acceleration": 0,
        "penalty_high": 1000.0,
        "penalty_low": 1000.0
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)

    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            viz_site_targets:tuple = None,  # site to use for targets visualization []
            target_jnt_range: dict = {'joint_range' :(0, 90)},   # joint ranges as tuples {name:(min, max)}_nq
            target_jnt_value: list = None,   # desired joint vector [des_qpos]_nq
            reset_type = "init",            # none; init; random
            target_type = "fixed",       # generate; switch; fixed
            target_qpos = 0.5, # Desired knee_angle_r position
            spasticity = "passive",
            spasticity_level = 1,
            sigma = 0.5,
            weight_bodyname = None,
            weight_range = None,
            **kwargs,
        ):
        #Memmory variables
        self.action_prev1 = 0# np.asarray([self.sim.data.ctrl[0]])
        self.action_prev2 = 0# np.asarray([self.sim.data.ctrl[0]])
        self.excess_force_prev = 0
        self.cumulative_reward_value = 0
        self.qacc_prev = 0

        self.reset_type = reset_type
        self.target_type = target_type
        self.sigma = sigma
        self.spasticity = spasticity
        self.spasticity_level = spasticity_level
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range
        self.target_qpos = []
        self.target_qpos.append(target_qpos)
        print('target_qpos:',target_qpos)
        # resolve joint demands

        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value
        
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=viz_site_targets,
                **kwargs,
                )
   
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos
        obs_dict['qvel'] = sim.data.qvel
        obs_dict['qacc'] = sim.data.qacc
        obs_dict['act'] = np.asarray([sim.data.ctrl[0]])#if sim.model.na>0 else np.zeros_like(obs_dict['qpos'])
        obs_dict['qfrc_actuator'] = sim.data.qfrc_actuator
        obs_dict['qfrc_constraint'] = sim.data.qfrc_constraint
        obs_dict['pose_err'] = np.absolute(self.target_qpos - sim.data.qpos[6])
        obs_dict['inter_force'] = np.asarray([self.get_inter_force(pos=sim.data.qpos[6], torque=sim.data.qfrc_constraint[6])])
        return obs_dict

    def squeeze_obs_dict(self, obs_dict):
        obs_dict_squeeze = {}
        obs_dict_squeeze['time'] = np.squeeze(obs_dict['time'])
        obs_dict_squeeze['qpos'] = np.squeeze(obs_dict['qpos'])
        obs_dict_squeeze['qacc'] = np.squeeze(obs_dict['qacc'])
        obs_dict_squeeze['qvel'] = np.squeeze(obs_dict['qvel'])
        obs_dict_squeeze['act'] = np.squeeze(obs_dict['act'])
        obs_dict_squeeze['qfrc_actuator'] = np.squeeze(obs_dict['qfrc_actuator'])
        obs_dict_squeeze['qfrc_constraint'] = np.squeeze(obs_dict['qfrc_constraint'])
        obs_dict_squeeze['pose_err'] = np.squeeze(obs_dict['pose_err'])
        obs_dict_squeeze['inter_force'] = np.squeeze(obs_dict['inter_force'])
        return obs_dict_squeeze

    def get_inter_force(self, pos, torque):
        m = 68 / 1.57
        torque_pos = m * pos - 64
        torque_ext = torque - torque_pos
        #print("Position",pos, "   Torque", torque, "Torque ext:", torque_ext)
        return torque_ext

    
    def cumulative_rw(self):
        self.cumulative_reward_value += 0.01
        return self.cumulative_reward_value
    def reset_cumulative_reward(self):
        self.cumulative_reward_value = 0
    
    def get_pose_reward(self, pose_err, vel):
        # generates pose reward based on qpos
        pose_reward = np.exp((-(1.*pose_err))/(2.*np.power(self.sigma,2)))
        lineal_penalty = np.linalg.norm(pose_err)
        pose_reward =  pose_reward - lineal_penalty

        if (pose_err < 0.05) and (vel< 0.05):
            pose_reward += self.cumulative_rw()
        else:
            self.reset_cumulative_reward()
        return pose_reward
    
    
    def get_inter_force_penalty(self, torque, force_threshold):
        
        excess_force = torque
        delta_excess_force = excess_force - self.excess_force_prev
        inter_force_penalty = np.linalg.norm(delta_excess_force)**2
        #self.excess_force_prev = excess_force
        # Apply exponential penalty only if the excess force is positive
        """if excess_force > 0:
            inter_force_penalty = excess_force/10
        else:
            inter_force_penalty = 0  # No penalty if below threshold
        """
        return inter_force_penalty
    
    def get_acc_penalty(self, exo_acc):
        acc_penalty = np.linalg.norm(exo_acc - self.qacc_prev)**2
        self.qacc_prev = exo_acc
        return acc_penalty
    
    def get_action_penalty(self):
        #print(self.action_prev1, self.action_prev2)
        action_penalty = np.linalg.norm(self.action_prev1*5 - self.action_prev2*5)**2
        #action_penalty = self.action_prev1[0]*10 - self.action_prev2[0]*10
        #print(action_penalty)
        return action_penalty
    
    def get_reward_dict(self, obs_dict):

        obs_dict_squeeze = self.squeeze_obs_dict(obs_dict)
        actual_pos = obs_dict_squeeze['qpos'][6]
        pose_err = obs_dict_squeeze['pose_err']
        knee_vel = obs_dict_squeeze['qvel'][6]
        inter_force_leg = obs_dict_squeeze['inter_force']
        knee_constraint = obs_dict_squeeze['qfrc_constraint'][6]
        act_exo = obs_dict_squeeze['act']
        exo_acc = obs_dict_squeeze['qacc'][6]
        force_threshold = 25

        #Pose reward
        pose_reward = self.get_pose_reward(pose_err=pose_err, vel=knee_vel)
        #Acc penalty
        acc_penalty = self.get_acc_penalty(exo_acc)
        #Interaction penalty
        inter_force_penalty = self.get_inter_force_penalty(torque=knee_constraint, force_threshold=force_threshold)
        #Action penalty 
        action_penalty=self.get_action_penalty()
        #Done conditions
        done_high = ((actual_pos > self.target_jnt_range[0][1]) & (inter_force_leg > force_threshold)).astype(float)
        done_low = ((actual_pos < self.target_jnt_range[0][0]) & (inter_force_leg > force_threshold)).astype(float)
        rwd_dict = collections.OrderedDict((

            # Optional Keys
            ('pose',    1.*pose_reward),
            ('acceleration',    -1.*acc_penalty),
            ('inter_force',  -1.*inter_force_penalty),
            ('penalty_high', -1.*(actual_pos > self.target_jnt_range[0][1]).astype(float)),
            ('penalty_low', -1.*(actual_pos < self.target_jnt_range[0][0]).astype(float)),
            ('smooth_act', -1*(action_penalty)),
            
            # Must keys
            ('sparse',  0),
            ('solved',  pose_err<self.sigma*0.1),
            ('done', (done_high + done_low).astype(float)), 
        ))       
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        #print("Reward", rwd_dict['dense'], "Error", pose_err, "Pos_score", pose_reward*5, "Inter_penalty:", -0.001*(inter_force_penalty), "acceleration_penalty", -0.001*acc_penalty)
        return rwd_dict

    def random_init(self, **kwargs):
        with open('/Users/achs/Documents/PHD/code/NAIR_Code/envs/myosuite/nair_envs/nair_envs/angles_dict.pkl', 'rb') as file:
            angles_dict = pickle.load(file)
        init_pos = random.randint(0, 90)
        random_qpos = angles_dict[init_pos][1:]
        print("random pos:", init_pos)
        obs = super().reset(reset_qpos=random_qpos, **kwargs)
        return obs
    
    def set_target_qpos(self, target_qpos):
        target_qpos = target_qpos*(2*np.pi/360)
        if isinstance(target_qpos, torch.Tensor):
            self.target_qpos = np.array([target_qpos.item()])
        else:
            self.target_qpos = np.array([target_qpos])
        print('New target qpos:', self.target_qpos)

    # reset_type = none; init; random
    # target_type = generate; fixed
    def reset(self, **kwargs):
        # update init state
        if self.reset_type is None or self.reset_type == "none":
            # no reset; use last state
            ## NOTE: fatigue is also not reset in this case!
            obs = self.get_obs()
        elif self.reset_type == "init":
            # reset to init state
            obs = super().reset(**kwargs)
        elif self.reset_type == "random":
            # reset to random state
            #jnt_init = self.np_random.uniform(high=self.sim.model.jnt_range[:,1], low=self.sim.model.jnt_range[:,0])
            obs = self.random_init()
    
        elif self.reset_type == "fixed": 
             # position with knee angle = 90º
             qpos = [0.005082630840485775, 0.033695272921116896, -1.6419641702281853, -2.978484348923629, 0.006955509476829092, 0.001288023442163489, 1.5439362618219066, 0.012157153984297496, 0.2594406777004281, 1.6269485818730451, -0.6989795193633237, 0.35052974007072063, -0.5246066599479225, -0.0340516687698901, -0.008846460674014444, -1.238730497394359]
             # position for knee angle = 0º
             #qpos = [0.005834685602873783, -0.00653851963837641, -1.610314825968498, -0.6692592235178847, 0.00566594464090885, 0.0009114920866788922, 1.4722420075124696, 0.006470171955450531, 0.056191264122378036, -0.1603820765979743, -0.4011910332346481, 0.3400333691081888, -0.16811998188958785, -0.017757503091114304, 0.02614206520113357, -1.3097044945216785]
             obs = super().reset(reset_qpos=qpos, **kwargs)
        else:
            print("Reset Type not found")
        
        if self.target_type == 'generate':
            self.set_target_qpos(target_qpos=random.randint(0, 90))
        elif self.target_type == 'fixed':
            self.target_qpos = self.target_qpos
        
        if self.spasticity == 'active':
            self.spasticity_level = random.randint(0,3) 
            #print("Spasticity level:", self.spasticity_level)
        
        return obs

    def set_env_state(self, qpos, **kwargs):
        """
        Set full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        obs = super().reset(reset_qpos=qpos, **kwargs)

        return obs
    
    def ad_spasticity(self, muscle_a):
        #   in sim.data.ctrl, [0] is for exo_motor and [1:40] for 40 muscles
        #    Spastic Muscles
        #    # Harmstrings
        #    muscle_a[32]  #Semimembranosus
        #    muscle_a[33]  #Semitendinosus
        #    muscle_a[7]   #Biceps Sural Long
        #    muscle_a[8]   #Biceps Sural Short
        #    # Calf
        #    muscle_a[13]  #Gastrocnemius Lateral
        #    muscle_a[14]  #Gastrocnemius Medial
        #    # Rectus 
        #    muscle_a[24]  #Gracillis
        #    # Quadriceps
        #    muscle_a[40]  #Vastus medial
        #    muscle_a[39]  #Vastus lateral
        #    muscle_a[38]  #Vastus intermedius
        #    muscle_a[30]  #Rectus femoral   
        #print(self.sim.data.ten_length)
        knee_angle = (360/2*np.pi)*self.sim.data.qpos[6]
        knee_vel = self.sim.data.qvel[6]

        # Modify parameters following the spasticity level
        if self.spasticity_level == 0:
            # Level 0: No spasticity
            k_ang = 5
            a0_flexion = 0
            a0_extension = 90
            L_ang = 0
            L_vel = 0 
            vel_limit = 10  
        elif self.spasticity_level == 1:
            # Level 1: Spasticity in the last 15 degrees of joint range
            k_ang = 0.5
            a0_flexion = 10
            a0_extension = 80
            L_ang = 0.2
            L_vel = 0.2
            vel_limit = 4
        elif self.spasticity_level == 2:
            # Level 2: Spasticity in all range, low level, velocity-dependent
            k_ang = 0.2
            a0_flexion = 15
            a0_extension = 75
            L_ang = 0.3
            L_vel = 0.2
            vel_limit = 2
        elif self.spasticity_level == 3:
        # Level 3: Spasticity in all range, moderate level
            k_ang = 0.1
            L_ang = 0.5
            a0_flexion = 20
            a0_extension = 70
            L_vel = 0.3
            vel_limit = 1
        
        # Sigmoid application Angle dependent
        # Sigmoid ecuation for flexion and extension (range)
        sigmoid_flexion_ang = 1 / (1 + np.exp(-k_ang * (knee_angle - a0_flexion)))  # Sigmoide izquierda
        sigmoid_extension_ang = 1 / (1 + np.exp(np.clip(k_ang * (knee_angle - a0_extension), -500, 500)))  # Sigmoide derecha
        spas_coef_ang =L_ang + (L_ang * (1 - (sigmoid_flexion_ang + sigmoid_extension_ang)))
        
        # Sigmoid application Velocity dependent
        k = 5
        v0_flexion = -vel_limit
        v0_extension = vel_limit
        slope_increase = 0.05
        # Sigmoid ecuation for flexion and extension (velocity)
        sigmoid_flexion = 1 / (1 + np.exp(-k * (knee_vel - v0_flexion)))  # Sigmoide izquierda
        sigmoid_extension = 1 / (1 + np.exp(k * (knee_vel - v0_extension)))  # Sigmoide derecha
        spas_coef_vel = L_vel + (L_vel * (1 - (sigmoid_flexion + sigmoid_extension)))
        spas_coef_vel = spas_coef_vel + slope_increase * (v0_flexion - knee_vel)
        spas_coef_vel = spas_coef_vel + slope_increase * (knee_vel - v0_extension)
        # Merge coef  
        spas_coef = spas_coef_vel + spas_coef_ang
        
        if knee_vel > 0:    # Flexion
            muscles_index = [7,8,13,14,24,32,33,38,39,40]
            muscle_a[muscles_index] += spas_coef  
        else:               # Extension
            muscles_index = [7,8,13,14,24,32,33,38,39,40]
            muscle_a[muscles_index] += spas_coef  

        return muscle_a

                
    def step(self, a, **kwargs):
        
        muscle_a = a.copy()
        #print(muscle_a, "-----")
        muscle_act_ind = self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        # Explicitely project normalized space (-1,1) to actuator space (0,1) if muscles
        
        muscle_a = self.ad_spasticity(muscle_a)
        
        if self.sim.model.na and self.normalize_act:
            # find muscle actuators
            muscle_a[muscle_act_ind] = 1.0 / (
                1.0 + np.exp(-5.0 * (muscle_a[muscle_act_ind] - 0.5))
            )
            # TODO: actuator space may not always be (0,1) for muscle or (-1, 1) for others
            isNormalized = (
                False  # refuse internal reprojection as we explicitly did it here
            )
        else:
            isNormalized = self.normalize_act  # accept requested reprojection
        #print(muscle_a, type(muscle_a))
        #Action smooth filter
        self.action_prev2 = self.action_prev1
        self.action_prev1 = muscle_a[0]
        # step forward
        self.last_ctrl = self.robot.step(
            ctrl_desired=muscle_a,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )
        #print(muscle_a, "-----")
        self.excess_force_prev = self.sim.data.qfrc_constraint[6]
        
        return self.forward(**kwargs)