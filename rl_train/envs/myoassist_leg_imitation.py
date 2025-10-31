import collections
import numpy as np
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
################################################################


class ImitationCustomLearningCallback(BaseCustomLearningCallback):
    
    def __init__(self, *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler:train_log_handler.TrainLogHandler,
                 original_reward_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights,
                 auto_reward_adjust_params:ImitationTrainSessionConfig.AutoRewardAdjustParams,
                 verbose=1):
        super().__init__(log_rollout_freq=log_rollout_freq, 
                         evaluate_freq=evaluate_freq,
                         log_handler=log_handler,
                         verbose=verbose)
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params
        

    def _init_callback(self):
        super()._init_callback()

        self.reward_accumulate = DictionableDataclass.create(ImitationTrainSessionConfig.EnvParams.RewardWeights)
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0

    # called after all envs step done
    def _on_step(self) -> bool:
        
        # --- 1. Original _on_step logic: Accumulate custom rewards ---
        for info in self.locals["infos"]:
            for key in self.reward_accumulate.keys():
                self.reward_accumulate[key] += info["rwd_dict"][key]
        
        # --- 2. Call parent's _on_step ---
        super()._on_step() 
            
        # --- 3. NEW: Moved logic from _on_rollout_end ---
        if self.n_calls % self.log_rollout_freq == 0:
            
            # Check if any episodes actually finished during this log period
            if np.sum(self.episode_counts) == 0:
                return True # Continue training, nothing to log

            # Calculate stats
            rollout_reward_mean = np.sum(self.rewards_sum) / np.sum(self.episode_counts)
            rollout_ep_len_mean = np.sum(self.episode_length_counts) / np.sum(self.episode_counts)
            
            # --- 5. Original logging logic from _on_rollout_end ---
            log_data = ImitationTrainCheckpointData(
                num_timesteps=self.num_timesteps,
                average_reward_per_episode=rollout_reward_mean,
                average_num_timestep=rollout_ep_len_mean,
                
                # Add the imitation-specific fields
                reward_weights=DictionableDataclass.to_dict(self._reward_weights),
                reward_accumulate=self.reward_accumulate.copy(),
            )
            
            # Write the log file
            self.train_log_handler.add_log_data(log_data)
            self.train_log_handler.write_json_file()
            
            # --- 6. Reset ALL counters ---
            self.rewards_sum = np.zeros(self.training_env.num_envs)
            self.episode_counts = np.zeros(self.training_env.num_envs)
            self.episode_length_counts = np.zeros(self.training_env.num_envs)
            
            for key in self.reward_accumulate.keys():
                self.reward_accumulate[key] = 0
            
        return True # Continue training

##############################################################################
#
#  MyoAssistLegImitation Class
#
##############################################################################

class MyoAssistLegImitation(MyoAssistLegBase):
    
    # automatically inherit from MyoAssistLegBase
    # DEFAULT_OBS_KEYS = ['qpos',
    #                     'qvel',
    #                     'act',
    #                     'target_velocity',
    #                     ]

    def _setup(self,*,
            env_params:ImitationTrainSessionConfig.EnvParams,
            reference_data:dict|None = None,
            loop_reference_data:bool = False,
            **kwargs,
        ):
        self._flag_random_ref_index = env_params.flag_random_ref_index
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self.reference_data_keys = env_params.reference_data_keys
        self._loop_reference_data = loop_reference_data
        self._reward_keys_and_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights = env_params.reward_keys_and_weights

        self.setup_reference_data(data=reference_data)

        super()._setup(env_params=env_params,
                       **kwargs,
                       )

        
        
    def set_reward_weights(self, reward_keys_and_weights:TrainSessionConfigBase.EnvParams.RewardWeights):
        self._reward_keys_and_weights = reward_keys_and_weights
    # override from MujocoEnv
    def get_obs_dict(self, sim):
        return super().get_obs_dict(sim)

    def _get_qpos_diff(self) -> dict:

        def get_qpos_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qpos[0].copy() - self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict
    def _get_qvel_diff(self):
        speed_ratio_to_target_velocity = self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]

        def get_qvel_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qvel[0].copy() - self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            # joint_weight = self._reward_keys_and_weights.qvel_imitation_rewards[q_key]
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict
    def _get_qpos_diff_nparray(self):
        return np.array([diff for diff in self._get_qpos_diff().values()])
    def _get_end_effector_diff(self):
        # body_pos = self.sim.data.body('pelvis').xpos.copy()
        # diff_array = []
        # for mapping in self.ANCHOR_SIM_TO_REF.values():
        #     sim_anchor = self.sim.data.joint(mapping.sim_name).xanchor.copy() - body_pos
        #     ref_anchor = self._reference_data[mapping.ref_name][self._imitation_index]
        #     diff = np.linalg.norm(sim_anchor - ref_anchor)
        #     diff_array.append(diff)
        # return diff_array
        return np.array([0])
    
    def _calculate_imitation_rewards(self, obs_dict):
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # Calculate joint position rewards
        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            q_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            dq_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))
        
        # Calculate end effector reward
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        # Calculate joint imitation rewards sum
        qpos_imitation_rewards = np.sum([q_reward_dict[key] * self._reward_keys_and_weights.qpos_imitation_rewards[key] for key in q_reward_dict.keys()])
        qvel_imitation_rewards = np.sum([dq_reward_dict[key] * self._reward_keys_and_weights.qvel_imitation_rewards[key] for key in dq_reward_dict.keys()])

        # Add new key-value pairs to the base_reward dictionary
        base_reward.update({
            'qpos_imitation_rewards': qpos_imitation_rewards,
            'qvel_imitation_rewards': qvel_imitation_rewards,
            'end_effector_imitation_reward': anchor_reward
        })

        # Use the updated base_reward as imitation_rewards
        imitation_rewards = base_reward
        info = base_info
        return imitation_rewards, info
    

    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):
        # Calculate common rewards
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        # Construct reward dictionary
        # Automatically add all imitation_rewards items to rwd_dict
        rwd_dict = collections.OrderedDict((key, imitation_rewards[key]) for key in imitation_rewards)

        # Add additional fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),
        })
        # Calculate final reward
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0)
        
        return rwd_dict
    

    def _follow_reference_motion(self, is_x_follow:bool):
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            if not is_x_follow and key == 'pelvis_tx':
                self.sim.data.joint(f"{key}").qpos = 0
            # if key == 'pelvis_ty':
            #     self.sim.data.joint(f"{key}").qpos += 0.05
        speed_ratio_to_target_velocity = self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
    def imitation_step(self, is_x_follow:bool, specific_index:int|None = None):
        if specific_index is None:
            self._imitation_index += 1
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        self._follow_reference_motion(is_x_follow)
        # should call this but I don't know why
        # next_obs, reward, terminated, truncated, info = super().step(np.zeros(self.sim.model.nu))
        # return (next_obs, reward, False, False, info)
        self.forward()
        return self._imitation_index
        # pass
    
    # override
    def step(self, a, **kwargs):
        if self._imitation_index is not None:
            self._imitation_index += 1
            if self._imitation_index < self._reference_data_length:
                is_out_of_index = False
            else:
                if self._loop_reference_data:
                    self._imitation_index = 0
                    is_out_of_index = False
                else:
                    is_out_of_index = True
                    self._imitation_index = self._reference_data_length - 1
        else:
            is_out_of_index = True
        
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        if is_out_of_index:
            reward = 0
            truncated = True
        else:
            q_diff_nparray:np.ndarray = self._get_qpos_diff_nparray()
            is_out_of_trajectory = np.any(np.abs(q_diff_nparray) >self._out_of_trajectory_threshold)
            terminated = terminated or is_out_of_trajectory
        
        return (next_obs, reward, terminated, truncated, info)
        
    
    def setup_reference_data(self, data:dict|None):
        self._reference_data = data
        self._imitation_index = None
        if data is not None:
            # self._follow_reference_motion(False)
            self._reference_data_length = self._reference_data["metadata"]["resampled_data_length"]
            
            # --- ADD THIS LINE ---
            print(f"\n\nDEBUG: self._reference_data_length = {self._reference_data_length}\n\n")
            # --- END ---
            
        else:
            raise ValueError("Reference data is not set")

    def reset(self, **kwargs):
        rng = np.random.default_rng()# TODO: refactoring random to use seed
        
        if self._flag_random_ref_index:
            #
            # --- THIS IS THE FIXED LINE ---
            #
            self._imitation_index = rng.integers(0, int(self._reference_data_length * 0.8))
        else:
            self._imitation_index = 0
        # generate random targets
        # new_qpos = self.generate_qpos()# TODO: should set qvel too.
        # self.sim.data.qpos = new_qpos
        self._follow_reference_motion(False)
        
        obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qpos, **kwargs)
        return obs

    # override
    def _initialize_pose(self):
        super()._initialize_pose()