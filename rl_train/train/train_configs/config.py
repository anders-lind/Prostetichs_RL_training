from dataclasses import dataclass, field

# 1. ADD THIS NEW DATACLASS
@dataclass
class SACParams:
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000       # Size of the replay buffer
    learning_starts: int = 10000       # How many steps to collect before training
    batch_size: int = 256              # Batch size for training
    tau: float = 0.005                 # Soft update coefficient
    gamma: float = 0.99                # Discount factor
    train_freq: int = 1                # Update the model every 'train_freq' steps
    gradient_steps: int = 1            # How many gradient steps to do per update
    use_sde: bool = False              # Whether to use gSDE for exploration
    sde_sample_freq: int = -1          # Sample new SDE noise every n steps
    device: str = "cpu"
    

@dataclass
class TrainSessionConfigBase:
    total_timesteps: int = 1000
    @dataclass
    class LoggerParams:
        logging_frequency: int = int(1)
        evaluate_frequency: int = int(64)
    logger_params: LoggerParams = field(default_factory=LoggerParams)
    
    @dataclass
    class EnvParams:
        @dataclass
        class RewardWeights:
            forward_reward: float = 0.01
            muscle_activation_penalty: float = 0.1
            muscle_activation_diff_penalty: float = 0.1

            # for reward per step
            footstep_delta_time:float = 0.0
            average_velocity_per_step:float = 0.0
            muscle_activation_penalty_per_step:float = 0.0

            joint_constraint_force_penalty: float = 0.0

            foot_force_penalty: float = 0.0
        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)
        
        env_id: str = ""
        num_envs: int = 1
        seed: int = 0
        safe_height: float = 0.65
        control_framerate: int = 30
        physics_sim_framerate: int = 1200
        
        min_target_velocity: float = 0.5
        max_target_velocity: float = 3.0
        min_target_velocity_period: float = 3
        max_target_velocity_period: float = 5

        custom_max_episode_steps: int = 500
        model_path: str = None
        prev_trained_policy_path: str = None
        reference_data_path: str = ""

        enable_lumbar_joint: bool = False
        lumbar_joint_fixed_angle: float = 0.0
        lumbar_joint_damping_value: float = 0.05

        observation_joint_pos_keys: list[str] = field(default_factory=list)
        observation_joint_vel_keys: list[str] = field(default_factory=list)
        observation_joint_sensor_keys: list[str] = field(default_factory=list)

        # terrain type: flat, random, sinusoidal, harmonic_sinusoidal, uphill, downhill, dev
        terrain_type: str = "flat"
        terrain_params: str = ""
        
    env_params: EnvParams = field(default_factory=EnvParams)

    evaluate_param_list: list[dict] = field(default_factory=list[dict])

    # 2. REMOVE THE PolicyParams DATACLASS (it was here)
    
    # 3. REMOVE THE PPOParams DATACLASS (it was here)

    # 4. REPLACE policy_params and ppo_params WITH sac_params
    sac_params: SACParams = field(default_factory=SACParams)