"""Session configuration dataclasses used by the rl training code.

This module contains small, JSON-serializable dataclasses parsed from the
training configuration files. Types are explicit where useful to help
readability and static analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List


@dataclass
class TrainSessionConfigBase:
    total_timesteps: int = 1000

    @dataclass
    class LoggerParams:
        logging_frequency: int = 1
        evaluate_frequency: int = 64

    logger_params: LoggerParams = field(default_factory=LoggerParams)

    @dataclass
    class EnvParams:
        @dataclass
        class RewardWeights:
            forward_reward: float = 0.01
            muscle_activation_penalty: float = 0.1
            muscle_activation_diff_penalty: float = 0.1

            # per-step reward scalars
            footstep_delta_time: float = 0.0
            average_velocity_per_step: float = 0.0
            muscle_activation_penalty_per_step: float = 0.0

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
        min_target_velocity_period: int = 3
        max_target_velocity_period: int = 5

        custom_max_episode_steps: int = 500
        model_path: Optional[str] = None
        prev_trained_policy_path: Optional[str] = None
        reference_data_path: Optional[str] = None

        enable_lumbar_joint: bool = False
        lumbar_joint_fixed_angle: float = 0.0
        lumbar_joint_damping_value: float = 0.05

        observation_joint_pos_keys: List[str] = field(default_factory=list)
        observation_joint_vel_keys: List[str] = field(default_factory=list)
        observation_joint_sensor_keys: List[str] = field(default_factory=list)

        terrain_type: str = "flat"
        terrain_params: str = ""

    env_params: EnvParams = field(default_factory=EnvParams)

    evaluate_param_list: List[Dict[str, Any]] = field(default_factory=list)

    @dataclass
    class PolicyParams:
        @dataclass
        class CustomPolicyParams:
            # Curriculum/reset flags
            reset_shared_net_after_load: bool = False
            reset_policy_net_after_load: bool = False
            reset_value_net_after_load: bool = False

            # net_arch maps named sub-networks to layer sizes
            net_arch: Dict[str, List[int]] = field(default_factory=dict)
            log_std_init: float = -2.0

            # net_indexing_info describes how to slice the flattened observation
            # and action vectors for sub-networks; it's intentionally typed as
            # Any to allow flexible JSON config shapes.
            net_indexing_info: Dict[str, Any] = field(default_factory=dict)

        custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)

    policy_params: PolicyParams = field(default_factory=PolicyParams)

    @dataclass
    class A2CParams:
        # These fields exist partly for backward-compatibility with older
        # configs; only A2C-relevant fields will be used when constructing
        # the model.
        learning_rate: float = 3e-4
        n_steps: int = 5
        # Legacy fields retained for config-compatibility
        batch_size: int = 2048
        n_epochs: int = 10
        gamma: float = 0.99
        gae_lambda: float = 0.95
        clip_range: float = 0.2
        clip_range_vf: float = 0.2
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 0.5
        use_sde: bool = False
        sde_sample_freq: int = -1
        target_kl: Optional[float] = None
        device: str = "cpu"

    a2c_params: A2CParams = field(default_factory=A2CParams)
from typing import Optional, Any, Dict, List



