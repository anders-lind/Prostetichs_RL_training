import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.sac.policies import Actor, SACPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_train.train.policies.network_index_handler import NetworkIndexHandler
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.train.policies.rl_agent_exo_features import HumanExoFeaturesExtractor

class HumanExoSACNetwork(Actor):
    """Custom network for SAC that handles both human and exo control."""
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        custom_policy_params: ExoImitationTrainSessionConfig.PolicyParams.CustomPolicyParams,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ):
        # Initialize parent Actor with flattened architecture
        # Create the features extractor if not provided
        if features_extractor is None:
            features_extractor = HumanExoFeaturesExtractor(observation_space)
        
        # Get feature dimensions from the extractor
        features_dim = features_extractor.features_dim
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=[256, 256],  # Placeholder, we'll use our own architecture
            features_extractor=features_extractor,
            features_dim=features_dim,
            normalize_images=False
        )
        # Store our custom architecture
        self.net_arch = custom_policy_params.net_arch
        self.net_indexing_info = custom_policy_params.net_indexing_info
        self.network_index_handler = NetworkIndexHandler(self.net_indexing_info, observation_space, action_space)

        # Build the actor networks (mu, log_std)
        self._build_actor_networks()
        # Build the dual Q-networks for critics
        self._build_critic_networks()

    def _build_mlp(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
        """Helper to build an MLP with given dimensions."""
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            last_dim = dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_actor_networks(self):
        """Build separate networks for human and exo control."""
        # Human actor network
        human_obs_dim = self.network_index_handler.get_observation_num("human_actor")
        human_act_dim = self.network_index_handler.get_action_num("human_actor")
        self.human_actor = self._build_mlp(human_obs_dim, self.net_arch["human_actor"], human_act_dim)
        
        # Exo actor network
        exo_obs_dim = self.network_index_handler.get_observation_num("exo_actor")
        exo_act_dim = self.network_index_handler.get_action_num("exo_actor")
        self.exo_actor = self._build_mlp(exo_obs_dim, self.net_arch["exo_actor"], exo_act_dim)

    def _build_critic_networks(self):
        """Build dual Q-networks (critics) using the common critic architecture."""
        critic_obs_dim = self.network_index_handler.get_observation_num("common_critic")
        action_dim = self.action_space.shape[0]  # Full action space dimension
        
        # Input dimension includes both observation and action
        q_net_input_dim = critic_obs_dim + action_dim
        
        # Build two Q-networks for SAC
        self.q1_net = self._build_mlp(q_net_input_dim, self.net_arch["common_critic"], 1)
        self.q2_net = self._build_mlp(q_net_input_dim, self.net_arch["common_critic"], 1)

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        """Forward pass for the actor (mu network)."""
        # Extract features if we have a features extractor
        if self.features_extractor is not None:
            processed_obs = self.features_extractor(obs)
        else:
            processed_obs = obs

        # Get observations for each network
        human_obs = self.network_index_handler.map_observation_to_network(processed_obs, "human_actor")
        exo_obs = self.network_index_handler.map_observation_to_network(processed_obs, "exo_actor")
        
        # Get actions from each network
        human_action = self.human_actor(human_obs)
        exo_action = self.exo_actor(exo_obs)
        
        # Combine actions using the network index handler
        network_output_dict = {"human_actor": human_action, "exo_actor": exo_action}
        return self.network_index_handler.map_network_to_action(network_output_dict)

    def forward_critics(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass for both critics (Q1 and Q2 networks)."""
        # Extract features if we have a features extractor
        if self.features_extractor is not None:
            processed_obs = self.features_extractor(obs)
        else:
            processed_obs = obs

        # Use processed observations for critic evaluation
        critic_obs = self.network_index_handler.map_observation_to_network(processed_obs, "common_critic")
        
        # Normalize actions before using them
        processed_actions = th.tanh(actions)  # Ensure actions are in [-1, 1]
        
        # Concatenate processed observations and actions
        q_input = th.cat([critic_obs, processed_actions], dim=1)
        
        # Get Q-values from both critics
        q1_values = self.q1_net(q_input)
        q2_values = self.q2_net(q_input)
        
        return q1_values, q2_values

    def reset_actor_networks(self):
        """Reset the actor networks while preserving architecture."""
        self._build_actor_networks()

    def reset_critic_networks(self):
        """Reset the critic networks while preserving architecture."""
        self._build_critic_networks()

class HumanExoSACPolicy(SACPolicy):
    """Custom SAC policy for human-exo control."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Any,
        **kwargs,
    ):
        # Extract custom policy params
        custom_policy_params_dict = kwargs.pop('custom_policy_params', None)
        self.custom_policy_params = DictionableDataclass.create(
            ExoImitationTrainSessionConfig.PolicyParams.CustomPolicyParams,
            custom_policy_params_dict
        )

        # Always use our custom features extractor
        kwargs['features_extractor_class'] = HumanExoFeaturesExtractor

        # Extract and update kwargs before passing to parent
        kwargs.setdefault('features_extractor_kwargs', {})
        kwargs.setdefault('optimizer_class', th.optim.Adam)
        kwargs.setdefault('optimizer_kwargs', dict(eps=1e-5))
        
        # Initialize parent with our custom policy parameters
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> HumanExoSACNetwork:
        """Create the custom actor-critic network."""
        # Create with proper feature extraction support
        return HumanExoSACNetwork(
            self.observation_space,
            self.action_space,
            self.custom_policy_params,
            features_extractor=features_extractor,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """SAC policy forward pass."""
        actions = self.actor.forward_actor(obs)
        # Always return the mean actions since our actor handles stochastic behavior
        return actions

    def reset_network(self, reset_shared_net: bool = False, reset_policy_net: bool = False, reset_value_net: bool = False):
        """Reset networks if specified."""
        if reset_policy_net:
            self.actor.reset_actor_networks()
        if reset_value_net:
            self.actor.reset_critic_networks()