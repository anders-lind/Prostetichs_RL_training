import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class HumanExoFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extraction for human and exoskeleton control."""
    
    def __init__(
        self, 
        observation_space: spaces.Space, 
        features_dim: int = 64
    ):
        super().__init__(observation_space, features_dim)

        n_input = np.prod(observation_space.shape)

        # Simple feature extraction: just flatten the observation space
        # We could add more complex feature extraction here if needed
        self.flatten = th.nn.Flatten()
        self.linear = th.nn.Sequential(
            th.nn.Linear(n_input, features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Extract features from observations."""
        flat_obs = self.flatten(observations)
        return self.linear(flat_obs)