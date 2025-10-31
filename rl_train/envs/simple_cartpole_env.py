"""
Simple CartPole environment wrapper for testing RL models.
This provides a clean interface that works with the existing training framework.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple


class SimpleCartPoleEnv:
    """Simple wrapper for CartPole-v1 environment that integrates with the existing framework."""
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the CartPole environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        else:
            # For newer gymnasium versions
            self.env.reset(seed=seed)
            return [seed]
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env.unwrapped


def create_cartpole_env(render_mode: Optional[str] = None) -> SimpleCartPoleEnv:
    """
    Factory function to create a CartPole environment.
    
    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        
    Returns:
        SimpleCartPoleEnv: Configured CartPole environment
    """
    return SimpleCartPoleEnv(render_mode=render_mode)