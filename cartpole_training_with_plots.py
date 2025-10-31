"""
CartPole RL Training with Real-time Plotting and Analysis

This script trains an RL agent on CartPole-v1 environment and creates comprehensive plots
to visualize the training progress, including episode rewards, episode lengths, and learning curves.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Add the project root to the Python path
sys.path.append('/home/anders/workspace/project_in_AI/myoassist')

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import gymnasium as gym

# Import your existing framework
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass


class PlottingCallback(BaseCallback):
    """
    Custom callback for collecting training data and creating plots
    """
    def __init__(self, plot_freq: int = 100, save_dir: str = "./plots", verbose: int = 0):
        super(PlottingCallback, self).__init__(verbose)
        self.plot_freq = plot_freq
        self.save_dir = save_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.learning_rates = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        # Collect episode data when episode ends
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer is not None and len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
                    self.timesteps.append(self.num_timesteps)
        
        # Collect training metrics
        if hasattr(self.logger, 'name_to_value') and self.logger.name_to_value is not None:
            if 'train/policy_loss' in self.logger.name_to_value:
                self.policy_losses.append(self.logger.name_to_value['train/policy_loss'])
            if 'train/value_loss' in self.logger.name_to_value:
                self.value_losses.append(self.logger.name_to_value['train/value_loss'])
            if 'train/entropy_loss' in self.logger.name_to_value:
                self.entropy_losses.append(self.logger.name_to_value['train/entropy_loss'])
            if 'train/learning_rate' in self.logger.name_to_value:
                self.learning_rates.append(self.logger.name_to_value['train/learning_rate'])
        
        # Create plots periodically
        if self.num_timesteps % self.plot_freq == 0 and len(self.episode_rewards) > 10:
            self._create_plots()
            
        return True
    
    def _create_plots(self):
        """Create and save training progress plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'CartPole RL Training Progress (Timestep: {self.num_timesteps})', fontsize=16)
        
        # Plot 1: Episode Rewards
        if len(self.episode_rewards) > 0:
            axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
            # Add moving average
            if len(self.episode_rewards) > 10:
                window = min(50, len(self.episode_rewards) // 4)
                moving_avg = pd.Series(self.episode_rewards).rolling(window=window).mean()
                axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
            axes[0, 0].set_title('Episode Rewards Over Time')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        if len(self.episode_lengths) > 0:
            axes[0, 1].plot(self.episode_lengths, alpha=0.6, color='green', label='Episode Length')
            # Add moving average
            if len(self.episode_lengths) > 10:
                window = min(50, len(self.episode_lengths) // 4)
                moving_avg = pd.Series(self.episode_lengths).rolling(window=window).mean()
                axes[0, 1].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
            axes[0, 1].set_title('Episode Lengths Over Time')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Episode Length (Steps)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add horizontal line at CartPole max (500 steps)
            axes[0, 1].axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Max Possible (500)')
            axes[0, 1].legend()
        
        # Plot 3: Loss Values
        if len(self.policy_losses) > 0:
            x_vals = range(len(self.policy_losses))
            if len(self.policy_losses) > 0:
                axes[1, 0].plot(x_vals, self.policy_losses, label='Policy Loss', alpha=0.7)
            if len(self.value_losses) > 0:
                axes[1, 0].plot(x_vals, self.value_losses, label='Value Loss', alpha=0.7)
            if len(self.entropy_losses) > 0:
                # Scale entropy loss for better visualization
                scaled_entropy = [abs(x) * 100 for x in self.entropy_losses]
                axes[1, 0].plot(x_vals, scaled_entropy, label='Entropy Loss (×100)', alpha=0.7)
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Training Iteration')
            axes[1, 0].set_ylabel('Loss Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')  # Log scale for better visibility
        
        # Plot 4: Learning Rate
        if len(self.learning_rates) > 0:
            axes[1, 1].plot(self.learning_rates, color='purple', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Over Time')
            axes[1, 1].set_xlabel('Training Iteration')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If no learning rate data, show recent performance summary
            if len(self.episode_rewards) > 10:
                recent_rewards = self.episode_rewards[-20:]
                recent_lengths = self.episode_lengths[-20:]
                
                axes[1, 1].bar(['Avg Reward', 'Avg Length'], 
                              [np.mean(recent_rewards), np.mean(recent_lengths)],
                              color=['blue', 'green'], alpha=0.7)
                axes[1, 1].set_title('Recent Performance (Last 20 Episodes)')
                axes[1, 1].set_ylabel('Value')
                
                # Add text annotations
                axes[1, 1].text(0, np.mean(recent_rewards) + 5, f'{np.mean(recent_rewards):.1f}', 
                               ha='center', va='bottom')
                axes[1, 1].text(1, np.mean(recent_lengths) + 10, f'{np.mean(recent_lengths):.1f}', 
                               ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(self.save_dir, f'training_progress_{self.num_timesteps:06d}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        # Also save as latest
        latest_filename = os.path.join(self.save_dir, 'latest_training_progress.png')
        plt.savefig(latest_filename, dpi=300, bbox_inches='tight')
        
        plt.close()  # Close the figure to free memory
        
        if self.verbose > 0:
            print(f"Plot saved to {plot_filename}")
    
    def save_training_data(self):
        """Save training data to CSV for further analysis"""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'timesteps': self.timesteps[:len(self.episode_rewards)]  # Align lengths
        }
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        csv_filename = os.path.join(self.save_dir, 'training_data.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"Training data saved to {csv_filename}")


def create_cartpole_config(total_timesteps: int = 20000) -> Dict[str, Any]:
    """Create optimized CartPole training configuration"""
    return {
        "comment": "CartPole RL training with plotting",
        "total_timesteps": total_timesteps,
        "evaluate_param_list": [
            {
                "num_timesteps": 1000
            }
        ],
        "logger_params": {
            "logging_frequency": 50
        },
        "env_params": {
            "env_id": "CartPole-v1",
            "seed": 42,
            "num_envs": 1
        },
        "ppo_params": {
            "learning_rate": 0.001,
            "n_steps": 512,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": 100,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": 0.01,
            "device": "cpu"
        },
        "policy_params": {
            "comment": "Using standard MlpPolicy for discrete CartPole environment"
        }
    }


def main(total_timesteps: int = 20000, plot_freq: int = 1000):
    """
    Main training function with plotting
    
    Args:
        total_timesteps: Total number of training timesteps
        plot_freq: Frequency of plot updates (in timesteps)
    """
    print("🚀 Starting CartPole RL Training with Plotting")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Plot update frequency: {plot_freq}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./cartpole_results_{timestamp}"
    plots_dir = os.path.join(results_dir, "plots")
    models_dir = os.path.join(results_dir, "models")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Create configuration
    config_dict = create_cartpole_config(total_timesteps)
    config_dict["total_timesteps"] = total_timesteps
    
    # Save configuration
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create config object
    config = DictionableDataclass.create(TrainSessionConfigBase, config_dict)
    
    # Create environment
    print("🏗️  Creating CartPole environment...")
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=False)
    
    # Create model
    print("🧠 Creating RL model...")
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)
    
    # Set up logging
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
    
    # Create plotting callback
    plotting_callback = PlottingCallback(
        plot_freq=plot_freq,
        save_dir=plots_dir,
        verbose=1
    )
    
    # Train the model
    print("🎯 Starting training...")
    print(f"Check progress plots in: {plots_dir}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=plotting_callback,
            progress_bar=True
        )
        
        print("✅ Training completed successfully!")
        
        # Save the final model
        model_path = os.path.join(models_dir, "final_cartpole_model")
        model.save(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save training data
        plotting_callback.save_training_data()
        
        # Create final comprehensive plot
        plotting_callback._create_plots()
        
        # Evaluate the trained model
        print("🧪 Evaluating trained model...")
        evaluate_model(model, env, plots_dir, n_episodes=10)
        
        print(f"\n🎉 Training complete! Results saved in: {results_dir}")
        print(f"📊 Check plots in: {plots_dir}")
        print(f"🤖 Model saved in: {models_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        plotting_callback.save_training_data()
        plotting_callback._create_plots()
        
        # Save intermediate model
        model_path = os.path.join(models_dir, "interrupted_cartpole_model")
        model.save(model_path)
        print(f"💾 Intermediate model saved to: {model_path}")
    
    finally:
        env.close()


def evaluate_model(model, env, save_dir: str, n_episodes: int = 10):
    """Evaluate the trained model and create evaluation plots"""
    print(f"Running evaluation for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}")
    
    # Create evaluation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Episode rewards
    ax1.bar(range(1, n_episodes + 1), episode_rewards, alpha=0.7, color='blue')
    ax1.set_title('Evaluation Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.axhline(y=np.mean(episode_rewards), color='red', linestyle='--', 
                label=f'Average: {np.mean(episode_rewards):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths
    ax2.bar(range(1, n_episodes + 1), episode_lengths, alpha=0.7, color='green')
    ax2.set_title('Evaluation Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length (Steps)')
    ax2.axhline(y=np.mean(episode_lengths), color='red', linestyle='--', 
                label=f'Average: {np.mean(episode_lengths):.1f}')
    ax2.axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='Max Possible (500)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    eval_plot_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Evaluation plot saved to: {eval_plot_path}")
    print(f"📈 Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"📏 Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CartPole RL Training with Plotting")
    parser.add_argument("--timesteps", type=int, default=20000, 
                       help="Total training timesteps (default: 20000)")
    parser.add_argument("--plot-freq", type=int, default=1000, 
                       help="Plot update frequency in timesteps (default: 1000)")
    
    args = parser.parse_args()
    
    main(total_timesteps=args.timesteps, plot_freq=args.plot_freq)