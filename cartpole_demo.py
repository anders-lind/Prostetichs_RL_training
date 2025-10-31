"""
Simple script to run CartPole training with different configurations and compare results.
This demonstrates how well the RL model learns the CartPole task.
"""

import os
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append('/home/anders/workspace/project_in_AI/myoassist')

from cartpole_training_with_plots import main


def quick_training_demo():
    """Run a quick CartPole training demonstration"""
    print("🎮 CartPole RL Training Demo")
    print("=" * 50)
    
    # Run training with different configurations
    configs = [
        {"timesteps": 3000, "name": "Quick Training (3k steps)"},
        {"timesteps": 8000, "name": "Extended Training (8k steps)"}
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n🚀 Running {config['name']}...")
        print(f"Training for {config['timesteps']} timesteps")
        
        # Run training
        main(total_timesteps=config['timesteps'], plot_freq=500)
        
        # Find the latest results directory
        result_dirs = glob.glob("./cartpole_results_*")
        if result_dirs:
            latest_dir = max(result_dirs, key=os.path.getctime)
            results.append({
                'name': config['name'],
                'dir': latest_dir,
                'timesteps': config['timesteps']
            })
            print(f"✅ Results saved in: {latest_dir}")
    
    # Display summary
    print("\n" + "=" * 50)
    print("🏆 TRAINING SUMMARY")
    print("=" * 50)
    
    for result in results:
        print(f"\n📊 {result['name']}:")
        
        # Try to read training data
        data_file = os.path.join(result['dir'], 'plots', 'training_data.csv')
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            if not df.empty:
                avg_reward = df['episode_rewards'].mean()
                max_reward = df['episode_rewards'].max()
                final_rewards = df['episode_rewards'].tail(10).mean()
                
                print(f"   • Average reward: {avg_reward:.1f}")
                print(f"   • Maximum reward: {max_reward:.1f}")
                print(f"   • Final 10 episodes avg: {final_rewards:.1f}")
                print(f"   • Total episodes: {len(df)}")
        
        print(f"   • Results directory: {result['dir']}")
        print(f"   • Plots available in: {os.path.join(result['dir'], 'plots')}")


def view_latest_results():
    """View the results from the most recent training run"""
    result_dirs = glob.glob("./cartpole_results_*")
    if not result_dirs:
        print("❌ No training results found. Run training first!")
        return
    
    latest_dir = max(result_dirs, key=os.path.getctime)
    plots_dir = os.path.join(latest_dir, 'plots')
    
    print(f"📊 Viewing results from: {latest_dir}")
    print(f"🖼️  Plots directory: {plots_dir}")
    
    # Check available plots
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        print(f"📈 Available plots ({len(plot_files)}):")
        for plot in sorted(plot_files):
            print(f"   • {plot}")
        
        # Read and summarize training data
        data_file = os.path.join(plots_dir, 'training_data.csv')
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            print(f"\n📈 Training Summary:")
            print(f"   • Total episodes: {len(df)}")
            print(f"   • Average episode reward: {df['episode_rewards'].mean():.1f}")
            print(f"   • Maximum episode reward: {df['episode_rewards'].max():.1f}")
            print(f"   • Average episode length: {df['episode_lengths'].mean():.1f}")
            print(f"   • Maximum episode length: {df['episode_lengths'].max():.1f}")
            
            # Show learning progress
            first_10 = df['episode_rewards'].head(10).mean()
            last_10 = df['episode_rewards'].tail(10).mean()
            improvement = ((last_10 - first_10) / first_10) * 100 if first_10 > 0 else 0
            
            print(f"\n🎯 Learning Progress:")
            print(f"   • First 10 episodes avg: {first_10:.1f}")
            print(f"   • Last 10 episodes avg: {last_10:.1f}")
            print(f"   • Improvement: {improvement:+.1f}%")
            
            # Success rate (episodes reaching near-maximum performance)
            success_threshold = 450  # Consider 450+ steps as success
            success_rate = (df['episode_lengths'] >= success_threshold).mean() * 100
            print(f"   • Success rate (450+ steps): {success_rate:.1f}%")
    
    else:
        print("❌ No plots directory found!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CartPole RL Demo and Results Viewer")
    parser.add_argument("--demo", action="store_true", help="Run training demo")
    parser.add_argument("--view", action="store_true", help="View latest results")
    parser.add_argument("--quick", action="store_true", help="Quick single training run")
    
    args = parser.parse_args()
    
    if args.demo:
        quick_training_demo()
    elif args.view:
        view_latest_results()
    elif args.quick:
        print("🚀 Quick CartPole Training...")
        main(total_timesteps=5000, plot_freq=500)
        view_latest_results()
    else:
        print("CartPole RL Training Options:")
        print("  --demo   : Run training demo with multiple configurations")
        print("  --view   : View results from the latest training run")
        print("  --quick  : Quick training run (5000 timesteps)")
        print("\nOr run directly: python cartpole_training_with_plots.py")