import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the algorithms and their corresponding file paths
data_sources = {
    'SAC': 'SAC/train_log_plots/train_log_data.csv',
    'PPO': 'PPO/train_log_plots/train_log_data.csv',
    'A2C': 'A2C/train_log_plots/train_log_data.csv'
}

plt.figure(figsize=(8.0, 4.0))

for algo, file_path in data_sources.items():
    # Check if file exists to avoid crashing
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {algo} at {file_path}")
        continue

    # Load the data
    df = pd.read_csv(file_path)

    # Filter out rows with NaN values in the columns of interest
    df_clean = df.dropna(subset=['num_timesteps', 'average_reward_per_episode'])

    # Filter out rows where average_reward_per_episode is 0
    df_clean = df_clean[df_clean['average_reward_per_episode'] != 0]

    # Apply specific scaling for SAC
    if algo == 'SAC':
        df_clean['average_reward_per_episode'] = df_clean['average_reward_per_episode'] / 64.0

    # Plot the data
    # Removed marker='o' and markersize
    # Added linewidth=2.5 for a solid, constant thickness
    plt.plot(df_clean['num_timesteps'], df_clean['average_reward_per_episode'], 
             linestyle='-', linewidth=2.5, label=algo)

# Add labels, title, and legend
plt.title('Average Reward per Episode vs Num Timesteps (SAC, PPO, A2C)')
plt.xlabel('Num Timesteps')
plt.ylabel('Average Reward per Episode')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig('average_reward.png')
plt.savefig('average_reward.pdf')