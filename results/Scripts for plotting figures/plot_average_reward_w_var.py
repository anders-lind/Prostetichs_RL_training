import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Define the algorithms and their corresponding file paths
data_sources = {
    'SAC': 'SAC/train_log_plots/train_log_data.csv',
    'PPO': 'PPO/train_log_plots/train_log_data.csv',
    'A2C': 'A2C/train_log_plots/train_log_data.csv'
}

# Window size for the rolling standard deviation
# Increase this (e.g., to 20) for a smoother, wider shadow.
# Decrease this (e.g., to 5) for a sharper, more jagged shadow.
ROLLING_WINDOW = 20

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
plt.figure(figsize=(8, 4))

for algo, file_path in data_sources.items():
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {algo} at {file_path}")
        continue

    # Load the data
    df = pd.read_csv(file_path)

    # 1. Filter out NaNs
    df_clean = df.dropna(subset=['num_timesteps', 'average_reward_per_episode'])
    
    # 2. Filter out 0 entries (as per your request)
    df_clean = df_clean[df_clean['average_reward_per_episode'] != 0]

    # 3. Apply specific scaling for SAC
    if algo == 'SAC':
        df_clean['average_reward_per_episode'] = df_clean['average_reward_per_episode'] / 100.0

    # Sort by timesteps to ensure rolling calculation is correct
    df_clean = df_clean.sort_values('num_timesteps')

    # 4. Calculate Rolling Statistics
    # We use the rolling standard deviation of the *reward* itself to show stability
    reward_mean = df_clean['average_reward_per_episode']
    reward_std_shadow = df_clean['average_reward_per_episode'].rolling(window=ROLLING_WINDOW).std()
    
    # Backfill the first few NaNs so the shadow starts at t=0
    reward_std_shadow = reward_std_shadow.bfill()

    # 5. Plot the Mean Line
    line, = plt.plot(df_clean['num_timesteps'], reward_mean, 
                     linestyle='-', linewidth=2.5, label=algo)
    
    # Get the color of the line so the shadow matches
    color = line.get_color()

    # 6. Plot the Shadow (Mean +/- Rolling Std)
    plt.fill_between(df_clean['num_timesteps'], 
                     reward_mean - reward_std_shadow,
                     reward_mean + reward_std_shadow,
                     color=color, alpha=0.15, linewidth=0)

# Add labels, title, and legend
plt.title('Average Reward per Episode vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Average Reward per Episode')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('average_reward_w_var.png') 
plt.savefig('average_reward_w_var.pdf')