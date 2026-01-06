import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
algos = ['SAC', 'PPO', 'A2C']

# File name for the trajectory data
file_relative_path = 'trajectory.csv' 

# CONTROL TIME LIMIT HERE
# Set to 10.0 to see only the first 10 seconds. 
# Set to None to see the whole file.
max_time = 10.0  

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

for algo in algos:
    # Construct file path (e.g., SAC/trajectory_data.csv)
    file_path = os.path.join(algo, file_relative_path)
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {algo} at {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip() # Clean column names
        
        # --- Filter data by time if max_time is set ---
        if max_time is not None:
            df = df[df['t'] <= max_time]

        # Plot X vs Time
        ax1.plot(df['t'], df['x'], linestyle='-', linewidth=2.5, label=algo)
        
        # Plot Y vs Time
        ax2.plot(df['t'], df['y'], linestyle='-', linewidth=2.5, label=algo)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Styling
ax1.set_title(f'X Position vs Time (First {max_time}s)' if max_time else 'X Position vs Time')
ax1.set_ylabel('X Position')
ax1.grid(True)
ax1.legend(loc='upper right')

ax2.set_title(f'Y Position vs Time (First {max_time}s)' if max_time else 'Y Position vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Y Position')
ax2.grid(True)
# ax2.legend() # Legend is already on top, usually sufficient, but can be added here too

plt.tight_layout()
plt.savefig('trajectory_plots.png')
plt.savefig('trajectory_plots.pdf')