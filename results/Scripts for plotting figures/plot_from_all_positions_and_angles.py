import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Define the file name that contains the position/angle data
file_relative_path = 'all_positions_and_angles.csv'

# Define the full data sources dictionary
data_sources = {
    'SAC': os.path.join('SAC', file_relative_path),
    'PPO': os.path.join('PPO', file_relative_path),
    'A2C': os.path.join('A2C', file_relative_path)
}
algos = list(data_sources.keys()) 

# --- CRITICAL: Define the known logging frequency ---
LOGGING_FREQUENCY_HZ = 30.0 # 30 times per second

# CONTROL TIME WINDOW HERE (now based on calculated time in seconds)
min_time = 2.0  
max_time = 10.0  

# --- COLUMNS TO PLOT (Update this list with the columns you want to see) ---
PLOT_COLUMNS = [
    'joint_data_pelvis_tx',     
    'joint_data_hip_flexion_r', 
    'joint_data_knee_angle_r', 
    'joint_data_ankle_angle_r'
]
# ----------------------------------------------------------------------------

# Define the color mapping for consistency
COLOR_MAP = {
    'SAC': '#1f77b4', # Blue
    'PPO': '#ff7f0e', # Orange
    'A2C': '#2ca02c'  # Green
}

# ---------------------------------------------------------
# Data Processing and Plotting
# ---------------------------------------------------------

num_plots = len(PLOT_COLUMNS)
if num_plots == 0:
    print("❌ ERROR: PLOT_COLUMNS list is empty. Please specify which columns to plot.")
    exit()

fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
if num_plots == 1:
    axes = np.array([axes])
plt.subplots_adjust(hspace=0.2) 

for algo, file_path in data_sources.items():
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found for {algo} at {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip() 
        
        missing_cols = [col for col in PLOT_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"❌ Error for {algo}: Missing data column(s) {missing_cols}. Skipping.")
            continue
        
        # --- CALCULATE TIME AXIS ---
        df['row_index'] = df.index
        df['time_s'] = df['row_index'] / LOGGING_FREQUENCY_HZ
        time_col_name = 'time_s' 
        
        # --- Filter data by calculated time ---
        if min_time is not None:
            df = df[df[time_col_name] >= min_time]
            
        if max_time is not None:
            df = df[df[time_col_name] <= max_time]

        if df.empty:
            print(f"Skipping {algo}: No data available in the time range.")
            continue

        # --- Plotting Loop ---
        color = COLOR_MAP.get(algo, 'gray')
        
        for i, data_col in enumerate(PLOT_COLUMNS):
            ax = axes[i]
            
            ax.plot(df[time_col_name], df[data_col], 
                    linestyle='-', linewidth=1.5, label=algo, color=color)
            
    except Exception as e:
        print(f"❌ Unhandled Error reading {file_path} for {algo}: {e}")

# --- Styling and Final Touches ---
time_limit_str = f' (From {min_time}s to {max_time}s)'

for i, data_col in enumerate(PLOT_COLUMNS):
    ax = axes[i]
    
    if i == 0:
        ax.set_title(f'Kinematic and Angle Data Comparison{time_limit_str}', fontsize=16)

    ax.set_ylabel(data_col.replace('_', ' ').title(), fontsize=12) 
    
    # --- Grid Density (Doubled) ---
    # X-axis ticks (10 -> 20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20)) 
    
    # Y-axis ticks (6 -> 12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12)) 
    
    # Draw grid using only major ticks and very low opacity
    ax.grid(True, which='major', linestyle='--', alpha=0.2, color='gray') 
    
    if i == num_plots - 1:
        ax.set_xlabel('Time (s)', fontsize=14)
    else:
        ax.tick_params(labelbottom=False)
    
    if i == 0:
        ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()

# --- FILENAME: Use the simple, requested name ---
output_name = 'positions_and_angles'
# ---------------------------------------------

# Save the final figure
plt.savefig(f'{output_name}.png')
plt.savefig(f'{output_name}.pdf')
print(f"\n✅ Plots generated and saved to '{output_name}.png' and .pdf")