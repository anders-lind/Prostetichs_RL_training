import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# List of directories/algorithms to load data from
algos = ['SAC', 'PPO', 'A2C']

# File name for the trajectory data within each directory
file_relative_path = 'trajectory.csv' 

# CONTROL TIME WINDOW HERE
min_time = 2.0  
max_time = 10.0  

# --- TREND LINE CONFIGURATION (RUNNING AVERAGE) ---
# TOGGLE: Set to True to plot trend lines on the velocity graphs, False to hide them.
plot_trend_lines = False 
# WINDOW SIZE: Number of data points to include in the moving average calculation.
# Larger values result in smoother, but less responsive, trends.
trend_window_size = 50

# ---------------------------------------------------------
# Function to calculate derivatives
# ---------------------------------------------------------
def calculate_kinematics(df, axis_name):
    """Calculates Speed, Acceleration, and Jerk for a given axis."""
    
    # Calculate time difference (dt) between consecutive points
    dt = df['t'].diff()
    
    # --- 1. Speed (First Derivative of Position) ---
    dx = df[axis_name].diff()
    df[f'v_{axis_name}'] = dx / dt
    
    # --- 2. Acceleration (Second Derivative of Position) ---
    dv = df[f'v_{axis_name}'].diff()
    df[f'a_{axis_name}'] = dv / dt
    
    # --- 3. Jerk (Third Derivative of Position) ---
    da = df[f'a_{axis_name}'].diff()
    df[f'j_{axis_name}'] = da / dt
    
    return df

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
# Create 4 subplots (Position, Speed, Acceleration, Jerk) for X and Y
fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
plt.subplots_adjust(hspace=0.4) 

# Define the plot properties: (row_index, col_index, y_data_column, title, y_label)
plot_specs = [
    (0, 0, 'x', 'X Position', 'X Position'),
    (0, 1, 'y', 'Y Position', 'Y Position'),
    (1, 0, 'v_x', 'X Speed (Velocity)', 'X Velocity'), # Velocity plots are at row index 1
    (1, 1, 'v_y', 'Y Speed (Velocity)', 'Y Velocity'), # Velocity plots are at row index 1
    (2, 0, 'a_x', 'X Acceleration', 'X Acceleration'),
    (2, 1, 'a_y', 'Y Acceleration', 'Y Acceleration'),
    (3, 0, 'j_x', 'X Jerk', 'X Jerk'),
    (3, 1, 'j_y', 'Y Jerk', 'Y Jerk'),
]

for algo in algos:
    file_path = os.path.join(algo, file_relative_path)
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found for {algo} at {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip() 
        
        # --- Filter data by time (From and To) ---
        if min_time is not None:
            df = df[df['t'] >= min_time]
            
        if max_time is not None:
            df = df[df['t'] <= max_time]

        if df.empty:
            print(f"Skipping {algo}: No data available in the time range [{min_time}s, {max_time}s].")
            continue

        # --- Calculate Derivatives (Speed, Acceleration, Jerk) ---
        df = calculate_kinematics(df.copy(), 'x')
        df = calculate_kinematics(df.copy(), 'y')
        
        # Prepare clean data for plotting and fitting (removes NaNs from derivatives)
        df_clean = df.dropna(subset=['v_x', 'v_y', 'a_x', 'a_y', 'j_x', 'j_y'])
        
        # --- Plotting Loop ---
        for row, col, data_col, _, y_label in plot_specs:
            ax = axes[row, col]
            
            # Plot the raw data 
            plot_df = df if row == 0 else df_clean
            
            ax.plot(plot_df['t'], plot_df[data_col], linestyle='-', linewidth=1.5, alpha=0.6, label=algo)
            
            # --- RUNNING AVERAGE TREND LINE LOGIC ---
            if plot_trend_lines and row == 1:
                # This block only executes if the toggle is True AND it's a velocity plot
                line_color = ax.lines[-1].get_color() 
                
                # Check if there are enough points for the rolling average
                if len(df_clean) >= trend_window_size:
                    
                    # Calculate the running average
                    # 'center=True' centers the window around the point
                    # 'min_periods=1' ensures the line starts immediately
                    running_avg = df_clean[data_col].rolling(
                        window=trend_window_size, 
                        center=True, 
                        min_periods=1
                    ).mean()
                    
                    # Plot the trend line
                    ax.plot(df_clean['t'], running_avg, 
                            linestyle='--', linewidth=3, 
                            color=line_color, 
                            label=f'{algo} Avg (Window {trend_window_size})')
                else:
                     print(f"Note: Not enough data points ({len(df_clean)}) to calculate {algo} running average with window size {trend_window_size}.")

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

# --- Styling and Final Touches ---
time_limit_str = f' (From {min_time}s to {max_time}s)' if (min_time is not None and max_time is not None) else ''

for row, col, data_col, title, y_label in plot_specs:
    ax = axes[row, col]
    
    ax.set_title(f'{title} vs Time{time_limit_str}', fontsize=14)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend control: Add combined legend to the top-left plot
    if row == 0 and col == 0:
        # Collect all handles and labels across all subplots
        all_handles = []
        all_labels = []
        for r in range(4):
             for c in range(2):
                 h, l = axes[r, c].get_legend_handles_labels()
                 all_handles.extend(h)
                 all_labels.extend(l)
        
        # Keep only unique labels
        unique_labels = {}
        for h, l in zip(all_handles, all_labels):
            # Key should be the algorithm name
            key = l.split(' Avg')[0] 
            # Prioritize the running average label if present
            if key not in unique_labels or 'Avg' in l:
                 unique_labels[key] = (h, l)
        
        # Place legend on the position plot
        ax.legend([v[0] for v in unique_labels.values()], 
                  [v[1] for v in unique_labels.values()], 
                  loc='best', fontsize=10)


# Only add X-axis label to the bottom plots
axes[3, 0].set_xlabel('Time (s)', fontsize=12)
axes[3, 1].set_xlabel('Time (s)', fontsize=12)

plt.tight_layout()

# Save the final figure
output_name = f'trajectory_extended{"_with_trend_lines" if plot_trend_lines else ""}'
plt.savefig(f'{output_name}.png')
plt.savefig(f'{output_name}.pdf')
print(f"\n✅ Plots generated and saved to '{output_name}.png' and '{output_name}.pdf'")