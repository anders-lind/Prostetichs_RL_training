import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# --- SELECT MODEL TO PLOT ---
# CHANGE THIS to 'SAC', 'PPO', or 'A2C'
MODEL_TO_PLOT = 'A2C' 
# ----------------------------

# File names
file_relative_path = 'trajectory.csv' 
steps_file = 'steps.csv'

# CONTROL TIME LIMIT HERE
min_time = 2.0 # Set explicit min time for span calculation
max_time = 4.0  

# Define the color mapping for consistency
COLOR_MAP = {
    'SAC': '#1f77b4', # Blue
    'PPO': '#ff7f0e', # Orange
    'A2C': '#2ca02c'  # Green
}

# --- SHADING COLORS (Intervals between events) ---
COLOR_RED = 'red'
COLOR_CYAN = 'cyan'
SHADE_ALPHA = 0.1
# --------------------------------------------------

# --- VERTICAL LINE COLORS (Events) ---
COLOR_RHS = '#8B0000' # Dark Red 
COLOR_LHS = '#008B8B' # Dark Cyan 
LINE_ALPHA = 0.8
# -------------------------------------

# ---------------------------------------------------------
# 1. Load and Prepare Steps Data
# ---------------------------------------------------------
df_steps_data = []
try:
    df_steps = pd.read_csv(steps_file)
    df_steps.columns = df_steps.columns.str.strip() 
    
    if MODEL_TO_PLOT not in df_steps.columns:
        print(f"❌ ERROR: Column '{MODEL_TO_PLOT}' not found in {steps_file}. Skipping event plotting.")
    else:
        events = pd.to_numeric(df_steps[MODEL_TO_PLOT], errors='coerce').dropna()
        events = events[events >= min_time]
        events = events[events <= max_time]
        
        df_steps_data = sorted(events.tolist())
    
except Exception as e:
    print(f"❌ ERROR: Could not load or process {steps_file}: {e}. Skipping event plotting.")


# ---------------------------------------------------------
# 2. Plotting
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
plot_successful = False

# Construct file path for the selected model
file_path = os.path.join(MODEL_TO_PLOT, file_relative_path)
color = COLOR_MAP.get(MODEL_TO_PLOT, 'gray')

if not os.path.exists(file_path):
    print(f"Warning: File not found for {MODEL_TO_PLOT} at {file_path}")
else:
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip() 
        
        if 't' not in df.columns or 'x' not in df.columns or 'y' not in df.columns:
             print(f"Error: Missing 't', 'x', or 'y' columns in {file_path}. Skipping plot.")
        else:
            # --- Filter data by time ---
            df = df[df['t'] >= min_time]
            if max_time is not None:
                df = df[df['t'] <= max_time]

            # Plot X vs Time
            ax1.plot(df['t'], df['x'], linestyle='-', linewidth=2.5, label=MODEL_TO_PLOT, color=color)
            
            # Plot Y vs Time
            ax2.plot(df['t'], df['y'], linestyle='-', linewidth=2.5, label=MODEL_TO_PLOT, color=color)
            
            plot_successful = True
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# --- 3. Plot Alternating Colored Areas (Shading) ---
boundaries = [min_time] + df_steps_data
if boundaries[-1] < max_time:
    boundaries.append(max_time)
elif boundaries[-1] > max_time:
    boundaries[-1] = max_time
    
if plot_successful and len(boundaries) > 1:
    for k in range(1, len(boundaries) - 1): 
        t_start = boundaries[k]
        t_end = boundaries[k+1]
        
        color_index = k - 1
        
        if color_index % 2 == 0:
            fill_color = COLOR_RED # Red shade for the interval [E1, E2], which is now 'Left Foot Swing'
        else:
            fill_color = COLOR_CYAN # Cyan shade for the interval [E2, E3], which is now 'Right Foot Swing'
            
        for ax in [ax1, ax2]:
             ax.axvspan(t_start, t_end, color=fill_color, alpha=SHADE_ALPHA, zorder=0)


# --- 4. Plot Vertical Event Lines (Swapped Colors) ---
if plot_successful and df_steps_data:
    for index, event_time in enumerate(df_steps_data):
        
        # --- SWAP IMPLEMENTATION ---
        # RHS event (index 0, 2, ...) now uses Dark Cyan line (COLOR_LHS)
        # LHS event (index 1, 3, ...) now uses Dark Red line (COLOR_RHS)
        if index % 2 == 0:
            line_color = COLOR_LHS  
        else:
            line_color = COLOR_RHS  

        for ax in [ax1, ax2]:
            ax.axvline(x=event_time, color=line_color, linestyle=':', linewidth=2, alpha=LINE_ALPHA)


# --- 5. Add Dummy Plots for Shading and Line Legend (Swapped Labels) ---
if plot_successful:
    # --- SWAPPED SHADING LEGEND LABELS ---
    # 1. Red Shade (Currently in plot for Right Foot Hit Strike -> Left Foot Swing)
    ax1.plot([], [], 
             color=COLOR_RED, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Left Foot Swing') 
    
    # 2. Cyan Shade (Currently in plot for Left Foot Hit Strike -> Right Foot Swing)
    ax1.plot([], [], 
             color=COLOR_CYAN, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Right Foot Swing')
    
    # --- VERTICAL LINE LEGENDS (Labels stay, Colors match section 4) ---
    # 3. Right Foot Hit Ground (RHS event, uses Dark Cyan line: COLOR_LHS)
    ax1.plot([], [], 
             color=COLOR_LHS, 
             linestyle=':', 
             linewidth=2, 
             alpha=LINE_ALPHA, 
             label='Right Foot Hit Ground')
    
    # 4. Left Foot Hit Ground (LHS event, uses Dark Red line: COLOR_RHS)
    ax1.plot([], [], 
             color=COLOR_RHS, 
             linestyle=':', 
             linewidth=2, 
             alpha=LINE_ALPHA, 
             label='Left Foot Hit Ground')


# --- 6. Styling ---
time_limit_str = f' (From {min_time}s to {max_time}s)'

ax1.set_title(f'{MODEL_TO_PLOT} X Position vs Time{time_limit_str}')
ax1.set_ylabel('X Position')
ax1.grid(True)
ax1.legend(loc='upper right')

ax2.set_title(f'{MODEL_TO_PLOT} Y Position vs Time{time_limit_str}')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Y Position')
ax2.grid(True)

plt.tight_layout()

# Save the final figure
output_name = f'trajectory_gait_analysis_{MODEL_TO_PLOT}'
plt.savefig(f'{output_name}.png')
plt.savefig(f'{output_name}.pdf')