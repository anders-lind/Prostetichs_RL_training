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
MODEL_TO_PLOT = 'PPO' 
# ----------------------------

# Define the file name that contains the position/angle data
file_relative_path = 'all_positions_and_angles.csv'
steps_file = 'steps.csv'

# Define the full data sources dictionary
data_sources = {
    'SAC': os.path.join('SAC', file_relative_path),
    'PPO': os.path.join('PPO', file_relative_path),
    'A2C': os.path.join('A2C', file_relative_path)
}

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

# Define the color mapping for consistency (for the line plot)
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

# --- VERTICAL LINE COLORS (Events - SWAPPED) ---
# RHS event line (index 0, 2, ...) now uses Dark Cyan line
# LHS event line (index 1, 3, ...) now uses Dark Red line
COLOR_RHS_LINE = '#008B8B' # Dark Cyan 
COLOR_LHS_LINE = '#8B0000' # Dark Red 
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
# 2. Plotting (Single Model)
# ---------------------------------------------------------

num_plots = len(PLOT_COLUMNS)
if num_plots == 0:
    print("❌ ERROR: PLOT_COLUMNS list is empty. Please specify which columns to plot.")
    exit()
if MODEL_TO_PLOT not in data_sources:
    print(f"❌ ERROR: Invalid model selected: {MODEL_TO_PLOT}. Must be SAC, PPO, or A2C.")
    exit()


fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
if num_plots == 1:
    axes = np.array([axes])
plt.subplots_adjust(hspace=0.2) 

# Process only the selected model
algo = MODEL_TO_PLOT
file_path = data_sources[algo]
color = COLOR_MAP.get(algo, 'gray')
plot_successful = False

if not os.path.exists(file_path):
    print(f"⚠️ Warning: File not found for {algo} at {file_path}")
else:
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip() 
        
        missing_cols = [col for col in PLOT_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"❌ Error for {algo}: Missing data column(s) {missing_cols}. Skipping.")
        else:
            # --- CALCULATE TIME AXIS ---
            df['row_index'] = df.index
            df['time_s'] = df['row_index'] / LOGGING_FREQUENCY_HZ
            time_col_name = 'time_s' 
            
            # --- Filter data by calculated time ---
            df = df[df[time_col_name] >= min_time]
            df = df[df[time_col_name] <= max_time]

            if df.empty:
                print(f"Skipping {algo}: No data available in the time range.")
            else:
                # --- Plotting Loop ---
                for i, data_col in enumerate(PLOT_COLUMNS):
                    ax = axes[i]
                    
                    ax.plot(df[time_col_name], df[data_col], 
                            linestyle='-', linewidth=1.5, label=algo, color=color)
                
                plot_successful = True
            
    except Exception as e:
        print(f"❌ Unhandled Error reading {file_path} for {algo}: {e}")


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
            fill_color = COLOR_RED # Red shade for interval starting after RHS event -> Left Foot Swing
        else:
            fill_color = COLOR_CYAN # Cyan shade for interval starting after LHS event -> Right Foot Swing
            
        for ax in axes.flatten():
             ax.axvspan(t_start, t_end, color=fill_color, alpha=SHADE_ALPHA, zorder=0)


# --- 4. Plot Vertical Event Lines (Swapped Colors) ---
if plot_successful and df_steps_data:
    for index, event_time in enumerate(df_steps_data):
        
        # --- SWAP IMPLEMENTATION ---
        # RHS event (index 0, 2, ...) now uses Dark Cyan line (COLOR_RHS_LINE)
        # LHS event (index 1, 3, ...) now uses Dark Red line (COLOR_LHS_LINE)
        if index % 2 == 0:
            line_color = COLOR_RHS_LINE  
        else:
            line_color = COLOR_LHS_LINE  

        for ax in axes.flatten():
            ax.axvline(x=event_time, color=line_color, linestyle=':', linewidth=2, alpha=LINE_ALPHA, zorder=1)


# --- 5. Add Dummy Plots for Shading and Line Legend (Swapped Labels) ---
if plot_successful:
    ax = axes[0] # Use the top plot for the combined legend
    
    # --- SWAPPED SHADING LEGEND LABELS ---
    # 1. Red Shade (Left Foot Swing)
    ax.plot([], [], 
             color=COLOR_RED, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Left Foot Swing') 
    
    # 2. Cyan Shade (Right Foot Swing)
    ax.plot([], [], 
             color=COLOR_CYAN, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Right Foot Swing')
    
    # --- VERTICAL LINE LEGENDS ---
    # 3. Right Foot Hit Ground (RHS event, uses Dark Cyan line: COLOR_RHS_LINE)
    ax.plot([], [], 
             color=COLOR_RHS_LINE, 
             linestyle=':', 
             linewidth=2, 
             alpha=LINE_ALPHA, 
             label='Right Foot Hit Ground')
    
    # 4. Left Foot Hit Ground (LHS event, uses Dark Red line: COLOR_LHS_LINE)
    ax.plot([], [], 
             color=COLOR_LHS_LINE, 
             linestyle=':', 
             linewidth=2, 
             alpha=LINE_ALPHA, 
             label='Left Foot Hit Ground')


# --- 6. Styling and Final Touches ---
time_limit_str = f' (From {min_time}s to {max_time}s)'

for i, data_col in enumerate(PLOT_COLUMNS):
    ax = axes[i]
    
    if i == 0:
        ax.set_title(f'{MODEL_TO_PLOT} Kinematic and Angle Data Comparison{time_limit_str}', fontsize=16)

    ax.set_ylabel(data_col.replace('_', ' ').title(), fontsize=12) 
    
    # --- Grid Density (Doubled) ---
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20)) 
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12)) 
    ax.grid(True, which='major', linestyle='--', alpha=0.2, color='gray') 
    
    if i == num_plots - 1:
        ax.set_xlabel('Time (s)', fontsize=14)
    else:
        ax.tick_params(labelbottom=False)
    
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10)

plt.tight_layout()

# --- FILENAME ---
output_name = f'positions_and_angles_gait_analysis_{MODEL_TO_PLOT}'
# ----------------

# Save the final figure
plt.savefig(f'{output_name}.png')
plt.savefig(f'{output_name}.pdf')