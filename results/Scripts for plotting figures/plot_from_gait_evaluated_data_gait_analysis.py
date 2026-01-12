import json
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# --- SELECT MODEL TO PLOT ---
# CHANGE THIS to 'SAC', 'PPO', or 'A2C'
MODEL_TO_PLOT = 'PPO' 
# ----------------------------

# File names
file_relative_path = 'gait_evaluated_data.json'
steps_file = 'steps.csv' # File containing event data

# Define the full data sources dictionary (used only to derive file path for the selected model)
data_sources = {
    'SAC': os.path.join('SAC', file_relative_path),
    'PPO': os.path.join('PPO', file_relative_path),
    'A2C': os.path.join('A2C', file_relative_path)
}

# --- CRITICAL: Define the known logging frequency (REQUIRED FOR X-AXIS) ---
LOGGING_FREQUENCY_HZ = 30.0 

# CONTROL TIME WINDOW HERE (based on calculated time in seconds)
min_time = 1.0  
max_time = 8.0

# --- VARIABLES TO PLOT (UPDATED TO THE USER'S LIST) ---
PLOT_VARIABLES = [
    'series_data.actuator_data.Exo_L.force',
    'series_data.joint_data.ankle_angle_l.qpos',
    'series_data.joint_data.mtp_angle_r.qpos'
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
# RHS event line (index 0, 2, ...) now uses Dark Cyan line (COLOR_LHS)
# LHS event line (index 1, 3, ...) now uses Dark Red line (COLOR_RHS)
COLOR_RHS_LINE = '#008B8B' # Dark Cyan 
COLOR_LHS_LINE = '#8B0000' # Dark Red 
LINE_ALPHA = 0.8
# -------------------------------------


# ---------------------------------------------------------
# Helper Function to Extract Nested Data
# ---------------------------------------------------------
def get_nested_value(data, path_str):
    """Accesses a nested value in a dictionary using a dot-separated string path."""
    keys = path_str.split('.')
    value = data
    try:
        for key in keys:
            value = value[key]
        if isinstance(value, list) and value and isinstance(value[0], list) and len(value[0]) == 1:
             return np.array([item[0] for item in value])
        if isinstance(value, list):
             return np.array(value)
        return None
    except (KeyError, IndexError, TypeError):
        return None

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

num_plots = len(PLOT_VARIABLES)
if num_plots == 0:
    print("❌ ERROR: PLOT_VARIABLES list is empty. Please specify which variables to plot.")
    exit()
if MODEL_TO_PLOT not in data_sources:
    print(f"❌ ERROR: Invalid model selected: {MODEL_TO_PLOT}. Must be SAC, PPO, or A2C.")
    exit()

fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
if num_plots == 1:
    axes = np.array([axes])
plt.subplots_adjust(hspace=0.2) 

algo = MODEL_TO_PLOT
file_path = data_sources[algo]
color = COLOR_MAP.get(algo, 'gray')
plot_successful = False


if not os.path.exists(file_path):
    print(f"⚠️ Warning: File not found for {algo} at {file_path}")
else:
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            
        # --- CALCULATE TIME AXIS ---
        first_array_path = PLOT_VARIABLES[0]
        first_array = get_nested_value(json_data, first_array_path)
        
        if first_array is None or first_array.size == 0:
             print(f"❌ Error for {algo}: Could not find or access the first required variable '{first_array_path}'. Skipping.")
        else:
            data_length = first_array.size
            time_array = np.arange(data_length) / LOGGING_FREQUENCY_HZ
            
            # Find indices corresponding to the time window
            start_index = np.searchsorted(time_array, min_time, side='left')
            end_index = np.searchsorted(time_array, max_time, side='right')
            
            # Slice the time array
            time_slice = time_array[start_index:end_index]
            
            if time_slice.size == 0:
                print(f"Skipping {algo}: No data available in the time range.")
            else:
                for i, data_path in enumerate(PLOT_VARIABLES):
                    ax = axes[i]
                    data_array = get_nested_value(json_data, data_path)
                    
                    if data_array is not None and data_array.size >= end_index:
                         data_slice = data_array[start_index:end_index]
                         
                         # Apply SAC scaling (division by 64)
                         if algo == 'SAC':
                             data_slice = data_slice / 64.0
                         
                         # Plot the data
                         ax.plot(time_slice, data_slice, 
                                 linestyle='-', linewidth=1.5, label=algo, color=color)
                         
                         plot_successful = True
                    else:
                         print(f"⚠️ Warning for {algo}: Data array '{data_path}' is too short or missing. Skipping plot.")
            
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON for {algo}: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing {algo}: {e}")

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
            fill_color = COLOR_RED # Red shade for interval starting after RHS event
        else:
            fill_color = COLOR_CYAN # Cyan shade for interval starting after LHS event
            
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
    # 1. Red Shade (Currently in plot for Right Foot Hit Strike -> Left Foot Swing)
    ax.plot([], [], 
             color=COLOR_RED, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Left Foot Swing') 
    
    # 2. Cyan Shade (Currently in plot for Left Foot Hit Strike -> Right Foot Swing)
    ax.plot([], [], 
             color=COLOR_CYAN, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Right Foot Swing')
    
    # --- VERTICAL LINE LEGENDS (Labels stay, Colors match section 4) ---
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

for i, data_path in enumerate(PLOT_VARIABLES):
    ax = axes[i]
    
    if i == 0:
        ax.set_title(f'{MODEL_TO_PLOT} Actuator Data Comparison{time_limit_str}', fontsize=16)

    # --- Y-AXIS LABEL LOGIC (Last two parts of the path) ---
    parts = data_path.split('.')
    y_label = f"{parts[-2]}.{parts[-1]}"
    # -------------------------------------------------------
    
    ax.set_ylabel(y_label, fontsize=12) 
    
    # --- Grid Density (Contrained for large data points) ---
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
output_name = f'gait_data_gait_cycle_{MODEL_TO_PLOT}'
# ----------------

# Save the final figure
plt.savefig(f'{output_name}.png')
plt.savefig(f'{output_name}.pdf')