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
MODEL_TO_PLOT = 'SAC'
# ----------------------------

# File names
file_relative_path = 'trajectory.csv' 
steps_file = 'steps.csv' # File containing event timesteps

# CONTROL TIME WINDOW HERE
min_time = 2.0  
max_time = 4.0  

# --- PLOTTING CONSTANTS (For Gait Cycle Visualization) ---
COLOR_RED = 'red'
COLOR_CYAN = 'cyan'
SHADE_ALPHA = 0.1
COLOR_RHS = '#008B8B' # Dark Cyan (RHS line, matching LHS swing interval color in previous logic)
COLOR_LHS = '#8B0000' # Dark Red (LHS line, matching RHS swing interval color in previous logic)
LINE_ALPHA = 0.8
# ---------------------------------------------------------

# --- TREND LINE CONFIGURATION (RUNNING AVERAGE) ---
plot_trend_lines = False 
trend_window_size = 50
# --------------------------------------------------

# --- Selective plotting variables (set True/False) ---
# Toggle these to choose which signals to plot (no CLI required)
plot_pos = True    # Position (x, y)
plot_vel = True    # Velocity (v_x, v_y)
plot_acc = True    # Acceleration (a_x, a_y)
plot_jerk = False   # Jerk (j_x, j_y)
# You can also change MODEL_TO_PLOT, min_time and max_time above directly.
# --------------------------------------------------

# --- FONT SIZE CONFIGURATION ---
# Adjust these to control font sizes used in the figure
TITLE_FONT_SIZE = 14*2
LABEL_FONT_SIZE = 12*2
LEGEND_FONT_SIZE = 10*1.7
TICK_FONT_SIZE = 10*2
# --------------------------------------------------

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
# Build dynamic plot layout based on selected signal types
enabled_types = []
if plot_pos:
    enabled_types.append(('pos', 'Position'))
if plot_vel:
    enabled_types.append(('vel', 'Velocity'))
if plot_acc:
    enabled_types.append(('acc', 'Acceleration'))
if plot_jerk:
    enabled_types.append(('jerk', 'Jerk'))

# At least one type should be enabled; if none, default to position
if not enabled_types:
    enabled_types = [('pos', 'Position')]

# Number of rows equals number of enabled signal types
nrows = len(enabled_types)
fig, axes = plt.subplots(nrows, 2, figsize=(15, 5 * nrows), sharex=True)
plt.subplots_adjust(hspace=0.4)

# Normalize axes shape to 2D array for consistent indexing
if nrows == 1:
    axes = np.array([axes])

# Build plot_specs dynamically: (row, col, data_col, title, y_label, data_type)
plot_specs = []
for row_idx, (dtype, dtype_label) in enumerate(enabled_types):
    # X column (col 0) and Y column (col 1)
    if dtype == 'pos':
        plot_specs.append((row_idx, 0, 'x', f'X {dtype_label}', 'X Position', 'pos'))
        plot_specs.append((row_idx, 1, 'y', f'Y {dtype_label}', 'Y Position', 'pos'))
    elif dtype == 'vel':
        plot_specs.append((row_idx, 0, 'v_x', f'X {dtype_label}', 'X Velocity', 'vel'))
        plot_specs.append((row_idx, 1, 'v_y', f'Y {dtype_label}', 'Y Velocity', 'vel'))
    elif dtype == 'acc':
        plot_specs.append((row_idx, 0, 'a_x', f'X {dtype_label}', 'X Acceleration', 'acc'))
        plot_specs.append((row_idx, 1, 'a_y', f'Y {dtype_label}', 'Y Acceleration', 'acc'))
    elif dtype == 'jerk':
        plot_specs.append((row_idx, 0, 'j_x', f'X {dtype_label}', 'X Jerk', 'jerk'))
        plot_specs.append((row_idx, 1, 'j_y', f'Y {dtype_label}', 'Y Jerk', 'jerk'))

# Process only the selected model
algo = MODEL_TO_PLOT
file_path = os.path.join(algo, file_relative_path)
plot_successful = False

if not os.path.exists(file_path):
    print(f"Warning: File not found for {algo} at {file_path}")
else:
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
        else:
            # --- Calculate Derivatives (Speed, Acceleration, Jerk) ---
            df = calculate_kinematics(df.copy(), 'x')
            df = calculate_kinematics(df.copy(), 'y')

            # Create a cleaned dataframe depending on which derivative columns are required
            dropna_cols = []
            if plot_vel:
                dropna_cols += ['v_x', 'v_y']
            if plot_acc:
                dropna_cols += ['a_x', 'a_y']
            if plot_jerk:
                dropna_cols += ['j_x', 'j_y']

            if dropna_cols:
                df_clean = df.dropna(subset=dropna_cols)
            else:
                df_clean = df
            
            # --- Plotting Loop (respects selective plotting variables) ---
            for row, col, data_col, _, y_label, data_type in plot_specs:
                ax = axes[row, col]

                # Skip/hide axes if the user disabled this data_type
                if (data_type == 'pos' and not plot_pos) or \
                   (data_type == 'vel' and not plot_vel) or \
                   (data_type == 'acc' and not plot_acc) or \
                   (data_type == 'jerk' and not plot_jerk):
                    ax.set_visible(False)
                    continue

                plot_df = df if data_type == 'pos' else df_clean

                # Plot the raw data
                ax.plot(plot_df['t'], plot_df[data_col],
                    linestyle='-', linewidth=1.5, alpha=0.8,
                    color=None) # Color will be default or cycle color

                # --- RUNNING AVERAGE TREND LINE LOGIC (apply only for velocity rows) ---
                if plot_trend_lines and data_type == 'vel':
                    if ax.lines:
                        line_color = ax.lines[-1].get_color()
                    else:
                        line_color = None

                    if len(df_clean) >= trend_window_size:
                        running_avg = df_clean[data_col].rolling(
                            window=trend_window_size,
                            center=True,
                            min_periods=1
                        ).mean()

                        ax.plot(df_clean['t'], running_avg,
                                linestyle='--', linewidth=3,
                                color=line_color,
                                label=f'{algo} Avg (Window {trend_window_size})')
            
            plot_successful = True # Set flag only if data was loaded and plotted

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

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
            
        # Apply shading to all 8 subplots
        for ax in axes.flatten():
             ax.axvspan(t_start, t_end, color=fill_color, alpha=SHADE_ALPHA, zorder=0)


# --- 4. Plot Vertical Event Lines (Colored) ---
if plot_successful and df_steps_data:
    for index, event_time in enumerate(df_steps_data):
        
        # --- COLOR SWAP: RHS line is Dark Cyan, LHS line is Dark Red ---
        if index % 2 == 0:
            line_color = COLOR_LHS  # RHS event (0, 2, ...) now uses Dark Cyan
        else:
            line_color = COLOR_RHS  # LHS event (1, 3, ...) now uses Dark Red

        # Apply lines to all 8 subplots
        for ax in axes.flatten():
            ax.axvline(x=event_time, color=line_color, linestyle=':', linewidth=2, alpha=LINE_ALPHA, zorder=1)


# --- 5. Add Dummy Plots for Legend ---
if plot_successful:
    ax = axes[0, 0] # Use the top-left plot for the combined legend
    
    # --- SWAPPED SHADING LEGEND LABELS ---
    # 1. Red Shade (Right Foot Hit Strike -> Left Foot Swing)
    ax.plot([], [], 
             color=COLOR_RED, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Left Foot Swing') 
    
    # 2. Cyan Shade (Left Foot Hit Strike -> Right Foot Swing)
    ax.plot([], [], 
             color=COLOR_CYAN, 
             alpha=SHADE_ALPHA * 10, 
             linewidth=8, 
             label='Right Foot Swing')
    
    # --- VERTICAL LINE LEGENDS (Colors match section 4, Dark Cyan/Dark Red) ---
    # 3. Right Foot Hit Ground (RHS event, uses Dark Cyan line: COLOR_LHS)
    ax.plot([], [], 
             color=COLOR_LHS, 
             linestyle=':', 
             linewidth=2, 
             alpha=LINE_ALPHA, 
             label='Right Foot Hit Ground')
    
    # 4. Left Foot Hit Ground (LHS event, uses Dark Red line: COLOR_RHS)
    ax.plot([], [], 
             color=COLOR_RHS, 
             linestyle=':', 
             linewidth=2, 
             alpha=LINE_ALPHA, 
             label='Left Foot Hit Ground')


# --- 6. Styling and Final Touches ---
time_limit_str = ''

for row, col, data_col, title, y_label, data_type in plot_specs:
    ax = axes[row, col]

    # Skip styling for axes that were turned off
    if not ax.get_visible():
        continue

    ax.set_title(f'{MODEL_TO_PLOT} {title} vs Time{time_limit_str}', fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Tick label sizes
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # Legend control: Add combined legend to the top-left plot (if visible)
    if row == 0 and col == 0 and ax.get_visible():
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', fontsize=LEGEND_FONT_SIZE)


# Only add X-axis label to the bottom plots (last enabled row)
axes[nrows-1, 0].set_xlabel('Time (s)', fontsize=LABEL_FONT_SIZE)
axes[nrows-1, 1].set_xlabel('Time (s)', fontsize=LABEL_FONT_SIZE)

plt.tight_layout()

# Save the final figure
output_name = f'trajectory_extended_gait_cycle{"_with_trend_lines" if plot_trend_lines else ""}_{MODEL_TO_PLOT}'
plt.savefig(f'{output_name}.png')
plt.savefig(f'{output_name}.pdf')

# Note: The output includes a large number of print statements from derivative calculation warnings 
# (NaNs) and the file-not-found warning, which is expected based on the script's design.