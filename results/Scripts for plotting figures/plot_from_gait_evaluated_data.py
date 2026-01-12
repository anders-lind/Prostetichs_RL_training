import json
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Define the file path relative to the algorithm directory
file_relative_path = 'gait_evaluated_data.json'

# Define the full data sources dictionary
data_sources = {
    'SAC': os.path.join('SAC', file_relative_path),
    'PPO': os.path.join('PPO', file_relative_path),
    'A2C': os.path.join('A2C', file_relative_path)
}
algos = list(data_sources.keys()) 

# --- CRITICAL: Define the known logging frequency (REQUIRED FOR X-AXIS) ---
LOGGING_FREQUENCY_HZ = 30.0 

# CONTROL TIME WINDOW HERE (based on calculated time in seconds)
min_time = 1.0  
max_time = 8.0

# --- VARIABLES TO PLOT ---
PLOT_VARIABLES = [
    'series_data.actuator_data.Exo_L.force',
    'series_data.joint_data.ankle_angle_l.qpos',
    'series_data.joint_data.mtp_angle_r.qpos'
    # 'series_data.actuator_data.Exo_R.force',
    # 'series_data.sensor_data.r_ankle_sensor.data'

#    - series_data.actuator_data.Exo_L.ctrl
#    - series_data.actuator_data.Exo_L.force
#    - series_data.actuator_data.Exo_L.velocity
#    - series_data.actuator_data.Exo_R.ctrl
#    - series_data.actuator_data.Exo_R.force
#    - series_data.actuator_data.Exo_R.velocity
#    - series_data.actuator_data.bifemsh_l.ctrl
#    - series_data.actuator_data.bifemsh_l.force
#    - series_data.actuator_data.bifemsh_l.velocity
#    - series_data.actuator_data.bifemsh_r.ctrl
#    - series_data.actuator_data.bifemsh_r.force
#    - series_data.actuator_data.bifemsh_r.velocity
#    - series_data.actuator_data.edl_l.ctrl
#    - series_data.actuator_data.edl_l.force
#    - series_data.actuator_data.edl_l.velocity
#    - series_data.actuator_data.edl_r.ctrl
#    - series_data.actuator_data.edl_r.force
#    - series_data.actuator_data.edl_r.velocity
#    - series_data.actuator_data.fdl_l.ctrl
#    - series_data.actuator_data.fdl_l.force
#    - series_data.actuator_data.fdl_l.velocity
#    - series_data.actuator_data.fdl_r.ctrl
#    - series_data.actuator_data.fdl_r.force
#    - series_data.actuator_data.fdl_r.velocity
#    - series_data.actuator_data.gastroc_l.ctrl
#    - series_data.actuator_data.gastroc_l.force
#    - series_data.actuator_data.gastroc_l.velocity
#    - series_data.actuator_data.gastroc_r.ctrl
#    - series_data.actuator_data.gastroc_r.force
#    - series_data.actuator_data.gastroc_r.velocity
#    - series_data.actuator_data.glutmax_l.ctrl
#    - series_data.actuator_data.glutmax_l.force
#    - series_data.actuator_data.glutmax_l.velocity
#    - series_data.actuator_data.glutmax_r.ctrl
#    - series_data.actuator_data.glutmax_r.force
#    - series_data.actuator_data.glutmax_r.velocity
#    - series_data.actuator_data.hamstrings_l.ctrl
#    - series_data.actuator_data.hamstrings_l.force
#    - series_data.actuator_data.hamstrings_l.velocity
#    - series_data.actuator_data.hamstrings_r.ctrl
#    - series_data.actuator_data.hamstrings_r.force
#    - series_data.actuator_data.hamstrings_r.velocity
#    - series_data.actuator_data.iliopsoas_l.ctrl
#    - series_data.actuator_data.iliopsoas_l.force
#    - series_data.actuator_data.iliopsoas_l.velocity
#    - series_data.actuator_data.iliopsoas_r.ctrl
#    - series_data.actuator_data.iliopsoas_r.force
#    - series_data.actuator_data.iliopsoas_r.velocity
#    - series_data.actuator_data.rectfem_l.ctrl
#    - series_data.actuator_data.rectfem_l.force
#    - series_data.actuator_data.rectfem_l.velocity
#    - series_data.actuator_data.rectfem_r.ctrl
#    - series_data.actuator_data.rectfem_r.force
#    - series_data.actuator_data.rectfem_r.velocity
#    - series_data.actuator_data.soleus_l.ctrl
#    - series_data.actuator_data.soleus_l.force
#    - series_data.actuator_data.soleus_l.velocity
#    - series_data.actuator_data.soleus_r.ctrl
#    - series_data.actuator_data.soleus_r.force
#    - series_data.actuator_data.soleus_r.velocity
#    - series_data.actuator_data.tibant_l.ctrl
#    - series_data.actuator_data.tibant_l.force
#    - series_data.actuator_data.tibant_l.velocity
#    - series_data.actuator_data.tibant_r.ctrl
#    - series_data.actuator_data.tibant_r.force
#    - series_data.actuator_data.tibant_r.velocity
#    - series_data.actuator_data.vasti_l.ctrl
#    - series_data.actuator_data.vasti_l.force
#    - series_data.actuator_data.vasti_l.velocity
#    - series_data.actuator_data.vasti_r.ctrl
#    - series_data.actuator_data.vasti_r.force
#    - series_data.actuator_data.vasti_r.velocity
#    - series_data.joint_data.ankle_angle_l.qpos
#    - series_data.joint_data.ankle_angle_l.qvel
#    - series_data.joint_data.ankle_angle_r.qpos
#    - series_data.joint_data.ankle_angle_r.qvel
#    - series_data.joint_data.gastroc_l_med_gas_l-P2_x.qpos
#    - series_data.joint_data.gastroc_l_med_gas_l-P2_x.qvel
#    - series_data.joint_data.gastroc_l_med_gas_l-P2_y.qpos
#    - series_data.joint_data.gastroc_l_med_gas_l-P2_y.qvel
#    - series_data.joint_data.gastroc_l_med_gas_l-P2_z.qpos
#    - series_data.joint_data.gastroc_l_med_gas_l-P2_z.qvel
#    - series_data.joint_data.gastroc_r_med_gas_r-P2_x.qpos
#    - series_data.joint_data.gastroc_r_med_gas_r-P2_x.qvel
#    - series_data.joint_data.gastroc_r_med_gas_r-P2_y.qpos
#    - series_data.joint_data.gastroc_r_med_gas_r-P2_y.qvel
#    - series_data.joint_data.gastroc_r_med_gas_r-P2_z.qpos
#    - series_data.joint_data.gastroc_r_med_gas_r-P2_z.qvel
#    - series_data.joint_data.hamstrings_l_semimem_l-P2_x.qpos
#    - series_data.joint_data.hamstrings_l_semimem_l-P2_x.qvel
#    - series_data.joint_data.hamstrings_l_semimem_l-P2_y.qpos
#    - series_data.joint_data.hamstrings_l_semimem_l-P2_y.qvel
#    - series_data.joint_data.hamstrings_r_semimem_r-P2_x.qpos
#    - series_data.joint_data.hamstrings_r_semimem_r-P2_x.qvel
#    - series_data.joint_data.hamstrings_r_semimem_r-P2_y.qpos
#    - series_data.joint_data.hamstrings_r_semimem_r-P2_y.qvel
#    - series_data.joint_data.hip_flexion_l.qpos
#    - series_data.joint_data.hip_flexion_l.qvel
#    - series_data.joint_data.hip_flexion_r.qpos
#    - series_data.joint_data.hip_flexion_r.qvel
#    - series_data.joint_data.iliopsoas_l_psoas_l-P3_x.qpos
#    - series_data.joint_data.iliopsoas_l_psoas_l-P3_x.qvel
#    - series_data.joint_data.iliopsoas_l_psoas_l-P3_y.qpos
#    - series_data.joint_data.iliopsoas_l_psoas_l-P3_y.qvel
#    - series_data.joint_data.iliopsoas_l_psoas_l-P3_z.qpos
#    - series_data.joint_data.iliopsoas_l_psoas_l-P3_z.qvel
#    - series_data.joint_data.iliopsoas_r_psoas_r-P3_x.qpos
#    - series_data.joint_data.iliopsoas_r_psoas_r-P3_x.qvel
#    - series_data.joint_data.iliopsoas_r_psoas_r-P3_y.qpos
#    - series_data.joint_data.iliopsoas_r_psoas_r-P3_y.qvel
#    - series_data.joint_data.iliopsoas_r_psoas_r-P3_z.qpos
#    - series_data.joint_data.iliopsoas_r_psoas_r-P3_z.qvel
#    - series_data.joint_data.knee_angle_l.qpos
#    - series_data.joint_data.knee_angle_l.qvel
#    - series_data.joint_data.knee_angle_r.qpos
#    - series_data.joint_data.knee_angle_r.qvel
#    - series_data.joint_data.knee_l_translation1.qpos
#    - series_data.joint_data.knee_l_translation1.qvel
#    - series_data.joint_data.knee_l_translation2.qpos
#    - series_data.joint_data.knee_l_translation2.qvel
#    - series_data.joint_data.knee_r_translation1.qpos
#    - series_data.joint_data.knee_r_translation1.qvel
#    - series_data.joint_data.knee_r_translation2.qpos
#    - series_data.joint_data.knee_r_translation2.qvel
#    - series_data.joint_data.l_elbow_flex.qpos
#    - series_data.joint_data.l_elbow_flex.qvel
#    - series_data.joint_data.l_shoulder_abd.qpos
#    - series_data.joint_data.l_shoulder_abd.qvel
#    - series_data.joint_data.l_shoulder_flex.qpos
#    - series_data.joint_data.l_shoulder_flex.qvel
#    - series_data.joint_data.l_shoulder_rot.qpos
#    - series_data.joint_data.l_shoulder_rot.qvel
#    - series_data.joint_data.l_wrist_dev.qpos
#    - series_data.joint_data.l_wrist_dev.qvel
#    - series_data.joint_data.l_wrist_flex.qpos
#    - series_data.joint_data.l_wrist_flex.qvel
#    - series_data.joint_data.l_wrist_rot.qpos
#    - series_data.joint_data.l_wrist_rot.qvel
#    - series_data.joint_data.mtp_angle_l.qpos
#    - series_data.joint_data.mtp_angle_l.qvel
#    - series_data.joint_data.mtp_angle_r.qpos
#    - series_data.joint_data.mtp_angle_r.qvel
#    - series_data.joint_data.pelvis_tilt.qpos
#    - series_data.joint_data.pelvis_tilt.qvel
#    - series_data.joint_data.pelvis_tx.qpos
#    - series_data.joint_data.pelvis_tx.qvel
#    - series_data.joint_data.pelvis_ty.qpos
#    - series_data.joint_data.pelvis_ty.qvel
#    - series_data.joint_data.r_elbow_flex.qpos
#    - series_data.joint_data.r_elbow_flex.qvel
#    - series_data.joint_data.r_shoulder_abd.qpos
#    - series_data.joint_data.r_shoulder_abd.qvel
#    - series_data.joint_data.r_shoulder_flex.qpos
#    - series_data.joint_data.r_shoulder_flex.qvel
#    - series_data.joint_data.r_shoulder_rot.qpos
#    - series_data.joint_data.r_shoulder_rot.qvel
#    - series_data.joint_data.r_wrist_dev.qpos
#    - series_data.joint_data.r_wrist_dev.qvel
#    - series_data.joint_data.r_wrist_flex.qpos
#    - series_data.joint_data.r_wrist_flex.qvel
#    - series_data.joint_data.r_wrist_rot.qpos
#    - series_data.joint_data.r_wrist_rot.qvel
#    - series_data.joint_data.rect_fem_l_rect_fem_l-P3_x.qpos
#    - series_data.joint_data.rect_fem_l_rect_fem_l-P3_x.qvel
#    - series_data.joint_data.rect_fem_l_rect_fem_l-P3_y.qpos
#    - series_data.joint_data.rect_fem_l_rect_fem_l-P3_y.qvel
#    - series_data.joint_data.rect_fem_r_rect_fem_r-P3_x.qpos
#    - series_data.joint_data.rect_fem_r_rect_fem_r-P3_x.qvel
#    - series_data.joint_data.rect_fem_r_rect_fem_r-P3_y.qpos
#    - series_data.joint_data.rect_fem_r_rect_fem_r-P3_y.qvel
#    - series_data.joint_data.vasti_l_vas_int_l-P4_x.qpos
#    - series_data.joint_data.vasti_l_vas_int_l-P4_x.qvel
#    - series_data.joint_data.vasti_l_vas_int_l-P4_y.qpos
#    - series_data.joint_data.vasti_l_vas_int_l-P4_y.qvel
#    - series_data.joint_data.vasti_r_vas_int_r-P4_x.qpos
#    - series_data.joint_data.vasti_r_vas_int_r-P4_x.qvel
#    - series_data.joint_data.vasti_r_vas_int_r-P4_y.qpos
#    - series_data.joint_data.vasti_r_vas_int_r-P4_y.qvel
#    - series_data.physics_data.contacts.data
#    - series_data.sensor_data.l_ankle_sensor.data
#    - series_data.sensor_data.l_foot.data
#    - series_data.sensor_data.l_hip_sensor.data
#    - series_data.sensor_data.l_knee_sensor.data
#    - series_data.sensor_data.l_mtp_sensor.data
#    - series_data.sensor_data.l_toes.data
#    - series_data.sensor_data.r_ankle_sensor.data
#    - series_data.sensor_data.r_foot.data
#    - series_data.sensor_data.r_hip_sensor.data
#    - series_data.sensor_data.r_knee_sensor.data
#    - series_data.sensor_data.r_mtp_sensor.data
#    - series_data.sensor_data.r_toes.data
#    - series_data.target_data.target_velocity
]
# ----------------------------------------------------------------------------

# Define the color mapping for consistency
COLOR_MAP = {
    'SAC': '#1f77b4', # Blue
    'PPO': '#ff7f0e', # Orange
    'A2C': '#2ca02c'  # Green
}

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
        # Flatten nested list structures like [[v1], [v2], ...] to [v1, v2, ...]
        if isinstance(value, list) and value and isinstance(value[0], list) and len(value[0]) == 1:
             return np.array([item[0] for item in value])
        # Convert non-nested list to NumPy array
        if isinstance(value, list):
             return np.array(value)
        return None
    except (KeyError, IndexError, TypeError):
        return None

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

num_plots = len(PLOT_VARIABLES)
if num_plots == 0:
    print("❌ ERROR: PLOT_VARIABLES list is empty. Please specify which variables to plot.")
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
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            
        # --- CALCULATE TIME AXIS ---
        first_array_path = PLOT_VARIABLES[0]
        first_array = get_nested_value(json_data, first_array_path)
        
        if first_array is None or first_array.size == 0:
             print(f"❌ Error for {algo}: Could not find or access the first required variable '{first_array_path}'. Skipping.")
             continue

        data_length = first_array.size
        time_array = np.arange(data_length) / LOGGING_FREQUENCY_HZ
        
        # Find indices corresponding to the time window
        start_index = np.searchsorted(time_array, min_time, side='left')
        end_index = np.searchsorted(time_array, max_time, side='right')
        
        # Slice the time array
        time_slice = time_array[start_index:end_index]
        
        if time_slice.size == 0:
            print(f"Skipping {algo}: No data available in the time range.")
            continue
            
        # --- Plotting Loop ---
        color = COLOR_MAP.get(algo, 'gray')
        
        for i, data_path in enumerate(PLOT_VARIABLES):
            ax = axes[i]
            
            data_array = get_nested_value(json_data, data_path)
            
            if data_array is not None and data_array.size >= end_index:
                 # Extract the data slice corresponding to the time window
                 data_slice = data_array[start_index:end_index]
                 
                 # Apply SAC scaling (division by 64)
                #  if algo == 'SAC':
                #      data_slice = data_slice / 64.0
                 
                 # Plot the data
                 ax.plot(time_slice, data_slice, 
                         linestyle='-', linewidth=1.5, label=algo, color=color)
            else:
                 print(f"⚠️ Warning for {algo}: Data array '{data_path}' is too short or missing. Skipping plot.")
            
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON for {algo}: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing {algo}: {e}")

# --- Styling and Final Touches ---
time_limit_str = f' (From {min_time}s to {max_time}s)'

for i, data_path in enumerate(PLOT_VARIABLES):
    ax = axes[i]
    
    # Title only for the first plot
    if i == 0:
        ax.set_title(f'Actuator Data Comparison{time_limit_str}', fontsize=16)

    # --- UPDATED Y-AXIS LABEL LOGIC ---
    parts = data_path.split('.')
    # Format: Last_Part_2.Last_Part_1
    y_label = f"{parts[-2]}.{parts[-1]}"
    # ----------------------------------
    
    ax.set_ylabel(y_label, fontsize=12) 
    
    # --- Grid Density (Contrained for large data points) ---
    # X-axis ticks 
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20)) 
    
    # Y-axis ticks 
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

# --- FILENAME ---
output_name = 'gait_data'
# ----------------

# Save the final figure
plt.savefig(f'{output_name}.png')
plt.savefig(f'{output_name}.pdf')
print(f"\n✅ Plots generated and saved to '{output_name}.png' and .pdf")