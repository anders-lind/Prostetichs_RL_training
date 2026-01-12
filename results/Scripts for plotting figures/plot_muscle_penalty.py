import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# File specification using the dictionary convention
data_sources = {
    'SAC': 'SAC/train_log_plots/train_log_data.csv',
    'PPO': 'PPO/train_log_plots/train_log_data.csv',
    'A2C': 'A2C/train_log_plots/train_log_data.csv'
}

# Name of the column containing the training step data (X-axis)
time_column_name = 'num_timesteps' 

# CONTROL TIMESTEP WINDOW HERE
min_step = None 
max_step = 30000000 # 30 million timesteps

# Name of the column containing the penalty data
penalty_column_name = 'muscle_activation_penalty_per_step'

# Define the color mapping based on Matplotlib's default cycle (tab10)
# Order: SAC (Blue), PPO (Orange), A2C (Green)
COLOR_MAP = {
    'SAC': '#1f77b4', # Matplotlib Default Blue
    'PPO': '#ff7f0e', # Matplotlib Default Orange
    'A2C': '#2ca02c'  # Matplotlib Default Green
}

# ---------------------------------------------------------
# Data Processing Function
# ---------------------------------------------------------

def load_and_process_data(data_sources, time_col, penalty_col, min_s, max_s):
    """Loads, filters, cleans, and returns processed data frames."""
    processed_data = {}
    summary_data = []
    sac_scaling_factor = 64
    plot_x_max = max_s or 0

    for algo, file_path in data_sources.items():
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found for {algo} at {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip() 
            
            if time_col not in df.columns or penalty_col not in df.columns:
                print(f"❌ Error: Required column(s) not found in {file_path}")
                continue

            # FIX 1: Explicitly convert the penalty column to numeric, coercing errors to NaN
            df[penalty_col] = pd.to_numeric(df[penalty_col], errors='coerce')
            
            # --- SAC specific scaling ---
            if algo == 'SAC':
                df[penalty_col] = df[penalty_col] / sac_scaling_factor
            # --------------------------
            
            # Filter data by Timestep (From and To)
            if min_s is not None:
                df = df[df[time_col] >= min_s]
            if max_s is not None:
                df = df[df[time_col] <= max_s]

            # CRITICAL STEP: CLEAN DATA - Keep only recorded samples
            df_final = df.dropna(subset=[time_col, penalty_col])

            if df_final.empty:
                print(f"Skipping {algo}: No valid data remains after processing.")
                continue
            
            processed_data[algo] = df_final
            
            # Calculate and store Average Penalty
            avg_penalty = df_final[penalty_col].mean()
            summary_data.append({
                'Algorithm': algo,
                'Avg Penalty per Step': avg_penalty,
            })
            
        except Exception as e:
            print(f"❌ Unhandled Error processing {file_path}: {e}")

    return processed_data, summary_data, plot_x_max

# Run processing
processed_data, summary_data, plot_x_max = load_and_process_data(
    data_sources, time_column_name, penalty_column_name, min_step, max_step
)

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

time_range_str = f' (Timesteps: 0 to {plot_x_max})'
y_label = 'Muscle Activation Penalty (per Step)'

# --- Single Plot: All Algorithms Comparison ---

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

if processed_data:
    for algo, df in processed_data.items():
        label = algo
        color = COLOR_MAP.get(algo, 'gray')
        
        if algo == 'SAC':
            # SAC: Connected dots
            ax.plot(df[time_column_name], df[penalty_column_name], 
                    linestyle='-', marker='o', markersize=4, 
                    label=label, color=color, alpha=0.8, linewidth=1.5)
        else:
            # PPO/A2C: Continuous lines
            ax.plot(df[time_column_name], df[penalty_column_name], 
                    linestyle='-', linewidth=2.5, label=label, color=color)

    # X-axis range from 0 to max_step (30 million)
    ax.set_xlim(left=0, right=plot_x_max)
    
    ax.set_title(f'{y_label}: All Algorithms Comparison{time_range_str}', fontsize=16)
    ax.set_xlabel('Training Timesteps', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=12)
    fig.tight_layout() 
    
    plt.savefig('muscle_penalty.png')
    plt.savefig('muscle_penalty.pdf')
    print("\n✅ Final Comparison Plot saved to 'muscle_penalty.png' and .pdf")
else:
    print("\n⚠️ No data processed to generate the Final Comparison Plot.")

# ---------------------------------------------------------
# Print Summary Table (Scaling Mention Kept Here for Documentation)
# ---------------------------------------------------------

if summary_data:
    print("\n" + "="*75)
    print(f"📊 AVERAGE PENALTY SUMMARY (Per Step) {time_range_str}")
    print("NOTE: SAC values were divided by 64 for internal scaling and averaging.")
    print("="*75)
    
    summary_df = pd.DataFrame(summary_data)
    float_formatter = lambda x: f"{x:10.4f}"
    
    print(summary_df.to_string(index=False, formatters={'Avg Penalty per Step': float_formatter}))
    print("="*75)
else:
    print("\nNo valid data to generate summary averages.")