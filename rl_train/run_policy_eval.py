import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Setup and Config Loading
# ---------------------------------------------------------

if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = ""

if log_dir == "":
    log_dir = input("Enter the log directory: ")

show_plot = False

from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.train_log_handler import TrainLogHandler
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_evaluate import GaitData, ImitationGaitEvaluator

# Load Configuration
with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
    config_dict = json.load(f)
config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

# ---------------------------------------------------------
# 2. Main Evaluation Loop
# ---------------------------------------------------------
for (idx, evaluate_param) in enumerate(config.evaluate_param_list):
    analyze_result_dir = os.path.join(log_dir, f"analyze_results_{idx:02d}")
    if not os.path.exists(analyze_result_dir):
        os.makedirs(analyze_result_dir)

    # Load Log Data
    log_handler = TrainLogHandler(log_dir)
    log_handler.load_log_data(ImitationTrainCheckpointData)

    DictionableDataclass.to_dict(log_handler.log_datas[-1])

    sys.modules.pop('package.train_log_analyzer', None)
    from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
    train_log_analyzer = TrainLogAnalyzer(log_handler)
    train_log_analyzer.plot_reward(result_dir=analyze_result_dir, show_plot=show_plot)

    # ---------------------------------------------------------
    # 3. Run Evaluation (Physics Simulation)
    # ---------------------------------------------------------
    gait_data_name = f"gait_evaluated_data.json"
    gait_data_path = os.path.join(analyze_result_dir, gait_data_name)
    
    if os.path.exists(gait_data_path):
        user_input = input(f"Regenerate evaluate data? ({gait_data_name}) (y/n(anything))")
    else:
        user_input = "y"
    is_regen_evaluating_data = True if user_input == "y" else False

    gait_evaluator = ImitationGaitEvaluator(log_handler, config)
    gait_evaluator.load_reference_data()
    gait_evaluator.initialize_env()

    if is_regen_evaluating_data:
        gait_data_path = gait_evaluator.evaluate(
            result_dir=analyze_result_dir,
            file_name=gait_data_name,
            velocity_mode=MyoAssistLegBase.VelocityMode[evaluate_param["velocity_mode"]],
            target_velocity_period=evaluate_param["target_velocity_period"],
            max_timestep=evaluate_param["num_timesteps"],
            min_target_velocity=evaluate_param["min_target_velocity"],
            max_target_velocity=evaluate_param["max_target_velocity"],
            terminate_when_done=True
        )

    # Load the generated data
    gait_data = GaitData()
    gait_data.read_json_data(gait_data_path)

    # =========================================================
    # 4. CUSTOM PLOTTING & CSV EXPORT: Trajectory Analysis
    # =========================================================
    print(f"Generating Raw Trajectory Plots & CSV in {analyze_result_dir}...")

    def extract_numeric_data(source_dict, key):
        """Helper to safely extract float array from potentially nested dicts."""
        if key not in source_dict:
            return None
        
        raw = source_dict[key]
        
        if isinstance(raw, dict):
            if 'qpos' in raw:
                raw = raw['qpos']
            elif 'data' in raw:
                raw = raw['data']
            else:
                return None
        
        try:
            return np.array(raw, dtype=np.float64)
        except Exception:
            return None

    try:
        series = getattr(gait_data, 'series_data', {})
        
        # Look in likely locations
        search_locations = [
            series.get('joint_data', {}), 
            series.get('physics_data', {})
        ]

        framerate = config.env_params.control_framerate
        x_key = 'pelvis_tx'
        z_key = 'pelvis_ty'
        
        z_values = None
        x_values = None

        for loc in search_locations:
            if not loc: continue
            
            temp_z = extract_numeric_data(loc, z_key)
            
            if temp_z is not None:
                z_values = np.atleast_1d(temp_z)
                temp_x = extract_numeric_data(loc, x_key)
                if temp_x is not None:
                    x_values = np.atleast_1d(temp_x)
                else:
                    x_values = np.zeros_like(z_values)
                break
        
        if z_values is None:
            print(f"  -> ERROR: Could not find numeric data for '{z_key}'.")
        else:
            num_steps = len(z_values)
            time = np.linspace(0, num_steps / framerate, num_steps)

            # --- 1. GENERATE PLOT ---
            plt.figure(figsize=(10, 8))
            
            # Subplot 1: Forward (X)
            plt.subplot(2, 1, 1)
            plt.plot(time, x_values, label=f'Forward ({x_key})', color='blue', linewidth=1.5)
            plt.title('Global Forward Position x(t)')
            plt.ylabel('Position [m]')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Subplot 2: Height (Z/Y)
            plt.subplot(2, 1, 2)
            plt.plot(time, z_values, label=f'Height ({z_key})', color='green', linewidth=1.5)
            safe_height = getattr(config.env_params, 'safe_height', 0.7)
            plt.axhline(y=safe_height, color='r', linestyle='--', alpha=0.7, label=f'Safe Height ({safe_height}m)')
            plt.title('Global Vertical Position z(t)')
            plt.xlabel('Time [s]')
            plt.ylabel('Height [m]')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.tight_layout()
            save_path = os.path.join(analyze_result_dir, "trajectory_analysis_xz.png")
            plt.savefig(save_path)
            plt.close()
            print(f"  -> Trajectory plot saved to {save_path}")

            # --- 2. SAVE CSV (t, x, y) ---
            csv_path = os.path.join(analyze_result_dir, "trajectory.csv")
            
            # Stack the data columns: Time, X (Forward), Z (Height/Y)
            data_stack = np.column_stack((time, x_values, z_values))
            
            # Save using numpy
            # header="t,x,y" creates the column names
            # comments="" removes the default "# " hash from the header
            np.savetxt(csv_path, data_stack, delimiter=",", header="t,x,y", comments="", fmt="%.6f")
            
            print(f"  -> Trajectory CSV saved to {csv_path}")

    except Exception as e:
        print(f"  -> Error processing trajectory: {e}")
        import traceback
        traceback.print_exc()
    # =========================================================

    # ---------------------------------------------------------
    # 5. Video Replay Generation
    # ---------------------------------------------------------
    gait_evaluator.replay(
        gait_data_path, 
        os.path.join(analyze_result_dir, "replay.mp4"),
        cam_distance=evaluate_param["cam_distance"],
        use_activation_visualization=evaluate_param["visualize_activation"],
        cam_type=evaluate_param["cam_type"],
        realtime_plotting_info=evaluate_param.get("realtime_plotting_info", []),
        video_fps=config.env_params.control_framerate
    )

    # ---------------------------------------------------------
    # 6. Detailed Gait Analysis
    # ---------------------------------------------------------
    segmented_ref_data = np.load("rl_train/reference_data/segmented.npz", allow_pickle=True)
    segmented_ref_data = {key: segmented_ref_data[key] for key in segmented_ref_data.files}

    gait_analyzer = GaitAnalyzer(gait_data, segmented_ref_data, show_plot)

    if len(gait_analyzer.get_gait_segment_index(is_right_foot_based=True)) < 1:
        print("="*10 + "Warning" + "="*10)
        print("Warning! Not enough gait data to plot standard metrics. Skipping detailed plotting.")
        print("="*10 + "Warning" + "="*10)
        continue

    gait_analyzer.plot_entire_result(result_dir=analyze_result_dir, is_right_foot_based=True)
    gait_analyzer.plot_exo_segmented_data(result_dir=analyze_result_dir)
    gait_analyzer.plot_segmented_kinematics_result(result_dir=analyze_result_dir)
    gait_analyzer.plot_left_right_comparison(result_dir=analyze_result_dir)
    gait_analyzer.plot_right_ref_comparison(result_dir=analyze_result_dir)
    gait_analyzer.plot_segmented_muscle_data(result_dir=analyze_result_dir, is_plot_right=True)
    gait_analyzer.joint_angle_by_velocity(result_dir=analyze_result_dir)