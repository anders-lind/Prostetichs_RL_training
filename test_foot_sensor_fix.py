#!/usr/bin/env python3
"""
Test script for policy evaluation - auto-regenerates gait data without prompting
"""

import sys
if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = input("Enter the log directory: ")

import os
import numpy as np
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.train_log_handler import TrainLogHandler
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
import json
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
    config_dict = json.load(f)
config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

evaluate_param = config.evaluate_param_list[0]  # Use first evaluation param
analyze_result_dir = os.path.join(log_dir, "analyze_results_00")
if not os.path.exists(analyze_result_dir):
    os.makedirs(analyze_result_dir)

log_handler = TrainLogHandler(log_dir)
log_handler.load_log_data(ImitationTrainCheckpointData)

# Initialize gait evaluator
from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.analyzer.gait_data import GaitData

gait_evaluator = ImitationGaitEvaluator(log_handler, config)
gait_evaluator.load_reference_data()
gait_evaluator.initialize_env()

# Force regeneration of gait data
gait_data_name = "gait_evaluated_data.json"
gait_data_path = os.path.join(analyze_result_dir, gait_data_name)

print(f"\n{'='*80}")
print(f"Regenerating gait evaluation data (with fixed sensors)...")
print(f"{'='*80}\n")

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

# Read back and verify the data
print(f"\n{'='*80}")
print(f"Verifying gait data...")
print(f"{'='*80}\n")

gait_data = GaitData()
gait_data.read_json_data(gait_data_path)

# Check sensor data
if "sensor_data" in gait_data.series_data:
    sensors = gait_data.series_data["sensor_data"]
    print("Sensor data collected:")
    for sensor_name in ['r_foot', 'l_foot', 'r_toes', 'l_toes', 'r_ankle_sensor', 'l_ankle_sensor']:
        if sensor_name in sensors:
            values = sensors[sensor_name]["data"]
            non_zero = sum(1 for v in values if isinstance(v, list) and v[0] > 0.1)
            if isinstance(values[0], list):
                min_val = min([v[0] for v in values])
                max_val = max([v[0] for v in values])
            else:
                min_val = min(values)
                max_val = max(values)
            print(f"  {sensor_name:20} | {len(values):4d} entries | Non-zero: {non_zero:3d} | Range: [{min_val:.4f}, {max_val:.4f}]")

print(f"\n{'='*80}")
print(f"FOOT SENSOR FIX TEST RESULTS")
print(f"{'='*80}\n")

r_foot_data = sensors.get('r_foot', {}).get('data', [])
l_foot_data = sensors.get('l_foot', {}).get('data', [])
r_foot_non_zero = sum(1 for v in r_foot_data if isinstance(v, list) and v[0] > 0.1)
l_foot_non_zero = sum(1 for v in l_foot_data if isinstance(v, list) and v[0] > 0.1)

if r_foot_non_zero > 0 or l_foot_non_zero > 0:
    print("✓ SUCCESS! Foot sensors now detect contact!")
    print(f"  r_foot: {r_foot_non_zero} non-zero readings (previously: 0)")
    print(f"  l_foot: {l_foot_non_zero} non-zero readings (previously: 0)")
else:
    print("✗ FAILED! Foot sensors still recording zeros")
    print(f"  r_foot: {r_foot_non_zero} non-zero readings")
    print(f"  l_foot: {l_foot_non_zero} non-zero readings")

print(f"\n{'='*80}\n")
