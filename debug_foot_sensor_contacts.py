#!/usr/bin/env python3
"""
Deep debugging script to understand why foot sensors aren't detecting contact
"""

import sys
import json
import os

if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = input("Enter the log directory: ")

from rl_train.utils.data_types import DictionableDataclass
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.train_log_handler import TrainLogHandler
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.analyzer.gait_data import GaitData

import numpy as np
import mujoco

with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
    config_dict = json.load(f)
config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

log_handler = TrainLogHandler(log_dir)
log_handler.load_log_data(ImitationTrainCheckpointData)

config.env_params.num_envs = 1
config.env_params.custom_max_episode_steps = 1000000000
config.env_params.out_of_trajectory_threshold = 1000000

env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
trained_model_path = log_handler.get_path2save_model(log_handler.log_datas[-1].num_timesteps)
model = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=trained_model_path)

env_myoassist = env.unwrapped
env_myoassist.set_target_velocity_mode_manually(MyoAssistLegBase.VelocityMode.CONSTANT, 0, 1.0, 0.5, 1.5, target_velocity_period=2.0)

mj_model = env.sim.model
mj_data = env.sim.data

print("="*80)
print("DEBUGGING FOOT SENSOR CONTACT DETECTION")
print("="*80)

# Check if sensors exist
print("\nChecking sensors in model:")
for idx in range(mj_model.nsensor):
    name = mj_model.sensor(idx).name
    sensor_type = mj_model.sensor(idx).type
    if 'foot' in name or 'toe' in name:
        print(f"  {name}: type={sensor_type}")

# Reset and run a few steps
obs, info = env.reset()

print("\n" + "="*80)
print("STEP-BY-STEP CONTACT ANALYSIS")
print("="*80)

for step in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
    
    # Get sensor values
    r_foot_sensor = mj_data.sensor("r_foot").data.copy()[0]
    l_foot_sensor = mj_data.sensor("l_foot").data.copy()[0]
    r_toes_sensor = mj_data.sensor("r_toes").data.copy()[0]
    l_toes_sensor = mj_data.sensor("l_toes").data.copy()[0]
    
    # Check contacts
    num_contacts = mj_data.ncon
    foot_contacts = []
    
    for i in range(num_contacts):
        contact = mj_data.contact[i]
        geom1_name = mj_model.id2name(contact.geom1, 'geom')
        geom2_name = mj_model.id2name(contact.geom2, 'geom')
        
        # Check if foot geometries are in contact
        if ('calcn' in geom1_name or 'calcn' in geom2_name or 
            'foot' in geom1_name or 'foot' in geom2_name or
            'talus' in geom1_name or 'talus' in geom2_name):
            
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(mj_model.ptr, mj_data.ptr, i, force)
            foot_contacts.append({
                'geom1': geom1_name,
                'geom2': geom2_name,
                'force_mag': np.linalg.norm(force[:3])
            })
    
    # Get site positions
    r_foot_touch_pos = mj_data.site("r_foot_touch").xpos.copy()
    r_toes_touch_pos = mj_data.site("r_toes_touch").xpos.copy()
    ground_y = 0  # Assuming ground is at y=0
    
    if step == 0 or (step > 0 and (r_foot_sensor > 0.01 or len(foot_contacts) > 0)):
        print(f"\nSTEP {step}:")
        print(f"  Sensors: r_foot={r_foot_sensor:.4f}, l_foot={l_foot_sensor:.4f}, r_toes={r_toes_sensor:.4f}, l_toes={l_toes_sensor:.4f}")
        print(f"  r_foot_touch site pos: {r_foot_touch_pos}")
        print(f"  r_toes_touch site pos: {r_toes_touch_pos}")
        print(f"  Total contacts: {num_contacts}")
        if foot_contacts:
            print(f"  Foot contacts ({len(foot_contacts)}):")
            for fc in foot_contacts:
                print(f"    {fc['geom1']} <-> {fc['geom2']}: force_mag={fc['force_mag']:.4f}")
        else:
            print(f"  No foot/heel contacts detected!")

print("\n" + "="*80)
print("CHECKING SITE DIMENSIONS AND PARENT BODIES")
print("="*80)

try:
    # Get info about r_foot_touch site
    r_foot_site_id = mj_model.site("r_foot_touch")
    r_foot_body_id = mj_model.site(r_foot_site_id).bodyid
    r_foot_body_name = mj_model.body(r_foot_body_id).name
    r_foot_size = mj_model.site(r_foot_site_id).size
    
    print(f"\nr_foot_touch site:")
    print(f"  Body: {r_foot_body_name} (ID: {r_foot_body_id})")
    print(f"  Size: {r_foot_size}")
    print(f"  Current position: {mj_data.site(r_foot_site_id).xpos}")
    
    # Get r_toes_touch for comparison
    r_toes_site_id = mj_model.site("r_toes_touch")
    r_toes_body_id = mj_model.site(r_toes_site_id).bodyid
    r_toes_body_name = mj_model.body(r_toes_body_id).name
    r_toes_size = mj_model.site(r_toes_site_id).size
    
    print(f"\nr_toes_touch site:")
    print(f"  Body: {r_toes_body_name} (ID: {r_toes_body_id})")
    print(f"  Size: {r_toes_size}")
    print(f"  Current position: {mj_data.site(r_toes_site_id).xpos}")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
