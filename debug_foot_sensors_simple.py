#!/usr/bin/env python3
"""
Lightweight debug script to investigate foot sensor zeros.
XML-based analysis only - no environment loading.
"""

import sys
import json
import os
import xml.etree.ElementTree as ET

if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = input("Enter the log directory: ")

print("=" * 80)
print("FOOT SENSOR DEBUG - XML ANALYSIS")
print("=" * 80)

# Load config to find model file
from rl_train.utils.data_types import DictionableDataclass
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
    config_dict = json.load(f)
config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

model_file = config.env_params.model_path
print(f"Model file: {model_file}")

# Parse the model XML
try:
    tree = ET.parse(model_file)
    root = tree.getroot()
    
    print("\n" + "=" * 80)
    print("SENSOR DEFINITIONS IN XML")
    print("=" * 80)
    
    # Find all sensors
    sensor_section = root.find('.//sensor')
    if sensor_section is not None:
        for sensor in sensor_section:
            tag = sensor.tag
            name = sensor.get('name', 'N/A')
            site = sensor.get('site', 'N/A')
            print(f"\n{tag}: {name}")
            if site != 'N/A':
                print(f"  Site: {site}")
            for attr in sensor.attrib:
                if attr not in ['name', 'site']:
                    print(f"  {attr}: {sensor.get(attr)}")
    
    print("\n" + "=" * 80)
    print("TOUCH SENSOR SITES CONFIGURATION")
    print("=" * 80)
    
    # Find all sites related to touch sensors
    touch_sites = ['r_foot_touch', 'l_foot_touch', 'r_toes_touch', 'l_toes_touch']
    for site_name in touch_sites:
        sites = root.findall(f".//site[@name='{site_name}']")
        if sites:
            site = sites[0]
            print(f"\n{site_name}:")
            for attr in site.attrib:
                print(f"  {attr}: {site.get(attr)}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS & COMPARISON")
    print("=" * 80)
    
    # Compare r_foot and r_toes
    r_foot_sites = root.findall(".//site[@name='r_foot_touch']")
    r_toes_sites = root.findall(".//site[@name='r_toes_touch']")
    
    if r_foot_sites and r_toes_sites:
        r_foot = r_foot_sites[0]
        r_toes = r_toes_sites[0]
        
        print("\nComparison of r_foot_touch vs r_toes_touch:")
        print(f"\nr_foot_touch:")
        print(f"  Position: {r_foot.get('pos')}")
        print(f"  Size: {r_foot.get('size')}")
        print(f"  Type: {r_foot.get('type')}")
        print(f"  Euler: {r_foot.get('euler')}")
        
        print(f"\nr_toes_touch:")
        print(f"  Position: {r_toes.get('pos')}")
        print(f"  Size: {r_toes.get('size')}")
        print(f"  Type: {r_toes.get('type')}")
        print(f"  Euler: {r_toes.get('euler')}")
        
        print(f"\nKEY DIFFERENCES:")
        print(f"  Size: r_foot={r_foot.get('size')} vs r_toes={r_toes.get('size')}")
        print(f"  Rotation: r_foot euler={r_foot.get('euler')} vs r_toes euler={r_toes.get('euler')}")

except Exception as e:
    print(f"Error parsing XML: {e}")

print("\n" + "=" * 80)
print("JSON DATA VERIFICATION")
print("=" * 80)

# Verify the gait_evaluated_data shows zeros
gait_data_path = os.path.join(log_dir, "analyze_results_00", "gait_evaluated_data.json")
if os.path.exists(gait_data_path):
    with open(gait_data_path, 'r') as f:
        data = json.load(f)
    
    if "sensor_data" in data["series_data"]:
        sensors = data["series_data"]["sensor_data"]
        print(f"\nSensor data summary from gait_evaluated_data.json:")
        for sensor_name in ['r_foot', 'l_foot', 'r_toes', 'l_toes']:
            if sensor_name in sensors:
                values = sensors[sensor_name]["data"]
                non_zero = sum(1 for v in values if v[0] > 0.1)
                print(f"  {sensor_name}: {len(values)} entries, {non_zero} non-zero (>0.1)")
else:
    print(f"\nNo gait_evaluated_data.json found at {gait_data_path}")

print("\n" + "=" * 80)
