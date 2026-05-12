import json
import numpy as np
import torch
from stable_baselines3 import PPO
from simulation.environment import DebrisEnv

def test_on_real_data(json_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading real-world data from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    
    # Take the first 8 objects to match our environment target count
    test_objects = data[:8]
    
    # Initialize environment
    env = DebrisEnv(num_targets=8)
    obs, _ = env.reset()
    
    # MANUALLY OVERRIDE DEBRIS ORBITS with real data
    for idx, obj in enumerate(test_objects):
        env.debris_orbits[idx] = [
            7000 + (14 - obj.get('MEAN_MOTION', 14)) * 500,
            obj.get('ECCENTRICITY', 0),
            obj.get('INCLINATION', 0),
            obj.get('RA_OF_ASC_NODE', 0),
            obj.get('ARG_OF_PERICENTER', 0),
            obj.get('MEAN_ANOMALY', 0)
        ]
    
    print(f"Loading PPO v4 Model from {model_path}...")
    try:
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n🚀 Starting Real-World Mission Scenario (Data Validation)...")
    print("-" * 60)
    
    total_reward = 0
    done = False
    truncated = False
    steps = 0
    cleared = 0
    collected_names = []
    
    while not (done or truncated) and steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if reward > 5: # Successful collection reward threshold
            cleared += 1
            obj_name = test_objects[action]['OBJECT_NAME'] if action < len(test_objects) else "Unknown"
            if obj_name not in collected_names:
                collected_names.append(obj_name)
                print(f"Step {steps:02d}: ✅ SUCCESS! Captured: {obj_name}")
            
    print("-" * 60)
    print(f"MISSION REPORT (FINAL VALIDATION):")
    print(f"  Total Steps taken: {steps}")
    print(f"  Objects Cleared:   {cleared}/{len(test_objects)}")
    print(f"  Efficiency Score:  {total_reward:.2f}")
    print(f"  Fuel Remaining:    {env.fuel_remaining:.2f} m/s")
    print(f"  Status:            {'MISSION ACCOMPLISHED' if cleared > 0 else 'MISSION FAILED'}")
    print("-" * 60)

if __name__ == "__main__":
    test_on_real_data("Data_For_Test.json", "results/models/ppo_debris_v4.zip")
