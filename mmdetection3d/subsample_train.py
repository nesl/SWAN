import pickle
import random
import numpy as np

# CONFIGURATION
# ---------------------------------------------------------
input_pkl = 'data/nuscenes/nuscenes_infos_train.pkl'
output_pkl = 'data/nuscenes/nuscenes_infos_train_40pct.pkl'
keep_ratio = 0.4
# ---------------------------------------------------------

print(f"Loading {input_pkl}...")
with open(input_pkl, 'rb') as f:
    data = pickle.load(f)

# Handle version differences (v1.x has 'data_list', v0.x is just a list)
if isinstance(data, dict) and 'data_list' in data:
    info_list = data['data_list']
    is_v1 = True
else:
    info_list = data
    is_v1 = False

print("Grouping frames into scenes based on timestamps...")

scenes = []
current_scene = []
prev_timestamp = -1

for i, info in enumerate(info_list):
    curr_timestamp = info['timestamp']  # Microseconds
    
    # Logic: 
    # 1. First frame always starts a scene.
    # 2. If time jumps backward, it's a new scene (different recording).
    # 3. If time jumps forward by > 1.5 seconds (standard gap is 0.5s), it's a new scene.
    if prev_timestamp != -1:
        time_diff_sec = (curr_timestamp - prev_timestamp) / 1e6
        
        # Threshold: 1.5s allows for small dropped packets, but catches scene changes
        if time_diff_sec < 0 or time_diff_sec > 1.5:
            scenes.append(current_scene)
            current_scene = []
    
    current_scene.append(info)
    prev_timestamp = curr_timestamp

# Don't forget the last scene
if current_scene:
    scenes.append(current_scene)

print(f"Detected {len(scenes)} distinct scenes.")

# Random Sampling
num_scenes_to_keep = int(len(scenes) * keep_ratio)
print(f"Sampling {num_scenes_to_keep} scenes ({keep_ratio*100}%)...")

selected_scenes = random.sample(scenes, num_scenes_to_keep)

# Flatten back into a single list
new_info_list = [frame for scene in selected_scenes for frame in scene]

print(f"Original frames: {len(info_list)}")
print(f"New frames: {len(new_info_list)}")

# Save
if is_v1:
    data['data_list'] = new_info_list
    save_data = data
else:
    save_data = new_info_list

with open(output_pkl, 'wb') as f:
    pickle.dump(save_data, f)

print(f"Saved to {output_pkl}")