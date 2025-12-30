import mmengine

# 1. Load the original huge file
print("Loading original pickle...")
original_data = mmengine.load('data/nuscenes/nuscenes_infos_train.pkl')

# 2. Slice the data list (keep only 10 samples)
# Note: Structure differs slightly by version. 
# MMEngine usually puts samples in a 'data_list' key.
if 'data_list' in original_data:
    original_data['data_list'] = original_data['data_list'][:1000]
elif 'infos' in original_data:
    # Older MMDetection3D structure
    original_data['infos'] = original_data['infos'][:1000]
else:
    # List based structure
    original_data = original_data[:1000]

# 3. Save as a new debug file
print("Saving debug pickle...")
mmengine.dump(original_data, 'data/nuscenes/nuscenes_infos_debug.pkl')
print("Done! Created data/nuscenes/nuscenes_infos_debug.pkl")