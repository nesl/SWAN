import re
import ast

def calculate_average_utilization(file_path):
    lidar_sample_totals = []
    image_sample_totals = []

    # State variables for the current sample
    current_lidar_sum = 0
    current_image_sum = 0

    # Compiled regex for speed
    lidar_pattern = re.compile(r"Lidar Decision List (\[.*?\])")
    image_pattern = re.compile(r"Image Early Exit Decisions (\[.*?\])")
    label_pattern = re.compile(r"Gt_corruption_labels")

    with open(file_path, 'r') as f:
        for line in f:
            # 1. Check for Lidar decisions in this sample
            lidar_match = lidar_pattern.search(line)
            if lidar_match:
                # ast.literal_eval safely converts the string "[1.0, 0.0...]" to a list
                current_lidar_sum += sum(ast.literal_eval(lidar_match.group(1)))

            # 2. Check for Image decisions in this sample
            image_match = image_pattern.search(line)
            if image_match:
                current_image_sum += sum(ast.literal_eval(image_match.group(1)))

            # 3. Check for the end-of-sample marker
            if label_pattern.search(line):
                lidar_sample_totals.append(current_lidar_sum)
                image_sample_totals.append(current_image_sum)
                
                # Reset counters for the next sample
                current_lidar_sum = 0
                current_image_sum = 0

    # Calculate final averages
    num_samples = len(lidar_sample_totals)
    if num_samples == 0:
        return 0, 0, 0

    avg_lidar = sum(lidar_sample_totals) / num_samples
    avg_image = sum(image_sample_totals) / num_samples

    return avg_lidar, avg_image, num_samples

# Usage:
import os
root_dir = '/workspace/mmdetection3d/Table_1_Results/Swan-SC/Accuracy'
for path in sorted(os.listdir(root_dir)):
    file_path = os.path.join(root_dir, path)
    if file_path[-4:] != '.txt':
        continue
    avg_l, avg_i, total = calculate_average_utilization(file_path)
    print(path)
    print(f"\tProcessed {total} samples.")
    print(f"\tAverage Lidar Utilization: {avg_l:.2f} / 8.0")
    print(f"\tAverage Image Utilization: {avg_i:.2f} / 12.0")

#print(calculate_average_utilization('/workspace/mmdetection3d/work_dirs/EE_Universal_test_8_lidar_motionblur.txt'))