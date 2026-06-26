import re

def calculate_average_usage(file_path):
    lidar_active_counts = []
    image_active_counts = []
    
    # Defaults for layer totals based on your sample
    lidar_total_layers = 8
    image_total_layers = 12

    # Regex to find the lists inside the tensor brackets
    # Matches: tensor([0, 1, ...], device=...)
    lidar_regex = re.compile(r"Controller LiDAR tensor\(\[(.*?)\]")
    image_regex = re.compile(r"Controller Image tensor\(\[(.*?)\]")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Check for LiDAR tensor lines
                lidar_match = lidar_regex.search(line)
                if lidar_match:
                    # Convert string "1, 0, 1..." to list of ints
                    values = [int(x.strip()) for x in lidar_match.group(1).split(',')]
                    lidar_active_counts.append(sum(values))
                    lidar_total_layers = len(values) # Update based on actual data

                # Check for Image tensor lines
                image_match = image_regex.search(line)
                if image_match:
                    values = [int(x.strip()) for x in image_match.group(1).split(',')]
                    image_active_counts.append(sum(values))
                    image_total_layers = len(values) # Update based on actual data

        if not lidar_active_counts and not image_active_counts:
            print("No data found in the file.")
            return

        # Calculate averages
        avg_lidar = sum(lidar_active_counts) / len(lidar_active_counts) if lidar_active_counts else 0
        avg_image = sum(image_active_counts) / len(image_active_counts) if image_active_counts else 0

        print(f"Analysis Results:")
        print(f"-----------------")
        print(f"Average LiDAR layer usage: {avg_lidar:.2f}/{lidar_total_layers} layers")
        print(f"Average Image layer usage: {avg_image:.2f}/{image_total_layers} layers")
        print(f"Total samples processed: {len(lidar_active_counts)}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

# Usage: Replace 'your_log_file.txt' with your actual filename
import os
path = '/workspace/mmdetection3d/work_dirs/'
for txt_file in sorted(os.listdir(path)):

    if txt_file[-3:] == 'txt' and 'ECCV_Controller_Universal' in txt_file:
        print(txt_file)
        calculate_average_usage(path + txt_file)
        print()