import re

def calculate_mask_averages(file_path):
    points_mask_values = []
    img_mask_values = []

    # Regular expression patterns to match the tensor values
    # Matches: Points_Mask tensor(0.3052, ...
    points_pattern = re.compile(r"Points_Mask tensor\(([\d\.]+)")
    # Matches: Img_Mask tensor(0.0667, ...
    img_pattern = re.compile(r"Img_Mask tensor\(([\d\.]+)")

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Find Points_Mask values
                points_match = points_pattern.search(line)
                if points_match:
                    points_mask_values.append(float(points_match.group(1)))

                # Find Img_Mask values
                img_match = img_pattern.search(line)
                if img_match:
                    img_mask_values.append(float(img_match.group(1)))

        # Calculate averages
        if points_mask_values:
            avg_points = sum(points_mask_values) / len(points_mask_values)
            print(f"Average Points_Mask ({len(points_mask_values)} entries): {avg_points:.6f}")
        else:
            print("No Points_Mask values found.")

        if img_mask_values:
            avg_img = sum(img_mask_values) / len(img_mask_values)
            print(f"Average Img_Mask ({len(img_mask_values)} entries): {avg_img:.6f}")
        else:
            print("No Img_Mask values found.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return avg_img, avg_points

if __name__ == "__main__":
    # Replace 'your_log_file.txt' with the actual path to your file
    img_list, pts_list = [], []
    for budget in ['4', '6', '8', '16']:
        for corruption in ['beamsreducing', 'camera_fog', 'camera_motionblur', 'dark', 'lidar_motionblur']:
            avg_img, avg_points = calculate_mask_averages(f'/workspace/mmdetection3d/Table_1_Results/SWAN-PSC/Accuracy/Pruner_EE_Universal_test_{budget}_{corruption}.txt')
            img_list.append(avg_img)
            pts_list.append(avg_points)
    print(sum(img_list) / len(img_list))
    print(sum(pts_list) / len(pts_list))