import os
import numpy as np
def calculate_average(file_path):
    try:
        with open(file_path, 'r') as f:
            times = []
            for line in f:
                if line.strip() and line.strip()[:8] == 'Elapsed:':
                    times.append(float(line.split(':')[-1].strip()) * 1000)
            # Extract numbers: strip 'Elapsed: ', convert to float
        times = times[50:]
        if not times:
            return "No data found."

        #avg = sum(times)/len(times)
        avg = sorted(times)[len(times)//2]
        
        # print(f"\tTotal Entries: {len(times)}")
        # print(f"\tAverage Time:  {avg:.2f}")
        # print(f"\tMin Time:      {min(times):.2f}")
        # print(f"\tMax Time:      {max(times):.2f}")
        # print(f"\tStd Time:     {np.std(times)}")
        return avg
    except FileNotFoundError:
        print("Error: The file was not found.")
    except ValueError:
        print("Error: Could not parse a value into a number.")

    
# # Run the function
# for file in sorted(os.listdir('Latencies_Flash_Attn')):
#     print(file)
#     calculate_average(f'Latencies_Flash_Attn/{file}')

# for corr in ['camera_fog', 'lidar_motionblur', 'beamsreducing']:
#     for budget in [4, 6, 8, 16]:
#         diff = calculate_average(f'Latency_EE_{corr}_{budget}.txt') - calculate_average(f'Latency_controller_{corr}_{budget}.txt')
#         print(f'{corr} \t {budget} \t {diff}')

print(calculate_average('/workspace/mmdetection3d/Results_Latencies/Old_Latencies_Flash_Attn/Latency_controller_lidar_motionblur_16.txt'))