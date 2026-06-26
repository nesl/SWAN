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
        # print(f"\tMedian Time:  {avg:.2f}")
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
# The order of corruptions as they appear in your spreadsheet columns
corruptions = ['beamsreducing', 'camera_fog', 'camera_motionblur', 'dark', 'lidar_motionblur']

# The order of methods as they appear in each block
methods = ['Naive', 'SWAN-C', 'SWAN-SC', 'SWAN-PSC', 'ADMN']

# The order of budget blocks (from top to bottom)
budgets = [16, 8, 6, 4]

translation = {
    'Naive': 'naive',
    'SWAN-C': 'controller',
    'SWAN-SC': 'EE',
    'SWAN-PSC': 'Pruning',
    'ADMN': 'ADMN'
}

# 1. Fetch and store the data first so we can iterate in the right order
data_map = {}
for key in translation:
    data_map[key] = {}
    for corr in corruptions:
        data_map[key][corr] = {}
        for budget in budgets:
            avg = calculate_average(f'{key}/No_Flash_Latencies/Latency_{translation[key]}_{corr}_{budget}.txt') 
            data_map[key][corr][budget] = avg

for budget in budgets:
    for method in methods:
        # row_str = f"{method} {budget:<5}" # Standardize label width
        row_str = ''
        for corr in corruptions:
            val = data_map[method][corr][budget]
            # Use fixed-width formatting instead of multiple tabs
            # This pushes the value 16 spaces to the right to hit the NF column
            row_str += f"{val:>16.2f}" 
        print(row_str)
