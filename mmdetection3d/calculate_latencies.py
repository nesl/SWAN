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
        import pdb; pdb.set_trace()
        print("Error: The file was not found.")
    except ValueError:
        print("Error: Could not parse a value into a number.")

print(calculate_average('/workspace/mmdetection3d/EE_TEST_Latency_EE_camera_fog_16.txt'))