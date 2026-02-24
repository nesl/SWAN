import os
def calculate_average(file_path):
    try:
        with open(file_path, 'r') as f:
            times = []
            for line in f:
                if line.strip() and line.strip()[:8] == 'Elapsed:':
                    times.append(float(line.split(':')[-1].strip()))
            # Extract numbers: strip 'Elapsed: ', convert to float
        times = times[1:]
        if not times:
            return "No data found."

        avg = sum(times) / len(times)
        
        print(f"Total Entries: {len(times)}")
        print(f"Average Time:  {avg:.4f}")
        print(f"Min Time:      {min(times):.4f}")
        print(f"Max Time:      {max(times):.4f}")

    except FileNotFoundError:
        print("Error: The file was not found.")
    except ValueError:
        print("Error: Could not parse a value into a number.")

# Run the function
for file in sorted(os.listdir('No_Flash_Latencies')):
    print(file)
    calculate_average(f'No_Flash_Latencies/{file}')