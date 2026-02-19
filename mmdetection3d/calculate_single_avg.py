def calculate_average(file_path):
    try:
        with open(file_path, 'r') as f:
            # Extract numbers: strip 'Elapsed: ', convert to float
            times = [float(line.split(':')[-1].strip()) for line in f if line.strip()]
        
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
calculate_average('full_model_latency.txt')