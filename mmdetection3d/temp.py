import collections

def calculate_latencies(file_path):
    # Dictionary to store lists of latencies for each model
    data = collections.defaultdict(list)
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Swin' not in line and 'FlatFormer' not in line:
                    continue
                # Strip whitespace and split by space
                parts = line.strip().split()
                
                # We need at least a name and a number
                if len(parts) < 2:
                    continue
                
                model_name = parts[0]
                latency_str = parts[1]
                
                try:
                    # Attempt to convert the second part to a float
                    latency_val = float(latency_str)
                    
                    # Store data if it matches our target models
                    if "Swin" in model_name:
                        data["Swin"].append(latency_val)
                    elif "FlatFormer" in model_name:
                        data["FlatFormer"].append(latency_val)
                        
                except ValueError:
                    # This handles "junk" lines where the second part isn't a number
                    continue
                    
        # Calculate and print averages
        for model, latencies in data.items():
            if latencies:
                avg = sum(latencies) / len(latencies)
                print(f"{model} Average Latency: {avg:.10f} (Samples: {len(latencies)})")
            else:
                print(f"{model}: No valid data found.")
                
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

if __name__ == "__main__":
    # Change 'latencies.txt' to the name of your actual file
    calculate_latencies('backbone_timings_flash.txt')