from collections import defaultdict

def calculate_averages(file_path):
    totals = defaultdict(float)
    counts = defaultdict(int)

    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                # Split the line by commas to get each component segment
                parts = line.strip().split(',')
                for part in parts:
                    if ':' in part:
                        name, value = part.split(':')
                        name = name.strip()
                        try:
                            totals[name] += float(value.strip())
                            counts[name] += 1
                        except ValueError:
                            continue

        if not counts:
            print("No valid timing data found.")
            return

        print(f"{'Component':<20} | {'Average Latency (ms)':<20} | {'Samples':<10}")
        print("-" * 55)
        
        # Sort by component name for readability
        for name in sorted(totals.keys()):
            avg = totals[name] / counts[name]
            print(f"{name:<20} | {avg:<20.4f} | {counts[name]:<10}")

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")

if __name__ == "__main__":
    calculate_averages('component_timings_newscott_corrected_base.txt')