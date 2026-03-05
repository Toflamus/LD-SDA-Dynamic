import os
import json
import argparse
import glob
import sys

def parse_results(date_str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "results/three_stage_dynamic_model_switching/nfe30")
    target_dir = os.path.join(base_dir, date_str)
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return None

    data = []
    
    json_files = glob.glob(os.path.join(target_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {target_dir}")
        return None

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
            
            filename = os.path.basename(json_file)
            parts = filename.replace('.json', '').split('_')
            
            algo = "Unknown"
            subsolver = "Unknown"
            norm = "Unknown"
            
            # Simple heuristic for parsing filename
            # e.g. gdpopt.ldbd_conopt_L2.json
            if len(parts) >= 1:
                p0 = parts[0]
                if '.' in p0:
                    algo = p0.split('.')[1]
                else:
                    algo = p0
            
            if len(parts) >= 2:
                subsolver = parts[1]
            if len(parts) >= 3:
                norm = parts[2]

            problem = content.get('Problem', [{}])[0]
            solver = content.get('Solver', [{}])[0]
            
            # Extract objective value
            # Usually 'Upper bound' is the objective for minimization
            obj_val = problem.get('Upper bound', '')
            lb = problem.get('Lower bound', '')
            
            # Extract iterations
            iters = solver.get('Iterations', '')
            
            # Extract time
            time = solver.get('Wallclock time', '')
            
            # Extract status
            status = solver.get('Status', '')
            
            # Extract termination condition
            term = solver.get('Termination condition', '')

            row = {
                'File': filename,
                'Algorithm': algo,
                'Subsolver': subsolver,
                'Norm': norm,
                'Lower Bound': lb,
                'Objective': obj_val,
                'Status': status,
                'Time (s)': time,
                'Iterations': iters,
                'Termination': term
            }
            data.append(row)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

    # Sort data by Algorithm, Subsolver, Norm
    # Convert numeric types for sorting if possible, otherwise string sort
    # We can just sort by string representation
    data.sort(key=lambda x: (x['Algorithm'], x['Subsolver'], x['Norm']))
    
    return data

def print_table(data):
    if not data:
        return
    
    # Headers to display
    headers = ['Algorithm', 'Subsolver', 'Norm', 'Lower Bound', 'Objective', 'Time (s)', 'Iterations', 'Status', 'Termination']
    # Map data keys to headers
    key_map = {
        'Algorithm': 'Algorithm',
        'Subsolver': 'Subsolver',
        'Norm': 'Norm',
        'Lower Bound': 'Lower Bound',
        'Objective': 'Objective',
        'Time (s)': 'Time (s)',
        'Iterations': 'Iterations',
        'Status': 'Status',
        'Termination': 'Termination'
    }

    # Calculate column widths
    widths = {h: len(h) for h in headers}
    for row in data:
        for h in headers:
            val = str(row.get(key_map[h], ''))
            # Format floats
            if isinstance(row.get(key_map[h]), (int, float)):
                 val = f"{row.get(key_map[h]):.4f}"
            widths[h] = max(widths[h], len(val))
    
    # Print header
    header_row = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    print(header_row)
    print("-" * len(header_row))
    
    # Print rows
    for row in data:
        cols = []
        for h in headers:
            val = row.get(key_map[h], '')
            # Format specific columns if they are numbers
            if h in ['Lower Bound', 'Objective', 'Time (s)']:
                try:
                    f = float(val)
                    if f == float('inf') or f == float('-inf'):
                        val_str = str(f)
                    else:
                        val_str = f"{f:.4f}"
                except (ValueError, TypeError):
                    val_str = str(val)
            else:
                val_str = str(val)
            
            cols.append(f"{val_str:<{widths[h]}}")
        print(" | ".join(cols))

def save_csv(data, filename):
    if not data:
        return
    
    # All keys in data[0]
    if not data:
        return
    keys = list(data[0].keys())
    
    try:
        with open(filename, 'w') as f:
            # write header
            f.write(",".join(keys) + "\n")
            for row in data:
                vals = [str(row.get(k, '')) for k in keys]
                f.write(",".join(vals) + "\n")
        print(f"\nSaved to {filename}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate table from simulation results.")
    parser.add_argument("date", help="Date string for the simulation run (folder name under nfe30)", nargs='?')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "results/three_stage_dynamic_model_switching/nfe30")

    if args.date:
        data = parse_results(args.date)
        if data:
             print_table(data)
             # Also save csv 
             save_csv(data, os.path.join(base_dir, f"summary_{args.date}.csv"))
    else:
        # If no date provided, list available dates
        if os.path.exists(base_dir):
            dates = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
            print("Available dates:")
            for d in dates:
                print(d)
            print("\nPlease provide a date as argument.")
        else:
             print(f"Directory not found: {base_dir}")
