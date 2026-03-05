import os
import json
import argparse
import glob
import sys
import csv
import math

# Try to use pandas if available, else standard python
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def format_float(val, precision=4, use_exp=False):
    """Safe formatting of float values with handling for inf/None"""
    if val is None or val == '':
        return '-'
    try:
        f = float(val)
        if math.isinf(f):
            return str(f)
        if math.isnan(f):
            return '-'
        if use_exp:
            return f"{f:.{precision}e}"
        return f"{f:.{precision}f}"
    except (ValueError, TypeError):
        return str(val)

def parse_results_folder(folder_path):
    """
    Parses all JSON result files in the given folder and returns a list of dictionaries.
    """
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return None

    data = []
    
    # Try multiple patterns for json files
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        return None

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
            
            filename = os.path.basename(json_file)
            # Remove extension for parsing
            name_no_ext = filename.replace('.json', '')
            parts = name_no_ext.split('_')
            
            algo = "Unknown"
            subsolver = "Unknown"
            norm = "Unknown"
            
            # Robust parsing logic:
            # Expected format: gdpopt.ALGO_SUBSOLVER_NORM[_otherstuff].json
            # e.g. gdpopt.ldbd_conopt_L2.json
            
            # 1. Parse Algorithm
            if '.' in parts[0]:
                algo_part = parts[0].split('.')
                if len(algo_part) > 1:
                    algo = algo_part[1]  # 'ldbd' from 'gdpopt.ldbd'
                else:
                    algo = parts[0]
            else:
                 algo = parts[0]

            # 2. Parse Subsolver and Norm
            # Usually strict order: ALGO, SUBSOLVER, NORM
            if len(parts) >= 2:
                subsolver = parts[1]
            if len(parts) >= 3:
                norm = parts[2]
            
            # 3. Detect "mode_transfer"
            is_transfer = "mode_transfer" in filename
            model_type = "Transfer" if is_transfer else "Standard"

            problem = content.get('Problem', [{}])[0]
            solver = content.get('Solver', [{}])[0]
            
            # Extract objective value and bounds
            # For minimization:
            # Upper bound = Best Integer (Incumbent)
            # Lower bound = Best Relaxation
            
            ub_val_raw = problem.get('Upper bound', None)
            lb_val_raw = problem.get('Lower bound', None)
            
            ub = None
            lb = None
            
            try:
                if ub_val_raw is not None:
                     ub = float(ub_val_raw)
            except: pass
            
            try:
                if lb_val_raw is not None:
                     lb = float(lb_val_raw)
            except: pass
            
            # Calculate Gap %
            # Gap = |UB - LB| / max(|UB|, 1e-10) * 100
            
            gap_str = "-"
            if ub is not None and lb is not None:
                # Handle infinities
                if math.isinf(ub) or math.isinf(lb):
                     if ub == lb: # consistent infinity
                         gap_str = "0.00%"
                     else:
                         gap_str = "inf"
                else:
                    # Check for "optimal" status where gap might be practically 0 even if numerically slight
                    # But standard formula:
                    denominator = max(abs(ub), 1e-10)
                    gap_val = abs(ub - lb) / denominator * 100.0
                    gap_str = f"{gap_val:.2f}%"

            # Extract other metadata
            iters = solver.get('Iterations', '-')
            time_val = solver.get('Wallclock time', '')
            status = solver.get('Status', '-')
            term_cond = solver.get('Termination condition', '-')
            
            row = {
                'Algorithm': algo,
                'Subsolver': subsolver,
                'Norm': norm,
                'Model': model_type,
                'Lower Bound': format_float(lb),
                'Upper Bound': format_float(ub),
                'Gap (%)': gap_str,
                'Time (s)': format_float(time_val, 2),
                'Iterations': str(iters),
                'Status': status,
                'Termination': term_cond,
                'File': filename
            }
            data.append(row)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

    if not data:
        return None
        
    # Sort data by Algorithm, Subsolver, Norm
    # Using tuple sort (primary key first)
    data.sort(key=lambda x: (x.get('Algorithm',''), x.get('Subsolver',''), x.get('Norm',''), x.get('Model', '')))
    
    return data

def save_csv(data, filename):
    if not data:
        return

    # Define requested columns order
    headers = [
        'Algorithm', 
        'Subsolver', 
        'Norm', 
        'Model',
        'Lower Bound', 
        'Upper Bound', 
        'Gap (%)', 
        'Time (s)', 
        'Iterations', 
        'Status', 
        'Termination'
    ]
    
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        print(f"  -> Saved summary to: {filename}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def process_date_folder(date_folder_path):
    """
    Processes a specific date folder: generates summary table and saves CSV inside it.
    """
    print(f"Processing folder: {date_folder_path}")
    data = parse_results_folder(date_folder_path)
    if data:
        # Save CSV inside the folder
        csv_filename = "summary_results.csv"
        csv_path = os.path.join(date_folder_path, csv_filename)
        save_csv(data, csv_path)
        
        # Also print a preview if single folder requested
        # Simple ASCII table print - only print first few rows or all if small
        print_ascii_table(data)
        
        return data
    else:
        print(f"  -> No results found or empty.")
        return None

def print_ascii_table(data):
    if not data: return
    headers = ['Algorithm', 'Subsolver', 'Norm', 'Model', 'Lower Bound', 'Upper Bound', 'Gap (%)', 'Time (s)', 'Status']
    
    # Calculate widths
    widths = {h: len(h) for h in headers}
    for row in data:
        for h in headers:
            val = str(row.get(h, ''))
            widths[h] = max(widths[h], len(val))
    
    header_str = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    print("\n" + header_str)
    print("-" * len(header_str))
    
    for row in data:
        print(" | ".join(f"{str(row.get(h, '')):<{widths[h]}}" for h in headers))
    print("\n")


def process_stage_folder(stage_folder_path):
    """
    Processes all date folders within a stage folder (assuming structure stage/nfe30/date/).
    """
    nfe_path = os.path.join(stage_folder_path, "nfe30")
    
    # Check if dates are directly under stage folder or under nfe30
    target_path = nfe_path if os.path.isdir(nfe_path) else stage_folder_path
    
    if not os.path.exists(target_path):
        print(f"Skipping {stage_folder_path}: path not valid.")
        return

    # List date folders (assume folder starts with 20..)
    candidates = [f for f in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, f))]
    date_folders = [f for f in candidates if f.startswith('20')]
    
    if not date_folders:
         print(f"No date folders found in {target_path}")
         return

    print(f"Found {len(date_folders)} date folders in {stage_folder_path}")
    for date_folder in date_folders:
        full_path = os.path.join(target_path, date_folder)
        process_date_folder(full_path)

def main():
    parser = argparse.ArgumentParser(description="Generate summary tables for simulation results.")
    parser.add_argument("--stage", help="Specific stage folder name (e.g., 'four_stage_dynamic_model_switching_nonlinear')", default=None)
    parser.add_argument("--date", help="Specific date folder name (e.g., '2026-03-04_17-44-27')", default=None)
    parser.add_argument("--folder", help="Full path to a specific folder to process.", default=None)
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, "results")

    if args.folder:
        process_date_folder(args.folder)
        
    elif args.stage and args.date:
        # Try to resolve stage path
        stage_path = os.path.join(results_base_dir, args.stage)
        if not os.path.exists(stage_path):
             # fuzzy match
             candidates = glob.glob(os.path.join(results_base_dir, f"*{args.stage}*"))
             if candidates:
                 stage_path = candidates[0]
             else:
                 print(f"Stage folder '{args.stage}' not found in {results_base_dir}")
                 return
        
        # Check nfe30 structure first
        date_path = os.path.join(stage_path, "nfe30", args.date)
        if not os.path.exists(date_path):
             # Try direct
             date_path = os.path.join(stage_path, args.date)
        
        if os.path.exists(date_path):
            process_date_folder(date_path)
        else:
            print(f"Date folder '{args.date}' not found in {stage_path}")

    elif args.stage:
        stage_path = os.path.join(results_base_dir, args.stage)
        if not os.path.exists(stage_path):
             candidates = glob.glob(os.path.join(results_base_dir, f"*{args.stage}*"))
             if candidates:
                 stage_path = candidates[0]
             else:
                 print(f"Stage folder '{args.stage}' not found in {results_base_dir}")
                 return
        process_stage_folder(stage_path)
        
    else:
        # Batch: process everything
        print(f"Scanning results directory: {results_base_dir}")
        if not os.path.exists(results_base_dir):
            print(f"Results directory not found.")
            return

        stage_folders = [f for f in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, f))]
        
        for stage in stage_folders:
            stage_full_path = os.path.join(results_base_dir, stage)
            process_stage_folder(stage_full_path)

if __name__ == "__main__":
    main()
