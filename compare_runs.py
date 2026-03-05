import os
import json
import argparse
import re
from pathlib import Path
import sys

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
# Add your default comparisons here.
# Each entry is a tuple: (results_folder_path, date1, date2)
DEFAULT_COMPARISONS = [
    # Example:
    # ('results/three_stage_dynamic_model_switching/nfe30', '2026-03-03_13-40-33', '2024-03-07_11-29-45'),
    # ('results/five_stage_dynamic_model_switching_nonlinear/nfe30', '2026-03-03_15-57-47', '2024-02-26_00-17-16'),
]
# ==========================================

def load_result(path):
    """
    Load relevant data from a result JSON file.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Robustly handle list or dict wrapper if present, though example showed direct list under "Problem"
        prob_list = data.get("Problem", [])
        prob = prob_list[0] if prob_list else {}
        
        solver_list = data.get("Solver", [])
        solver = solver_list[0] if solver_list else {}
        
        lb = prob.get("Lower bound")
        ub = prob.get("Upper bound")
        status = solver.get("Termination condition")
        time = solver.get("Wallclock time")
        
        return {"lb": lb, "ub": ub, "status": status, "time": time}
    except Exception as e:
        # print(f"Error loading {path}: {e}")
        return None

def fmt(val):
    if val is None:
        return "None"
    if isinstance(val, (int, float)):
        return f"{val:.6f}"
    return str(val)

def compare_dates(folder_path, date1, date2):
    path1 = folder_path / date1
    path2 = folder_path / date2

    if not path1.exists():
        print(f"Error: Path {path1} does not exist.")
        return
    if not path2.exists():
        print(f"Error: Path {path2} does not exist.")
        return

    files1 = set(f for f in os.listdir(path1) if f.endswith('.json'))
    files2 = set(f for f in os.listdir(path2) if f.endswith('.json'))
    
    common_files = sorted(list(files1.intersection(files2)))
    
    print("\n" + "="*120)
    print(f"COMPARISON 1: {date1} vs {date2}".center(120))
    print("="*120)
    
    header = f"{'Algorithm File':<40} | {'UB (Date1)':<15} | {'UB (Date2)':<15} | {'Diff UB':<12} | {'LB (Date1)':<15} | {'LB (Date2)':<15}"
    print(header)
    print("-" * len(header))

    for fname in common_files:
        res1 = load_result(path1 / fname)
        res2 = load_result(path2 / fname)
        
        if res1 and res2:
            ub1 = res1.get('ub')
            ub2 = res2.get('ub')
            lb1 = res1.get('lb')
            lb2 = res2.get('lb')

            diff_ub = "N/A"
            if isinstance(ub1, (int, float)) and isinstance(ub2, (int, float)):
                diff_ub = f"{ub1 - ub2:.6f}"
            
            print(f"{fname:<40} | {fmt(ub1):<15} | {fmt(ub2):<15} | {diff_ub:<12} | {fmt(lb1):<15} | {fmt(lb2):<15}")

def compare_algos_in_folder(folder_path, date_label):
    if not folder_path.exists():
        return

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    groups = {} 
    
    # Identify pairs of ldbd and ldsda
    for f in files:
        # Match pattern like gdpopt.ldbd_conopt_L2.json vs gdpopt.ldsda_conopt_L2.json
        # Regex to capture the part after ldbd_ or ldsda_
        # Pattern assumes format: gdpopt.(ldbd|ldsda)_(rest_of_params).json
        m = re.match(r'gdpopt\.(ldbd|ldsda)_(.+)\.json', f)
        if m:
            algo_type = m.group(1) # ldbd or ldsda
            suffix = m.group(2)    # params
            if suffix not in groups:
                groups[suffix] = {}
            groups[suffix][algo_type] = f

    print("\n" + "="*120)
    print(f"COMPARISON 2: LDBD vs LDSDA within {date_label}".center(120))
    print("="*120)
    header = f"{'Parameters':<35} | {'LDBD UB':<15} | {'LDSDA UB':<15} | {'Diff UB':<12} | {'LDBD Time':<12} | {'LDSDA Time':<12}"
    print(header)
    print("-" * len(header))
    
    found_any = False
    for suffix, algos in groups.items():
        if 'ldbd' in algos and 'ldsda' in algos:
            found_any = True
            res_ldbd = load_result(folder_path / algos['ldbd'])
            res_ldsda = load_result(folder_path / algos['ldsda'])
            
            if res_ldbd and res_ldsda:
                ub_ldbd = res_ldbd.get('ub')
                ub_ldsda = res_ldsda.get('ub')
                t_ldbd = res_ldbd.get('time')
                t_ldsda = res_ldsda.get('time')

                diff = "N/A"
                if isinstance(ub_ldbd, (int, float)) and isinstance(ub_ldsda, (int, float)):
                    diff = f"{ub_ldbd - ub_ldsda:.6f}"
                    
                print(f"{suffix:<35} | {fmt(ub_ldbd):<15} | {fmt(ub_ldsda):<15} | {diff:<12} | {fmt(t_ldbd):<12} | {fmt(t_ldsda):<12}")
    
    if not found_any:
        print("No matching LDBD/LDSDA pairs found.")

def process_comparison(results_folder_str, date1, date2):
    print(f"\n{'='*120}")
    print(f"PROCESSING: {results_folder_str}".center(120))
    print(f"DATE 1: {date1} | DATE 2: {date2}".center(120))
    print(f"{'='*120}")

    folder_path = Path(results_folder_str)
    
    if not folder_path.exists():
         # Try resolving relative to current working dir if absolute fails
         folder_path = Path.cwd() / results_folder_str
         if not folder_path.exists():
             print(f"Error: Could not find folder {results_folder_str}")
             print(f"Current working directory: {Path.cwd()}")
             return

    # 1. Compare same files across dates
    compare_dates(folder_path, date1, date2)

    # 2. Compare LDBD vs LDSDA for date 1
    compare_algos_in_folder(folder_path / date1, date1)
    
    # 2. Compare LDBD vs LDSDA for date 2
    compare_algos_in_folder(folder_path / date2, date2)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="""
Compare run results for the same stage across two dates.
Supports batch processing of multiple comparisons.

Usage:
  1. Interactive Mode: Run without arguments.
     python compare_runs.py

  2. Single Comparison:
     python compare_runs.py <folder> <date1> <date2>

  3. Batch Mode (Multiple comparisons):
     python compare_runs.py <folder1> <date1_a> <date1_b> <folder2> <date2_a> <date2_b> ...
""")
    parser.add_argument("args", nargs='*', help="Triplets of (results_folder, date1, date2)")
    
    args = parser.parse_args()
    raw_args = args.args

    if not raw_args:
        # Check defaults first
        if DEFAULT_COMPARISONS:
            print("\n" + "="*80)
            print("No command-line arguments provided.".center(80))
            print("Using DEFAULT_COMPARISONS defined in the script.".center(80))
            print("="*80 + "\n")
            
            for folder, d1, d2 in DEFAULT_COMPARISONS:
               process_comparison(folder, d1, d2)
            return

        # Interactive mode if no args and no defaults
        print("Enter the path to the results folder (containing the date folders):")
        results_folder_str = input("> ").strip()
        print("Enter the first date folder name:")
        date1 = input("> ").strip()
        print("Enter the second date folder name:")
        date2 = input("> ").strip()
        process_comparison(results_folder_str, date1, date2)
    else:
        if len(raw_args) % 3 != 0:
            print("Error: Arguments must be provided in triplets: <folder> <date1> <date2>")
            print(f"You provided {len(raw_args)} arguments: {raw_args}")
            return
        
        # Process each triplet
        for i in range(0, len(raw_args), 3):
            folder = raw_args[i]
            d1 = raw_args[i+1]
            d2 = raw_args[i+2]
            process_comparison(folder, d1, d2)

if __name__ == "__main__":
    main()
