import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import json
import seaborn as sns
import numpy as np
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.labelsize'] = 12  # For x and y labels
plt.rcParams['xtick.labelsize'] = 12  # For x tick labels
plt.rcParams['ytick.labelsize'] = 12  # For y tick labels

json_file_folder_dict = {
    4: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/four_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-44-27',
    5: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/five_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-45-05',
    6: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/six_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-46-07',
    7: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/seven_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-54-27',
    8: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/eight_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-54-51',
    9: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/nine_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-55-10',
}


def read_data(file_folder):
    data = []
    for f_name in glob(file_folder + '/*.json'):
        with open(f_name, 'r') as f:
            json_data = json.load(f)
            base_name = os.path.basename(f_name)[:-5]
            parts = base_name.split('_')
            strategy = parts[0] if parts else '-'
            solver = parts[1] if len(parts) > 1 else '-'
            
            is_transfer = 'mode_transfer' in base_name

            if strategy in ('gdpopt.ldsda', 'gdpopt.ldbd') and len(parts) > 2:
                # strategy + '-' + L2/Linf
                s_name = strategy + '-' + parts[2]
            else:
                s_name = strategy
            
            if is_transfer:
                s_name += '-Transfer'
            
            json_data['strategy'] = s_name
            json_data['solver'] = solver
            if isinstance(json_data['Problem'], list):
                json_data['Problem'] = json_data['Problem'][0]
            if isinstance(json_data['Solver'], list):
                json_data['Solver'] = json_data['Solver'][0]
            data.append(json_data)
    df = pd.json_normalize(data)
    if df.empty:
        return pd.DataFrame(columns=[
            'strategy', 'solver', 'Lower bound', 'Upper bound', 'Time', 'Termination condition'
        ])

    result = df[
        [
            'strategy',
            'solver',
            'Problem.Lower bound',
            'Problem.Upper bound',
            'Solver.User time',
            'Solver.Termination condition',
        ]
    ]
    result = result.fillna('-')
    result.rename(
        columns={
            'Problem.Lower bound': 'Lower bound',
            'Problem.Upper bound': 'Upper bound',
            'Solver.User time': 'Time',
            'Solver.Termination condition': 'Termination condition',
        },
        inplace=True,
    )
    return result


optimal_objective_value = {
    4: -23.3046896, # Updated from -23.304689 based on result file ????????
    5: -52.79223,
    6: -122.423721,
    7: -160.08149,
    8: -188.449493,
    9: -212.00879,
}

result_list = []
for stage in json_file_folder_dict:
    result = read_data(json_file_folder_dict[stage])
    result['Stage'] = stage
    result['Optimal objective value'] = optimal_objective_value[stage]
    result_list.append(result)

result = pd.concat(result_list).reset_index(drop=True)

time_limit = {4: 900, 5: 900, 6: 900, 7: 1800, 8: 1800, 9: 3600}
for stage in time_limit:
    temp_result = result[
        (result['Time'] >= time_limit[stage])
        & (result['Stage'] == stage)
        & (
            abs(result['Upper bound'] - result['Optimal objective value'])
            < abs(result['Optimal objective value']) * 0.001
        )
    ]
    step = 0.1
    if len(temp_result) > 1:
        for idx, index in enumerate(temp_result.index):
            result.iloc[index, result.columns.get_loc('Stage')] += step * (
                -len(temp_result) + 1 + 2 * idx
            )

strategy_maker_dict = {
    'gdp.bigm': "X",
    'gdp.hull': "^",
    'gdpopt.enumerate': "D",
    'gdpopt.loa': "*",
    'gdpopt.gloa': "P",
    'gdpopt.ldsda-L2': "o",
    'gdpopt.ldsda-Linf': "s",
    'gdpopt.ldbd-L2': "v",
    'gdpopt.ldbd-Linf': "<",
    # 'gdpopt.lbb': color_palette[7],
}

# Increase figure size to accommodate legends at the bottom
fig, ax = plt.subplots(figsize=(10, 6))
solver_list = sorted(result['solver'].dropna().unique())

# Ensure we have enough colors
if len(solver_list) > 0:
    # Use 'deep', 'bright', or 'colorblind' for better visibility instead of 'Spectral'
    # 'tab10' is also a good default for distinct categorical colors
    color_palette = sns.color_palette("tab10", max(len(solver_list), 3))
    solver_color_dict = {
        solver: color_palette[idx] for idx, solver in enumerate(solver_list)
    }
else:
    solver_color_dict = {}

plt.xticks(ticks=[4, 5, 6, 7, 8, 9], labels=[4, 5, 6, 7, 8, 9])

# Get all unique strategies present in the data
unique_strategies = result['strategy'].unique()

for strategy in unique_strategies:
    # Determine base strategy (remove -Transfer suffix) for maker lookup
    base_strategy = strategy.replace('-Transfer', '')
    
    # Skip if strategy is unknown (not in our dict)
    if base_strategy not in strategy_maker_dict:
        continue
        
    maker = strategy_maker_dict[base_strategy]
    is_transfer = '-Transfer' in strategy
    
    for solver in solver_color_dict:
        color = solver_color_dict[solver]
        subset = result[
            (result['strategy'] == strategy)
            & (result['solver'] == solver)
            & (
                abs(result['Upper bound'] - result['Optimal objective value'])
                < abs(result['Optimal objective value']) * 0.001
            )
        ]
        
        if subset.empty:
            continue
            
        # Distinguish Transfer: e.g., using hollow markers or different alpha
        # Here using 'none' for facecolor to make hollow markers for Transfer
        face_color = 'none' if is_transfer else color
        label_suffix = " (Transfer)" if is_transfer else ""
        
        plt.plot(
            subset['Stage'],
            subset['Time'],
            label=f'{strategy} - {solver}',
            mec=color if is_transfer else 'black', # Edge color same as solver color for transfer, else black
            mfc=face_color,
            marker=maker,
            linestyle='',
            color=color,
            markersize=8,
            markeredgewidth=1.5 if is_transfer else 0.5, # Thicker edge for hollow markers
            alpha=0.9,
        )
plt.xlabel('Number of Stages')
plt.ylabel("Solution Time [s]\n(within 0.1% of known optimal value)")
plt.yscale('log')

# Remove any default legend first (we will build custom ones)
if ax.get_legend():
    ax.get_legend().remove()

# Sort the legend labels and handles
GDPopt_handles, GDPopt_labels = [], []
LD_handles, LD_labels = [], []
MINLP_handles, MINLP_labels = [], []
Enum_handles, Enum_labels = [], []


def format_solver_label(label):
    if " - " in label:
        prefix, solver = label.rsplit(" - ", 1)
        return f"{prefix} - {solver.upper()}"
    return label

# We need to collect ALL handles and labels first. 
# get_legend_handles_labels may return only what's currently in the legend if ax.legend() was called, 
# or all artists with labels if not. Since we removed the legend previously, ensure we get everything.
handles, labels = ax.get_legend_handles_labels()

# Sort handles and labels alphabetically to ensure consistent order
if labels:
    sorted_indices = np.argsort(labels)
    handles = [handles[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

for handle, label in zip(handles, labels):

    if 'gdpopt.ldsda' in label or 'gdpopt.ldbd' in label:
        LD_handles.append(handle)
        LD_labels.append(
            format_solver_label(
                label.replace('gdpopt.ldsda-', 'LDSDA ')
                .replace('gdpopt.ldbd-', 'LDBD ')
                .replace('gdpopt.ldsda', 'LDSDA')
                .replace('gdpopt.ldbd', 'LDBD')
            )
        )
    elif 'gdpopt.' in label:
        GDPopt_handles.append(handle)
        GDPopt_labels.append(
            format_solver_label(
                label.replace('gdpopt.', '')
                .replace('enumerate', 'Enum')
                .replace('gloa', 'GLOA')
                .replace('loa', 'LOA')
            )
        )
    elif 'gdp.' in label:
        MINLP_handles.append(handle)
        MINLP_labels.append(
            format_solver_label(
                label.replace('gdp.', '')
                .replace('bigm', 'BigM')
                .replace('hull', 'Hull')
            )
        )

# Create custom legends placed to the right of the plot area
# Adjust subplot params to leave space at the right for legends
plt.subplots_adjust(right=0.75)

# Prepare legends to add
legends_to_add = []
if LD_handles:
    legends_to_add.append((LD_handles, LD_labels, "LD Algorithms"))
if GDPopt_handles:
    legends_to_add.append((GDPopt_handles, GDPopt_labels, "GDPOpt"))
if MINLP_handles:
    legends_to_add.append((MINLP_handles, MINLP_labels, "MINLP"))

# Shrink the plot area to make space on the right (20% reduction in width)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

extra_artists = []
# Distribute legends vertically on the right side
num_legs = len(legends_to_add)
if num_legs > 0:
    for i, (leg_handles, leg_labels, title) in enumerate(legends_to_add):
        # Calculate vertical position.
        # We want to distribute them evenly along the vertical axis.
        # i=0 (top) -> near 1.0, i=last (bottom) -> near 0.0
        
        y_anchor = 1.0 - (i + 0.5) / num_legs
        
        leg = ax.legend(
            leg_handles,
            leg_labels,
            title=title,
            loc='center left', 
            bbox_to_anchor=(1.05, y_anchor), # Place outside to the right (slightly further)
            fontsize=10, 
            ncol=1,
            frameon=True,
            title_fontsize=11
        )
        ax.add_artist(leg)
        extra_artists.append(leg)

# Pass the extra artists so bbox_inches='tight' includes them
plt.savefig('figures/computational_results_comparison.pdf', bbox_inches='tight', bbox_extra_artists=extra_artists)
