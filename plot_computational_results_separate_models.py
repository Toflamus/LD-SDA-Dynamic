import os
import json
from glob import glob

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError as e:
    missing = getattr(e, "name", "<unknown>")
    raise SystemExit(
        "Missing python package: "
        + str(missing)
        + "\n\n"
        + "This plotting script requires: matplotlib, pandas, seaborn, numpy.\n"
        + "Recommended (conda, from repo root):\n"
        + "  conda env create -f environment.yml\n"
        + "  conda activate pyomo_ldbd\n"
        + "\n"
        + "Alternatively (pip):\n"
        + "  pip install matplotlib pandas seaborn numpy\n"
    )

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Update these folders to point at the run you want to plot
json_file_folder_dict = {
    4: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/four_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-44-27',
    5: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/five_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-45-05',
    6: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/six_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-46-07',
    7: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/seven_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-54-27',
    8: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/eight_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-54-51',
    9: '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/nine_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-55-10',
}

# Used only for jittering points at time limit
time_limit = {4: 900, 5: 900, 6: 900, 7: 1800, 8: 1800, 9: 3600}

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
}


def read_data(file_folder: str, stage: int) -> pd.DataFrame:
    data = []

    for f_name in glob(os.path.join(file_folder, '*.json')):
        with open(f_name, 'r') as f:
            json_data = json.load(f)

        base_name = os.path.basename(f_name)[:-5]
        parts = base_name.split('_')
        strategy = parts[0] if parts else '-'
        solver = parts[1] if len(parts) > 1 else '-'

        is_transfer = 'mode_transfer' in base_name
        model = 'Transfer' if is_transfer else 'Standard'

        # Create a strategy label like the original script (add norm for LDSDA/LDBD)
        if strategy in ('gdpopt.ldsda', 'gdpopt.ldbd') and len(parts) > 2:
            s_name = strategy + '-' + parts[2]
        else:
            s_name = strategy

        json_data['strategy'] = s_name
        json_data['solver'] = solver
        json_data['Model'] = model
        json_data['Stage'] = stage

        if isinstance(json_data.get('Problem'), list):
            json_data['Problem'] = json_data['Problem'][0]
        if isinstance(json_data.get('Solver'), list):
            json_data['Solver'] = json_data['Solver'][0]

        data.append(json_data)

    df = pd.json_normalize(data)

    if df.empty:
        return pd.DataFrame(
            columns=[
                'strategy',
                'solver',
                'Model',
                'Stage',
                'Problem.Lower bound',
                'Problem.Upper bound',
                'Solver.User time',
                'Solver.Termination condition',
            ]
        )

    keep_cols = [
        'strategy',
        'solver',
        'Model',
        'Stage',
        'Problem.Lower bound',
        'Problem.Upper bound',
        'Solver.User time',
        'Solver.Termination condition',
    ]

    result = df[keep_cols].copy()
    result.rename(
        columns={
            'Problem.Lower bound': 'Lower bound',
            'Problem.Upper bound': 'Upper bound',
            'Solver.User time': 'Time',
            'Solver.Termination condition': 'Termination condition',
        },
        inplace=True,
    )

    # Coerce numeric columns
    for col in ['Lower bound', 'Upper bound', 'Time']:
        result[col] = pd.to_numeric(result[col], errors='coerce')

    return result


def add_best_known_optimal(result: pd.DataFrame) -> pd.DataFrame:
    """Compute best-known UB per (Stage, Model) and attach as 'Optimal objective value'."""
    out = result.copy()

    # Minimization: best known upper bound = minimum UB among available runs
    best = (
        out.groupby(['Stage', 'Model'], as_index=False)['Upper bound']
        .min()
        .rename(columns={'Upper bound': 'Optimal objective value'})
    )

    out = out.merge(best, on=['Stage', 'Model'], how='left')
    return out


def apply_time_limit_jitter(result: pd.DataFrame, tol: float = 0.001) -> pd.DataFrame:
    """Jitter x positions (Stage) for points at time limit that are within tol of optimal."""
    out = result.copy()

    # We will add small offsets (e.g., 7 -> 7.1), so make Stage float.
    if 'Stage' in out.columns:
        out['Stage'] = pd.to_numeric(out['Stage'], errors='coerce').astype(float)

    for stage, limit in time_limit.items():
        for model in out['Model'].dropna().unique():
            mask = (
                (out['Stage'] == stage)
                & (out['Model'] == model)
                & (out['Time'] >= limit)
                & (out['Upper bound'].notna())
                & (out['Optimal objective value'].notna())
                & (
                    abs(out['Upper bound'] - out['Optimal objective value'])
                    < abs(out['Optimal objective value']) * tol
                )
            )
            temp = out[mask]

            if len(temp) <= 1:
                continue

            step = 0.1
            for idx, index in enumerate(temp.index):
                out.loc[index, 'Stage'] = out.loc[index, 'Stage'] + step * (-len(temp) + 1 + 2 * idx)

    return out


def plot_one_model(result: pd.DataFrame, model: str, output_path: str, tol: float = 0.001) -> None:
    subset_all = result[result['Model'] == model].copy()

    # Compute gap% relative to best-known per (Stage, Model)
    denom = subset_all['Optimal objective value'].abs().clip(lower=1e-10)
    subset_all['GapPct'] = (subset_all['Upper bound'] - subset_all['Optimal objective value']).abs() / denom * 100.0

    within_tol_mask = (
        subset_all['Upper bound'].notna()
        & subset_all['Optimal objective value'].notna()
        & (
            (subset_all['Upper bound'] - subset_all['Optimal objective value']).abs()
            < subset_all['Optimal objective value'].abs() * tol
        )
    )
    subset_all['WithinTol'] = within_tol_mask

    
    subset_all['Upper bound'] = pd.to_numeric(subset_all['Upper bound'], errors='coerce')
    subset_all['IsFail'] = ~np.isfinite(subset_all['Upper bound'])
    

    if subset_all.empty:
        print(f"No rows to plot for Model={model}")
        return

    # Color by solver
    solver_list = sorted(subset_all['solver'].dropna().unique())
    color_palette = sns.color_palette("tab10", max(len(solver_list), 3))
    solver_color_dict = {solver: color_palette[idx] for idx, solver in enumerate(solver_list)}

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(ticks=[4, 5, 6, 7, 8, 9], labels=[4, 5, 6, 7, 8, 9])

    unique_strategies = subset_all['strategy'].dropna().unique()

    for strategy in unique_strategies:
        if strategy not in strategy_maker_dict:
            continue

        marker = strategy_maker_dict[strategy]

        for solver, color in solver_color_dict.items():
            sub = subset_all[(subset_all['strategy'] == strategy) & (subset_all['solver'] == solver)]
            if sub.empty:
                continue

            # Plot within-tol points (solid)
            sub_in = sub[sub['WithinTol'] == True]
            label = f'{strategy} - {solver}'

            if not sub_in.empty:
                plt.plot(
                    sub_in['Stage'],
                    sub_in['Time'],
                    label=label,
                    marker=marker,
                    linestyle='',
                    color=color,
                    markersize=8,
                    markeredgewidth=0.6,
                    mec='black',
                    mfc=color,
                    alpha=0.9,
                )

            # Plot out-of-tol points (hollow) and annotate gap%
            sub_out = sub[sub['WithinTol'] == False]
            if not sub_out.empty:
                plt.plot(
                    sub_out['Stage'],
                    sub_out['Time'],
                    # If there are NO within-tol points for this (strategy, solver),
                    # keep the legend entry so markers/colors still match the legend.
                    label=label if sub_in.empty else '_nolegend_',
                    marker=marker,
                    linestyle='',
                    color=color,
                    markersize=8,
                    markeredgewidth=1.2,
                    mec=color,
                    mfc='none',
                    alpha=0.9,
                )

                for _, r in sub_out.iterrows():
                    if pd.isna(r.get('GapPct')) or pd.isna(r.get('Time')) or pd.isna(r.get('Stage')):
                        continue
                    # Annotate only when we can compute a finite gap
                    gap_val = r['GapPct']
                    if np.isfinite(gap_val):
                        ax.annotate(
                            f"{gap_val:.1f}%",
                            (r['Stage'], r['Time']),
                            textcoords="offset points",
                            xytext=(3, 3),
                            ha='left',
                            va='bottom',
                            fontsize=7,
                            color=color,
                        )

            # Annotate UB=inf (non-finite) as 'inf%'
            sub_fail = sub[sub['IsFail'] == True]
            if not sub_fail.empty:
                for _, r in sub_fail.iterrows():
                    if pd.isna(r.get('Time')) or pd.isna(r.get('Stage')):
                        continue
                    ax.annotate(
                        "inf%",
                        (r['Stage'], r['Time']),
                        textcoords="offset points",
                        xytext=(3, -10),
                        ha='left',
                        va='top',
                        fontsize=8,
                        color='red',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='red', alpha=0.7),
                        clip_on=False,
                    )

    plt.xlabel('Number of Stages')
    plt.ylabel("Solution Time [s]")
    plt.yscale('log')
    plt.title(f'Computational results ({model} model)')

    # Build grouped legends similar to the original
    if ax.get_legend():
        ax.get_legend().remove()

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        sorted_indices = np.argsort(labels)
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]

    LD_handles, LD_labels = [], []
    GDPopt_handles, GDPopt_labels = [], []
    MINLP_handles, MINLP_labels = [], []

    def format_solver_label(label: str) -> str:
        if " - " in label:
            prefix, solver = label.rsplit(" - ", 1)
            return f"{prefix} - {solver.upper()}"
        return label

    for h, lab in zip(handles, labels):
        if 'gdpopt.ldsda' in lab or 'gdpopt.ldbd' in lab:
            LD_handles.append(h)
            LD_labels.append(
                format_solver_label(
                    lab.replace('gdpopt.ldsda-', 'LDSDA ')
                    .replace('gdpopt.ldbd-', 'LDBD ')
                    .replace('gdpopt.ldsda', 'LDSDA')
                    .replace('gdpopt.ldbd', 'LDBD')
                )
            )
        elif 'gdpopt.' in lab:
            GDPopt_handles.append(h)
            GDPopt_labels.append(
                format_solver_label(
                    lab.replace('gdpopt.', '')
                    .replace('enumerate', 'Enum')
                    .replace('gloa', 'GLOA')
                    .replace('loa', 'LOA')
                )
            )
        elif 'gdp.' in lab:
            MINLP_handles.append(h)
            MINLP_labels.append(
                format_solver_label(
                    lab.replace('gdp.', '').replace('bigm', 'BigM').replace('hull', 'Hull')
                )
            )

    legends_to_add = []
    if LD_handles:
        legends_to_add.append((LD_handles, LD_labels, "LD Algorithms"))
    if GDPopt_handles:
        legends_to_add.append((GDPopt_handles, GDPopt_labels, "GDPOpt"))
    if MINLP_handles:
        legends_to_add.append((MINLP_handles, MINLP_labels, "MINLP"))

    plt.subplots_adjust(right=0.75)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    extra_artists = []
    num_legs = len(legends_to_add)
    if num_legs > 0:
        for i, (leg_handles, leg_labels, title) in enumerate(legends_to_add):
            y_anchor = 1.0 - (i + 0.5) / num_legs
            leg = ax.legend(
                leg_handles,
                leg_labels,
                title=title,
                loc='center left',
                bbox_to_anchor=(1.05, y_anchor),
                fontsize=10,
                ncol=1,
                frameon=True,
                title_fontsize=11,
            )
            ax.add_artist(leg)
            extra_artists.append(leg)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', bbox_extra_artists=extra_artists)
    plt.close(fig)


def main() -> None:
    result_list = []
    for stage, folder in json_file_folder_dict.items():
        df = read_data(folder, stage)
        if not df.empty:
            result_list.append(df)

    if not result_list:
        print("No JSON files found / no data to plot.")
        return

    result = pd.concat(result_list).reset_index(drop=True)
    result = add_best_known_optimal(result)
    result = apply_time_limit_jitter(result)

    out_dir = 'figures'
    plot_one_model(result, model='Standard', output_path=os.path.join(out_dir, 'computational_results_standard.pdf'))
    plot_one_model(result, model='Transfer', output_path=os.path.join(out_dir, 'computational_results_transfer.pdf'))


if __name__ == '__main__':
    main()
