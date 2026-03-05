import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load data
# Please replace 'your_data.csv' with your actual CSV file path
df = pd.read_csv('/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/merged_summary_results.csv')
target_folder = '/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/figures'

# 2. Define the optimal objective value dictionary
optimal_objective_value = {
    4: -23.3046896, 
    5: -52.79223,
    6: -122.423721,
    7: -160.08149,
    8: -188.449493,
    9: -212.00879,
}

# 3. Set plotting style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Scalability Analysis: Time vs. Tray Size by Solver & Norm', fontsize=16)

# Define sub-plot combinations
combinations = [
    ('conopt', 'L2', axes[0, 0]),
    ('conopt', 'Linf', axes[0, 1]),
    ('ipopt', 'L2', axes[1, 0]),
    ('ipopt', 'Linf', axes[1, 1])
]

# Define line styles and color combinations for algorithms and models for easy differentiation
style_dict = {
    ('ldbd', 'Standard'): {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},   # Blue solid line, circle marker
    ('ldbd', 'Transfer'): {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},   # Orange solid line, square marker
    ('ldsda', 'Standard'): {'color': '#1f77b4', 'marker': '^', 'linestyle': '--'}, # Blue dashed line, triangle marker
    ('ldsda', 'Transfer'): {'color': '#ff7f0e', 'marker': 'd', 'linestyle': '--'}  # Orange dashed line, diamond marker
}

# 4. Loop to plot the four subplots
for subsolver, norm, ax in combinations:
    subset = df[(df['Subsolver'] == subsolver) & (df['Norm'] == norm)]
    
    ax.set_title(f'Subsolver: {subsolver.upper()} | Norm: {norm}', fontsize=14)
    ax.set_xlabel('Number of Trays', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_xticks(list(optimal_objective_value.keys()))
    
    # Iterate through the four combinations of algorithm x model
    for algo in ['ldbd', 'ldsda']:
        for model in ['Standard', 'Transfer']:
            group = subset[(subset['Algorithm'] == algo) & (subset['Model'] == model)]
            
            if group.empty:
                continue
                
            group = group.sort_values(by='tray')
            x = group['tray'].values
            y = group['Time (s)'].values
            
            style = style_dict[(algo, model)]
            label = f"{algo.upper()} - {model}"
            
            ax.plot(x, y, label=label, color=style['color'], 
                    marker=style['marker'], linestyle=style['linestyle'], markersize=8)
            
            # Calculate and annotate the Gap (for points deviating from the optimal solution)
            for _, row in group.iterrows():
                tray_val = row['tray']
                ub_val = row['Upper Bound']
                opt_val = optimal_objective_value[tray_val]
                
                # Handle abnormal 'inf' or extremely large failure values
                if np.isinf(ub_val) or ub_val > 100000:
                    ax.annotate('Fail', 
                                (tray_val, row['Time (s)']),
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center',
                                color='red',
                                fontsize=9,
                                weight='bold')
                else:
                    # Calculate absolute percentage Gap
                    gap_pct = abs((ub_val - opt_val) / opt_val) * 100
                    
                    # Only annotate if the Gap is greater than 0.05% (considered a deviation from the optimal)
                    if gap_pct > 0.05:
                        ax.annotate(f"{gap_pct:.1f}%", 
                                    (tray_val, row['Time (s)']),
                                    textcoords="offset points", 
                                    xytext=(0,10), 
                                    ha='center',
                                    color='darkred',
                                    fontsize=9)

    ax.legend(title='Algorithm - Model', fontsize=10)



# Ensure the target directory exists
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    print(f"Created directory: {target_folder}")

# Define the output filename
output_filename = 'scalability_ldbd_vs_ldsda_time.png'
save_path = os.path.join(target_folder, output_filename)

# Save the plot with high resolution
plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
print(f"Plot successfully saved to: {save_path}")

# Show the plot
plt.show()



plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to leave space for the main title
plt.show()