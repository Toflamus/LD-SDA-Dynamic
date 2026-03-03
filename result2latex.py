import pandas as pd
from glob import glob
import json
import os


def pyomo_result_to_latex_table(json_file_folder, output_file_name='latex/results.tex'):
    data = []
    for f_name in glob(json_file_folder + '/*.json'):
        with open(f_name, 'r') as f:
            print(f_name)
            json_data = json.load(f)
            base_name = os.path.basename(f_name)[:-5]
            parts = base_name.split('_')
            strategy = parts[0] if parts else '-'
            solver = parts[1] if len(parts) > 1 else '-'
            if strategy in ('gdpopt.ldsda', 'gdpopt.ldbd') and len(parts) > 2:
                json_data['strategy'] = strategy + '-' + parts[2]
            else:
                json_data['strategy'] = strategy
            json_data['solver'] = solver
            if isinstance(json_data['Problem'], list):
                json_data['Problem'] = json_data['Problem'][0]
            if isinstance(json_data['Solver'], list):
                json_data['Solver'] = json_data['Solver'][0]
            data.append(json_data)
    df = pd.json_normalize(data)
    result = df[
        [
            'strategy',
            'solver',
            'Problem.Upper bound',
            'Solver.User time',
            'Solver.Termination condition',
        ]
    ]
    result['solver'] = result['solver'].str.upper()
    result.replace(
        {
            'gdpopt': 'GDPopt',
            'gdpopt.ldsda-L2': 'LDSDA $L_2$',
            'gdpopt.ldsda-Linf': 'LDSDA $L_\infty$',
            'gdpopt.ldbd-L2': 'LDBD $L_2$',
            'gdpopt.ldbd-Linf': 'LDBD $L_\infty$',
            'gdp.bigm': 'MINLP BigM',
            'gdp.hull': 'MINLP Hull',
            'gdpopt.enumerate': 'L-Enum',
            'gdpopt.loa': 'LOA',
            'gdpopt.gloa': 'GLOA',
            'infeasible': 'Infeasible',
            'optimal': 'Optimal',
        },
        inplace=True,
    )
    result = result.fillna('-')
    result.rename(
        columns={
            'strategy': 'Strategy',
            'solver': 'Solver',
            'Problem.Upper bound': 'Obj',
            'Solver.User time': 'Time',
            'Solver.Termination condition': 'Status',
        },
        inplace=True,
    )
    # result.to_csv('results.csv', index=False)
    result.to_latex(output_file_name, index=False, float_format="%.2f")


json_file_folder_dict = {
    3: 'results/three_stage_dynamic_model_switching/nfe30/2024-03-07_11-29-45',
    # 3: 'results/three_stage_dynamic_model_switching/nfe30/2024-02-24_11-21-02',
    4: 'results/four_stage_dynamic_model_switching_nonlinear/nfe30/2024-02-25_14-19-02',
    5: 'results/five_stage_dynamic_model_switching_nonlinear/nfe30/2024-02-26_00-17-16',
    6: 'results/six_stage_dynamic_model_switching_nonlinear/nfe30/2024-02-25_14-19-26',
    7: 'results/seven_stage_dynamic_model_switching_nonlinear/nfe30/2024-02-26_00-17-55',
    8: 'results/eight_stage_dynamic_model_switching_nonlinear/nfe30/2024-02-25_14-19-39',
    9: 'results/nine_stage_dynamic_model_switching_nonlinear/nfe30/2024-02-26_00-19-35',
}

for key in json_file_folder_dict:
    pyomo_result_to_latex_table(
        json_file_folder_dict[key], f'latex/results_{key}stage_ordering.tex'
    )
