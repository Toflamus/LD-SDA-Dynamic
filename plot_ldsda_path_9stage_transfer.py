import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# Background scatter: complete enumeration objective values (x,y)->obj
# Reused from LDSDA_visualization.py (fits the (1..9,2..9) space)
FEASIBLE_SOLUTION: Dict[Tuple[int, int], float] = {
    (1, 2): -197.14013122689622,
    (1, 3): -206.36041265432806,
    (1, 4): -212.00878719112268,
    (1, 5): -207.62611633572016,
    (1, 6): -196.93118428995865,
    (1, 7): -178.63847525562682,
    (1, 8): -146.87850122234687,
    (1, 9): -87.5338894428716,
    (2, 3): -117.07438084784795,
    (2, 4): -137.1797042946188,
    (2, 5): -192.870848809807,
    (2, 6): -192.88392569521721,
    (2, 7): -175.94473700875903,
    (2, 8): -144.71276519403037,
    (2, 9): -85.82696130620648,
    (3, 4): -77.19743879461578,
    (3, 5): -88.41538101128177,
    (3, 6): -119.38309938589616,
    (3, 7): -170.89539354574913,
    (3, 8): -141.82503814335138,
    (3, 9): -83.55206301486514,
    (4, 5): -55.81190616713566,
    (4, 6): -62.4732860007934,
    (4, 7): -77.86237831841856,
    (4, 8): -135.07901900226858,
    (4, 9): -80.51745201314861,
    (5, 6): -43.41238657553521,
    (5, 7): -47.28320969606174,
    (5, 8): -54.212220237141665,
    (5, 9): -66.30029047415469,
    (6, 7): -36.01889379296989,
    (6, 8): -37.96970292827284,
    (6, 9): -39.58228460866401,
    (7, 8): -31.679373632369185,
    (7, 9): -32.196321090299925,
    (8, 9): -29.327202909495576,
    (9, 9): -28.596739202011523,
}


@dataclass
class LdsdaRow:
    iteration: int
    search_type: str
    ext_vars: Tuple[int, ...]
    lower_bound: str
    upper_bound: str
    gap: str
    time_s: float
    starred: bool


@dataclass
class LdsdaLog:
    algorithm: str
    norm: str
    starting_point: Optional[Tuple[int, ...]]
    search_path: List[Tuple[int, ...]]
    rows: List[LdsdaRow]


@dataclass(frozen=True)
class LogMeta:
    algorithm: str
    solver: str
    norm: str
    model: str


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        raise ValueError(f"Empty tuple: {s}")
    return tuple(int(p) for p in parts)


def parse_log_filename(log_path: str) -> Optional[LogMeta]:
    """Parse (algorithm, solver, norm, model) from GDPopt log file name.

    Supported examples:
      - gdpopt.ldsda_ipopt_L2_mode_transfer.log
      - gdpopt.ldbd_conopt_Linf.log
    """
    base = os.path.basename(log_path)
    m = re.match(
        r'^gdpopt\.(?P<algo>ldsda|ldbd)_(?P<solver>[^_]+)_(?P<norm>L2|Linf)(?P<rest>.*)\.log$',
        base,
    )
    if not m:
        return None

    rest = m.group('rest') or ''
    model = 'Transfer' if 'mode_transfer' in rest else 'Standard'
    return LogMeta(
        algorithm=m.group('algo'),
        solver=m.group('solver'),
        norm=m.group('norm'),
        model=model,
    )


def parse_ldsda_log(log_path: str, algorithm_hint: Optional[str] = None) -> LdsdaLog:
    with open(log_path, 'r') as f:
        lines = f.read().splitlines()

    algorithm = algorithm_hint or 'unknown'
    if algorithm_hint is None:
        for line in lines:
            # e.g. - Name: GDPopt (...) - LDSDA
            m = re.search(r'\s-\s(LDSDA|LDBD)\s*$', line)
            if m:
                algorithm = m.group(1).lower()
                break

    norm = "-"
    starting_point: Optional[Tuple[int, ...]] = None
    rows: List[LdsdaRow] = []
    search_path: List[Tuple[int, ...]] = []

    # Find norm + starting point
    for line in lines:
        if line.startswith('direction_norm:'):
            norm = line.split(':', 1)[1].strip()
        if line.startswith('starting_point:'):
            # e.g. starting_point: [1, 2]
            m = re.search(r'\[(.*?)\]', line)
            if m:
                starting_point = _parse_int_tuple(m.group(1))

    # Parse table block
    in_table = False
    for line in lines:
        if line.strip().startswith('Iteration |'):
            in_table = True
            continue
        if in_table:
            if line.strip().startswith('Search path:'):
                in_table = False
                continue
            if not line.strip():
                continue
            if line.startswith('===='):
                continue

            # Example row:
            # 1   Neighbor search   (1, 3)   -inf   -206.36041  inf%  135.16  *
            # Columns are aligned; we parse using regex.
            m = re.match(
                r'^\s*(\d+)\s+(.*?)\s+\(([^)]*)\)\s+([-\w\.]+)\s+([-\w\.]+)\s+(\S+)\s+([0-9\.]+)\s*(\*)?\s*$',
                line,
            )
            if not m:
                # Ignore lines that don't match the row format
                continue

            iteration = int(m.group(1))
            search_type = m.group(2).strip()
            ext_vars = _parse_int_tuple(m.group(3).strip())
            lb = m.group(4)
            ub = m.group(5)
            gap = m.group(6)
            time_s = float(m.group(7))
            starred = m.group(8) is not None

            rows.append(
                LdsdaRow(
                    iteration=iteration,
                    search_type=search_type,
                    ext_vars=ext_vars,
                    lower_bound=lb,
                    upper_bound=ub,
                    gap=gap,
                    time_s=time_s,
                    starred=starred,
                )
            )

    # Parse Search path
    for line in lines:
        if line.strip().startswith('Search path:'):
            # e.g. Search path: (1, 2) -> (1, 3)
            tuples = re.findall(r'\(([^)]*)\)', line)
            search_path = [_parse_int_tuple(t) for t in tuples]
            break

    # Fallback: if path not found, use starred rows as path
    if not search_path:
        starred_points = [r.ext_vars for r in rows if r.starred]
        if starting_point is not None:
            # ensure starting point is first
            if not starred_points or starred_points[0] != starting_point:
                search_path = [starting_point] + starred_points
            else:
                search_path = starred_points
        else:
            search_path = starred_points

    return LdsdaLog(
        algorithm=algorithm,
        norm=norm,
        starting_point=starting_point,
        search_path=search_path,
        rows=rows,
    )


def build_exploration_edges(log: LdsdaLog) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Infer exploration arrows from incumbent-at-iteration to evaluated points."""
    if not log.search_path:
        return []

    edges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    def incumbent_at_iteration_start(iteration: int) -> Tuple[int, ...]:
        """Best-effort incumbent point at the *start* of a given iteration.

        Empirically, LDSDA logs may contain multiple starred rows within the same
        iteration (e.g., a neighbor acceptance followed by a line-search acceptance).
        For exploration steps, mapping each row to the incumbent at the start of the
        iteration avoids incorrect edges like (1,3)->(2,3) when (2,3) is evaluated
        during the neighbor search around (1,2).

        Rule used:
          incumbent(iter i) = last starred ext_vars with row.iteration < i
          (fallback to search_path[0]).
        """

        if iteration <= 0:
            return log.search_path[0]
        last: Optional[Tuple[int, ...]] = None
        for row in log.rows:
            if row.starred and row.iteration < iteration:
                last = row.ext_vars
        return last if last is not None else log.search_path[0]

    path = log.search_path

    for r in log.rows:
        if log.algorithm == 'ldsda':
            incumbent = incumbent_at_iteration_start(r.iteration)
        else:
            # LDBD: iteration i corresponds to incumbent path[i-1]
            idx = min(max(r.iteration - 1, 0), len(path) - 1)
            incumbent = path[idx]

        if r.ext_vars != incumbent:
            edges.append((incumbent, r.ext_vars))

    # De-duplicate while preserving order
    seen = set()
    out = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def visualize_compare_l2_linf(
    l2: LdsdaLog,
    linf: LdsdaLog,
    output_path: str,
    offset: float = 0.12,
    title: str = 'Search path comparison',
) -> None:
    if not l2.search_path or not linf.search_path:
        raise ValueError('Empty search path in one of the logs.')

    dims = {len(p) for p in (l2.search_path + linf.search_path)}
    if dims != {2}:
        raise ValueError(
            f"This visualization supports only 2D external variables, got dimensions: {sorted(dims)}"
        )

    plt.figure(figsize=(9, 7))

    feas_x = [k[0] for k in FEASIBLE_SOLUTION]
    feas_y = [k[1] for k in FEASIBLE_SOLUTION]
    objs = [FEASIBLE_SOLUTION[k] for k in FEASIBLE_SOLUTION]

    sc = plt.scatter(feas_x, feas_y, s=80, c=objs, cmap='viridis_r')

    # Styling
    arrow_width = 2.5
    arrow_head_size = 12
    font_size = 15
    number_font_size = 14

    l2_color = 'red'
    linf_color = 'blue'

    # Plot L2 incumbent path
    for i in range(len(l2.search_path) - 1):
        plt.annotate(
            '',
            xy=l2.search_path[i + 1],
            xytext=l2.search_path[i],
            arrowprops=dict(
                arrowstyle='->',
                color=l2_color,
                linestyle='solid',
                lw=arrow_width,
                mutation_scale=arrow_head_size,
            ),
        )

    # Plot Linf incumbent path with offset
    for i in range(len(linf.search_path) - 1):
        start = (linf.search_path[i][0] + offset, linf.search_path[i][1])
        end = (linf.search_path[i + 1][0] + offset, linf.search_path[i + 1][1])
        plt.annotate(
            '',
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle='->',
                color=linf_color,
                linestyle='solid',
                lw=arrow_width,
                mutation_scale=arrow_head_size,
            ),
        )

    # Exploration edges
    exp_l2 = build_exploration_edges(l2)
    exp_linf = build_exploration_edges(linf)

    for a, b in exp_l2:
        plt.annotate(
            '',
            xy=b,
            xytext=a,
            arrowprops=dict(
                arrowstyle='->',
                color=l2_color,
                linestyle='dashed',
                alpha=0.9,
                lw=arrow_width,
                mutation_scale=arrow_head_size,
            ),
        )

    for a, b in exp_linf:
        a2 = (a[0] + offset, a[1])
        b2 = (b[0] + offset, b[1])
        plt.annotate(
            '',
            xy=b2,
            xytext=a2,
            arrowprops=dict(
                arrowstyle='->',
                color=linf_color,
                linestyle='dotted',
                alpha=0.8,
                lw=arrow_width,
                mutation_scale=arrow_head_size,
            ),
        )

    # Legend
    custom_lines = [
        Line2D([0], [0], color=l2_color, lw=1, linestyle='solid', marker='>', markeredgewidth=0.5, markersize=arrow_head_size),
        Line2D([0], [0], color=linf_color, lw=1, linestyle='solid', marker='>', markeredgewidth=0.5, markersize=arrow_head_size),
        Line2D([0], [0], color=l2_color, lw=1, linestyle='dashed', marker='>', markeredgewidth=0.5, markersize=arrow_head_size),
        Line2D([0], [0], color=linf_color, lw=1, linestyle='dotted', marker='>', markeredgewidth=0.5, markersize=arrow_head_size),
    ]

    plt.legend(
        custom_lines,
        [
            '$L_2$ Path',
            r'$L_{\infty}$ Path',
            '$L_2$ Exploration',
            r'$L_{\infty}$ Exploration',
        ],
        fontsize=font_size,
    )

    cbar = plt.colorbar(sc)
    cbar.set_label('Objective function', rotation=270, labelpad=15, fontsize=font_size)
    cbar.ax.tick_params(labelsize=number_font_size)

    plt.xlabel('$Z_{s,1}$', fontsize=font_size)
    plt.ylabel('$Z_{s,2}$', fontsize=font_size)
    plt.xticks(fontsize=number_font_size)
    plt.yticks(fontsize=number_font_size)

    plt.title(title, fontsize=font_size)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()


def infer_stage_label_from_path(input_dir: str) -> Optional[str]:
    parts = os.path.normpath(input_dir).split(os.sep)
    if 'results' in parts:
        idx = parts.index('results')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def find_and_group_logs(
    input_dir: str,
    algorithm: str = 'all',
    model: str = 'transfer',
) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Scan a directory and group logs by (algorithm, solver, model).

    Returns:
      groups[(algo, solver, model)][norm] = log_path
    """
    groups: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for log_path in sorted(glob.glob(os.path.join(input_dir, 'gdpopt.*.log'))):
        meta = parse_log_filename(log_path)
        if meta is None:
            continue
        if algorithm != 'all' and meta.algorithm != algorithm:
            continue
        if model == 'transfer' and meta.model != 'Transfer':
            continue
        if model == 'standard' and meta.model != 'Standard':
            continue

        key = (meta.algorithm, meta.solver, meta.model)
        groups.setdefault(key, {})[meta.norm] = log_path

    return groups


def main() -> None:
    default_folder = (
        'results/nine_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-55-10'
    )
    parser = argparse.ArgumentParser(
        description=(
            'Plot GDPopt LDSDA/LDBD search paths (compare L2 vs Linf). '
            'Supports auto-detecting algorithm/solver/model from file names and generating plots per group.'
        )
    )
    parser.add_argument(
        '--input-dir',
        default=default_folder,
        help='Directory containing GDPopt log files to scan (relative to LD-SDA-Dynamic or absolute).',
    )
    parser.add_argument(
        '--algorithm',
        choices=['all', 'ldsda', 'ldbd'],
        default='all',
        help='Filter which algorithm logs to plot.',
    )
    parser.add_argument(
        '--model',
        choices=['transfer', 'standard', 'all'],
        default='transfer',
        help='Filter transfer/standard logs based on file name.',
    )
    parser.add_argument(
        '--output-dir',
        default='figures',
        help='Output directory for PDFs (relative to LD-SDA-Dynamic or absolute).',
    )
    parser.add_argument(
        '--offset',
        type=float,
        default=0.12,
        help='X-axis offset applied to Linf series to avoid overlap.',
    )
    parser.add_argument(
        '--l2-log',
        default=None,
        help='(Optional) Explicit L2 log path; if set with --linf-log, plots only this pair.',
    )
    parser.add_argument(
        '--linf-log',
        default=None,
        help='(Optional) Explicit Linf log path; if set with --l2-log, plots only this pair.',
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(script_dir, p)

    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stage_label = infer_stage_label_from_path(input_dir)

    # Manual mode: plot exactly one pair
    if (args.l2_log is not None) or (args.linf_log is not None):
        if not args.l2_log or not args.linf_log:
            raise SystemExit('Provide both --l2-log and --linf-log (or neither).')
        l2_log_path = resolve(args.l2_log)
        linf_log_path = resolve(args.linf_log)

        if not os.path.exists(l2_log_path):
            raise SystemExit(f"L2 log not found: {l2_log_path}")
        if not os.path.exists(linf_log_path):
            raise SystemExit(f"Linf log not found: {linf_log_path}")

        meta = parse_log_filename(l2_log_path) or parse_log_filename(linf_log_path)
        title = 'Search path comparison'
        if meta is not None:
            title_parts = [meta.algorithm.upper(), meta.solver, meta.model]
            if stage_label:
                title_parts.insert(1, stage_label)
            title = ' '.join(title_parts) + ' (L2 vs Linf)'

        out_name = 'search_path_L2_vs_Linf.pdf'
        if meta is not None:
            out_name = f"{meta.algorithm}_{meta.solver}_{meta.model}_L2_vs_Linf.pdf"
        out_path = os.path.join(output_dir, out_name)

        algo_hint = None
        if meta is not None:
            algo_hint = meta.algorithm
        l2 = parse_ldsda_log(l2_log_path, algorithm_hint=algo_hint)
        linf = parse_ldsda_log(linf_log_path, algorithm_hint=algo_hint)
        visualize_compare_l2_linf(l2, linf, out_path, offset=args.offset, title=title)
        print(f"Saved: {out_path}")
        return

    # Auto-detect mode: scan directory
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    groups = find_and_group_logs(
        input_dir,
        algorithm=args.algorithm,
        model=args.model,
    )

    if not groups:
        raise SystemExit(f"No matching gdpopt.*.log files found in: {input_dir}")

    created = 0
    for (algo, solver, model_name), by_norm in sorted(groups.items()):
        if 'L2' not in by_norm or 'Linf' not in by_norm:
            continue

        l2 = parse_ldsda_log(by_norm['L2'], algorithm_hint=algo)
        linf = parse_ldsda_log(by_norm['Linf'], algorithm_hint=algo)

        title_parts = [algo.upper(), solver, model_name]
        if stage_label:
            title_parts.insert(1, stage_label)
        title = ' '.join(title_parts) + ' (L2 vs Linf)'

        out_name = f"{algo}_{solver}_{model_name}_L2_vs_Linf.pdf"
        out_path = os.path.join(output_dir, out_name)
        visualize_compare_l2_linf(l2, linf, out_path, offset=args.offset, title=title)
        created += 1
        print(f"Saved: {out_path}")

    if created == 0:
        raise SystemExit('Found logs, but no (L2, Linf) pairs to plot after filtering.')


if __name__ == '__main__':
    main()
