"""
Split trading log windows (each covering 2 months) into monthly files.
Saves tool_policy_history and final_metrics computed per month.
"""

import json
import os
import numpy as np
from collections import defaultdict


def compute_metrics(daily_logs, start_equity):
    """Compute metrics for a given set of daily logs."""
    returns = [d['daily_return'] for d in daily_logs if d['daily_return'] != 0.0]
    equities = [d['total_equity'] for d in daily_logs]

    # Cumulative return from start_equity to last equity
    cum_ret = round((equities[-1] / start_equity - 1) * 100, 4)

    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = round(float(np.mean(returns) / np.std(returns)), 4)
        vol = round(float(np.std(returns)), 4)
    else:
        sharpe = 0.0
        vol = 0.0

    # Max drawdown over monthly equity curve (starting from start_equity)
    peak = start_equity
    max_dd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    max_dd = round(max_dd, 4)

    win_rate = round(sum(1 for r in returns if r > 0) / len(returns) * 100, 4) if returns else 0.0

    return {
        'cumulative_return': cum_ret,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'volatility': vol
    }


def split_log_file(input_path, output_dir):
    with open(input_path) as f:
        data = json.load(f)

    daily_logs = data['daily_logs']
    tool_policy_history = data['tool_policy_history']
    sim_info = data['simulation_info']

    # Group daily_logs by year-month
    months_logs = defaultdict(list)
    for log in daily_logs:
        ym = log['date'][:7]  # 'YYYY-MM'
        months_logs[ym].append(log)

    sorted_months = sorted(months_logs.keys())

    # Group tool_policy_history by year-month
    months_tph = defaultdict(list)
    for entry in tool_policy_history:
        ym = entry['date'][:7]
        months_tph[ym].append(entry)

    initial_equity = daily_logs[0]['total_equity']  # starting equity (e.g., 10000)

    # Track equity at start of each month
    current_equity = initial_equity

    for i, ym in enumerate(sorted_months):
        month_logs = months_logs[ym]
        month_tph = months_tph.get(ym, [])

        start_equity = current_equity
        metrics = compute_metrics(month_logs, start_equity)

        month_sim_info = {
            **sim_info,
            'start_date': month_logs[0]['date'],
            'end_date': month_logs[-1]['date'],
            'total_days': len(month_logs),
        }

        monthly_data = {
            'simulation_info': month_sim_info,
            'final_metrics': metrics,
            'final_portfolio': data['final_portfolio'] if i == len(sorted_months) - 1 else None,
            'daily_logs': month_logs,
            'tool_policy_history': month_tph,
        }

        # Remove None final_portfolio for non-last months
        if monthly_data['final_portfolio'] is None:
            monthly_data['final_portfolio'] = month_logs[-1].get('allocations', {})

        # Output filename: replace .json with _month{N}.json
        basename = os.path.basename(input_path)
        name, ext = os.path.splitext(basename)
        out_path = os.path.join(output_dir, f'{name}_month{i+1}{ext}')

        with open(out_path, 'w') as f:
            json.dump(monthly_data, f, indent=2, ensure_ascii=False)

        print(f'  Saved: {out_path}  ({ym}, {len(month_logs)} days, {len(month_tph)} policy entries)')

        # Update current_equity to end of this month
        current_equity = month_logs[-1]['total_equity']


def main():
    base = '/Users/sehee/trading-agent-multi-agents/trading_log'
    windows = ['window4_memory', 'window5_memory', 'window6_memory']

    for window in windows:
        window_dir = os.path.join(base, window)
        if not os.path.isdir(window_dir):
            print(f'Skipping {window} (not found)')
            continue

        run_files = [f for f in os.listdir(window_dir) if f.startswith('trading_log_base_run') and f.endswith('.json')]
        if not run_files:
            print(f'No run files in {window}')
            continue

        print(f'\n=== {window} ===')
        for run_file in sorted(run_files):
            input_path = os.path.join(window_dir, run_file)
            split_log_file(input_path, window_dir)


if __name__ == '__main__':
    main()
