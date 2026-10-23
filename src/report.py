"""Export curves from actual JSON metrics; never synthesize missing results."""
import argparse
import csv
import json
from pathlib import Path
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in ('1.5b', '3b'):
        run = args.runs / ('sdar_' + size)
        for path in sorted((run / 'evaluation').glob('update_*/summary.json')):
            item = json.loads(path.read_text())
            rows.append({'model': size, **{k: item[k] for k in (
                'update', 'successes', 'tasks', 'success_rate', 'reasoning_open_rate',
                'reasoning_complete_rate', 'strict_protocol_rate', 'native_action_valid_rate')}})
    if not rows:
        raise RuntimeError('No real validation results exist yet')
    with (args.out / 'seen140.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Wrote seen140.csv. Install matplotlib to draw the curve.')
        return
    for size in ('1.5b', '3b'):
        selected = sorted((r for r in rows if r['model'] == size), key=lambda x: x['update'])
        if selected:
            plt.plot([r['update'] for r in selected], [100*r['success_rate'] for r in selected],
                     marker='o', label='SDAR Qwen2.5-' + size)
    plt.xlabel('Training updates')
    plt.ylabel('ALFWorld seen140 success (%)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / 'seen140.png', dpi=180)
    print('Wrote seen140.csv and seen140.png from actual validations.')


if __name__ == '__main__':
    main()
