"""Plot total audit resources from the frozen planner replication, not PDFs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'code/oral_planner_v6/results/summary.json'
rows = json.loads(source.read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'mathtext.fontset': 'stix'})
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True)
fig.subplots_adjust(left=.12, right=.985, top=.86, bottom=.27, wspace=.36)
y = np.arange(len(rows))
labels = ['{} / {:.0%}'.format(r['law'], r['target']) for r in rows]
for ax, field, label, xmax, step in [
    (axes[0], 'mean_total_anchors', '(a) Total anchors (thousands)', 55, 10),
    (axes[1], 'mean_total_evaluations', '(b) Paired-loss evaluations (thousands)', 440, 100)]:
    for key, name, col, marker, dy in [
        ('valid', 'Pilot guarantee', '#0072B2', 'o', -.10),
        ('plugin', 'Plug-in', '#D55E00', 's', .10)]:
        vals = np.array([r[key][field] for r in rows])/1000
        ax.plot(vals, y+dy, linestyle='none', marker=marker, ms=4.2,
                color=col, label=name)
    pilot = 2.048 if field == 'mean_total_anchors' else 16.384
    ax.axvline(pilot, color='#777777', ls=':', lw=.8)
    ax.set(xlim=(0, xmax), xlabel=label, ylim=(5.6, -.6))
    ax.set_xticks(np.arange(0, xmax, step))
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.grid(axis='x', color='#dddddd', linewidth=.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Law / target power')
fig.text(.55, .95, 'All pilots are charged, including infeasible plans',
         ha='center', va='center', fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .005),
           ncol=2, frameon=False)
dest = ROOT / 'experiments/oral_planner_v6'; dest.mkdir(exist_ok=True, parents=True)
for ext in ['pdf', 'svg', 'png']:
    fig.savefig(str(dest/('costs.'+ext)), dpi=220)
plt.close(fig)
(ROOT/'research/cost_figure_provenance.json').write_text(json.dumps({
    'input': str(source.relative_to(ROOT)),
    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'estimand': 'Mean pilot-plus-final cost across all 24 pilots per law and target.',
    'uncertainty': 'Descriptive mean only; no confidence interval is implied.',
    'pilot': {'anchors': 2048, 'paired_loss_evaluations': 16384},
    'plot': 'Point positions use linear resource axes; dotted lines show the pilot floor.'
}, indent=2), encoding='utf-8')
print('Rendered costs.pdf/svg/png from summary.json.')
