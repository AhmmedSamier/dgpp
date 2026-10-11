from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).parent
dest = out / 'decode-hash-profile-figures'
dest.mkdir(exist_ok=True)
audit = json.loads((out / 'decode-wide-final-matched-audit.json').read_text())
classes = ('prose', 'code', 'json', 'math', 'chat')
cells = [(kind, c) for kind in classes for c in (1, 2, 4, 8)]
values = np.array([[next(r['gain_pct'] for r in audit['decode'] if
                        r['world'] == world and (r['kind'], r['concurrency']) == cell)
                    for cell in cells] for world in (1, 2, 4)])
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout='constrained',
                         gridspec_kw={'height_ratios': [1, 1.35]})
im = axes[0].imshow(values, aspect='auto', cmap='YlGnBu', vmin=0, vmax=max(45, values.max()))
axes[0].set(yticks=range(3), yticklabels=['1 node', '2 nodes', '4 nodes'],
            xticks=range(20), xticklabels=[f'{kind}\nC{c}' for kind, c in cells],
            title='Decode gain versus the stronger FP8 bracket median, %')
for y in range(3):
    for x in range(20):
        v = values[y, x]
        axes[0].text(x, y, f'{v:.1f}', ha='center', va='center',
                     color='white' if v > 24 else '#111111', fontsize=9)
for x in (3.5, 7.5, 11.5, 15.5):
    axes[0].axvline(x, color='white', lw=2)
fig.colorbar(im, ax=axes[0], label='NVFP4 throughput gain (%)', fraction=.025)
rows = [next(r for r in audit['decode'] if r['world'] == 4 and r['kind'] == kind
             and r['concurrency'] == 1) for kind in classes]
x = np.arange(5)
for offset, label, color, rates in [
    (-.18, 'FP8 (stronger bracket)', '#2a6fbb', [max(r['fp8a_tps'], r['fp8b_tps']) for r in rows]),
    (.18, 'NVFP4', '#e68432', [r['nvfp4_tps'] for r in rows]),
]:
    axes[1].bar(x + offset, rates, .36, label=label, color=color)
    for i, rate in enumerate(rates):
        axes[1].text(i + offset, rate + 2, f'{rate:.1f}', ha='center')
axes[1].set(xticks=x, xticklabels=[c.upper() if c == 'json' else c.title() for c in classes],
            ylabel='Engine decode tokens/s', title='Four nodes, one active request',
            ylim=(0, max(r['nvfp4_tps'] for r in rows) * 1.22))
axes[1].legend(loc='upper left')
fig.suptitle('Final uninstrumented Qwen3.8-27B comparison — d12e7e9f', fontsize=15)
fig.supxlabel('Three repetitions per cell; identical input prompts, native greedy outputs. Decode excludes prefill.', fontsize=10)
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(dest / f'decode-serving.{ext}', dpi=150)
