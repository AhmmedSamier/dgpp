from pathlib import Path
import csv,collections,json,statistics,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path(__file__).parent;prefix=sys.argv[1] if len(sys.argv)>1 else 'decode-cuda-only';dest=out/('decode-profile-figures' if prefix=='decode-cuda-only' else prefix+'-figures');dest.mkdir(exist_ok=True)
a=json.loads((out/(prefix+'-analysis.json')).read_text())['ranks'];byte=json.loads((out/'decode-byte-audit.json').read_text());colors=['#2a6fbb','#e68432']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
f,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
phases=['Draft + commit','Attention + norms','First 56 MLPs','Last 8 MLPs','Target collectives','Head + pick tail'];pc=['#777777','#69a9d9','#f3ab5a','#ac772e','#aa79ba','#76af87'];bottom=np.zeros(2)
values=[]
for q in ('fp8','nvfp4'):
 r=next(r for r in a if r['quant']==q and r['rank']==0);s=r['summary'];ls=r['layers'];values.append([s['draft_ms']['mean']+s['commit_ms']['mean'],sum(x['us']['mean'] for x in ls if x['phase']=='attention')/1000,sum(x['us']['mean'] for x in ls if x['phase']=='mlp' and x['layer']<56)/1000,sum(x['us']['mean'] for x in ls if x['phase']=='mlp' and x['layer']>=56)/1000,s['target_collectives_ms']['mean'],s['head_pick_tail_ms']['mean']])
for i,phase in enumerate(phases):
 v=np.array([row[i] for row in values]);axs[0,0].bar(['FP8','NVFP4'],v,bottom=bottom,label=phase,color=pc[i]);bottom+=v
for i,v in enumerate(bottom):axs[0,0].text(i,v+.5,f'{v:.2f} ms',ha='center')
axs[0,0].set(title='Decode pass: all-rank CUDA capture, rank 0',ylabel='Milliseconds per pass',ylim=(0,65));axs[0,0].legend(fontsize=8,ncol=2,loc='upper left')
for qi,q in enumerate(('fp8','nvfp4')):
 r=next(r for r in a if r['quant']==q and r['rank']==0);ls=r['layers'];x=range(64);y=[sum(z['us']['mean'] for z in ls if z['phase']=='mlp' and z['layer']==i) for i in x];axs[0,1].plot(x,y,label=q.upper(),color=colors[qi])
axs[0,1].axvline(55.5,color='#777',ls=':');axs[0,1].set(title='MLP time by layer, including local launch gaps',xlabel='Layer',ylabel='Microseconds per pass');axs[0,1].legend()
for qi,q in enumerate(('fp8','nvfp4')):
 vals=[]
 for op in ('gate_up','down'):
  times=[]
  for rank in range(4):
   rows=list(csv.DictReader((out/f'decode-weight-rank{rank}.csv').open()));times.append(statistics.median(float(r['us']) for r in rows if r['rows']=='8' and r['operation']==op and r['quant']==('fp8_block' if q=='fp8' else q)))
  vals.append(statistics.mean(times))
 axs[1,0].bar(np.arange(2)+(.18 if qi else -.18),vals,.36,label=q.upper(),color=colors[qi])
 for i,v in enumerate(vals):axs[1,0].text(i+(.18 if qi else -.18),v+3,f'{v:.1f}',ha='center')
axs[1,0].set(xticks=range(2),xticklabels=['Gate + up','Down'],ylabel='Microseconds per launch',title='TP4 shapes, synthetic weights; 8 rows, 8 copies rotated',ylim=(0,255));axs[1,0].legend()
bottom=np.zeros(2)
for key,label,c in [('first56_mlp','First 56 MLPs','#f3ab5a'),('last8_mlp','Last 8 MLPs','#ac772e'),('attention_and_norms','Attention + norms','#69a9d9')]:
 v=np.array([m['nominal_source_bytes_per_rank'][key]/1e9 for m in byte['models']]);axs[1,1].bar(['FP8','NVFP4'],v,bottom=bottom,label=label,color=c);bottom+=v
for i,v in enumerate(bottom):axs[1,1].text(i,v+.06,f'{v:.2f} GB',ha='center')
axs[1,1].set(title='Transformer checkpoint bytes / 4 ranks (approximation)',ylabel='Decimal GB; excludes embedding, head and drafter',ylim=(0,7.5));axs[1,1].legend(fontsize=8)
f.suptitle('PR 95 decode investigation — '+('pre-hash candidate' if prefix=='decode-cuda-only' else 'final d12e7e9f candidate')+' on four GB10 nodes',fontsize=15)
f.savefig(dest/'decode-breakdown.svg');f.savefig(dest/'decode-breakdown.pdf');f.savefig(dest/'decode-breakdown.png',dpi=150);plt.close(f)
f,axes=plt.subplots(2,1,figsize=(13,5.5),layout='constrained',sharex=True)
for qi,q in enumerate(('fp8','nvfp4')):
 heat=np.array([[sum(x['us']['mean'] for x in r['layers'] if x['layer']==i and x['phase'].endswith('collective')) for i in range(64)] for r in a if r['quant']==q]);im=axes[qi].imshow(heat,aspect='auto',vmin=100,vmax=240,cmap='magma');axes[qi].set(yticks=range(4),yticklabels=['Rank '+str(r) for r in range(4)],title=q.upper()+' attention + MLP collectives per layer');f.colorbar(im,ax=axes[qi],label='Microseconds / pass')
axes[-1].set_xlabel('Layer');f.savefig(dest/'decode-collectives.svg');f.savefig(dest/'decode-collectives.png',dpi=150)
(out/('decode-phase-totals.json' if prefix=='decode-cuda-only' else prefix+'-phase-totals.json')).write_text(json.dumps(dict(phases=phases,fp8=values[0],nvfp4=values[1]),indent=2))
