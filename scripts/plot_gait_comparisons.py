"""Standalone research figures from the same JSON shown in the local report."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gait_review_figures import build_review_figures
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gait_metrics_20261114';DEST=OUT/'figures';DEST.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
a=json.loads((OUT/'model_benchmark/results.json').read_text());b=json.loads((OUT/'model_benchmark/0.4b/results.json').read_text())
models=[a['models']['5b'],a['models']['1b'],b['models']['0.4b']];labels=['5B','1B (selected)','0.4B'];colors=['#8b9cae','#278d87','#c49a68']
fig,axes=plt.subplots(1,3,figsize=(11,3.7))
for ax,vals,label in zip(axes,[[m['fps'] for m in models],[m['peak_cuda_allocated_GB'] for m in models],[100*m['agreement']['feet']['pck_005'] for m in models]],['Inference frames/s','Peak allocated GPU memory (GB)','Feet PCK@0.05 (%)']):
 bars=ax.bar(labels,vals,color=colors);ax.set_ylabel(label);ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_ylim(0,max(vals)*1.22 if 'PCK' not in label else 100)
fig.suptitle('Sapiens2 local engineering benchmark · RTX 4090 Laptop GPU',fontsize=13)
fig.text(.5,.01,'180 matched quality frames; publisher-assisted reference. 5B timing: 30 frames. Not clinical accuracy.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(DEST/'model_benchmark.png',dpi=170);fig.savefig(DEST/'model_benchmark.svg');plt.close(fig)
r=json.loads((OUT/'comparison_results.json').read_text());files=[]
for mode in r['modes']:
 results=[x for x in r['results'] if x['mode']==mode and x['sensitivity']=='base' and x['stratum']['environment']=='overground']
 fig,axes=plt.subplots(5,2,figsize=(12,13));
 for ax,item in zip(axes.ravel(),results):
  for i,(name,color) in enumerate([('with_amputation','#bd803d'),('without_amputation','#278d87')]):
   g=item['groups'][name];v=[u['value'] for u in g['units']]
   if v:
    jitter=np.linspace(-.12,.12,len(v));ax.scatter(np.full(len(v),i)+jitter,v,s=22,color=color,alpha=.65)
    ax.plot([i,i],[g['q25'],g['q75']],lw=5,color=color);ax.scatter([i],[g['median']],s=55,color=color,edgecolor='white',zorder=3)
   else:ax.text(i,.5,'Unavailable',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
  ga,gb=[item['groups'][n] for n in ['with_amputation','without_amputation']]
  def unit(g):return 'people' if g['participant_count'] is not None else 'source blocks'
  ax.set_xticks([0,1],[f'With\n{ga["unit_count"]} {unit(ga)}',f'Without\n{gb["unit_count"]} {unit(gb)}'])
  ax.set_xlim(-.4,1.4);ax.set_ylabel(item['outcome']['unit']);ax.set_title(item['outcome']['label'],fontsize=9,wrap=True)
  ax.grid(axis='y',alpha=.15)
 title='Unsupported walking: unknown identities use source blocks' if mode.startswith('unsupported') else 'Known subjects: ProGait guarding context differs from reference'
 fig.suptitle(f'{r["status"].upper()} · {title}',fontsize=13)
 fig.text(.5,.01,'Projected observations only; dots = aggregation units; thick line = IQR; large dot = median. No clinical or causal interpretation.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.03,1,.96));stem='overground_'+mode
 for ext in ['png','svg']:fig.savefig(DEST/(stem+'.'+ext),dpi=170)
 plt.close(fig);files.append(stem)
(DEST/'manifest.json').write_text(json.dumps(dict(status=r['status'],comparison_updated_at=r['updated_at'],files=['model_benchmark']+files,formats=['png','svg']),indent=2))
print('Saved benchmark and two comparison figures ('+r['status']+')')
review_figures=build_review_figures(r,json.loads((OUT/'catalog.json').read_text()),DEST)
print('Saved participant scatterplots, method sensitivity and source coverage: '+str(len(review_figures['scatter_points']))+' paired participants')
