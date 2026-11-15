"""Render web charts from the unchanged pooling Table 2 evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
R=Path(__file__).resolve().parents[1]
W=R/'results/gait_selected_paper_20261115'
P=R/'deliverables/Theia_Selected_Gait_Paper_2026-11-15'
r=json.loads((P/'evidence/pooling_table2/results.json').read_text())
for f in (W/'fonts').glob('*.ttf'): fm.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'IBM Plex Mono','font.size':9,'text.color':'#1d1f23','axes.labelcolor':'#3a3d43','axes.edgecolor':'#d4d4d6','xtick.color':'#6b6e75','ytick.color':'#6b6e75','axes.spines.top':False,'axes.spines.right':False,'grid.color':'#d4d4d6','grid.linewidth':.5,'axes.titlesize':11,'axes.titleweight':'normal','figure.facecolor':'white','savefig.facecolor':'white'})
keys=['all_test','quality','showcase']; labels=['All test','Quality','Selected']
def save(fig,name):
    fig.savefig(W/'figures'/f'workbench_{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.8,4)); x=np.arange(3)
for off,key,col,label in [(-.12,'macro_ap','#0062b3','Macro AP'),(.12,'micro_accuracy','#3a3d43','Decision agreement')]:
    vals=[r[k][key] for k in keys];ax.plot(x+off,vals,'o',ms=5,color=col,label=label)
    for xx,v in zip(x+off,vals):ax.annotate(f'{v:.3f}',(xx,v),xytext=(0,10 if off<0 else -15),textcoords='offset points',ha='center',fontsize=9,color=col)
ax.set(xticks=x,xticklabels=labels,ylim=(.45,1.12),xlim=(-.45,2.5),ylabel='Score',title='POOLING / FIXED EVALUATION SUBSETS');ax.grid(axis='y');ax.legend(loc='lower right',frameon=False);fig.tight_layout();save(fig,'selection')
fig,axes=plt.subplots(1,3,figsize=(6.8,4))
for ax,k,label in zip(axes,keys,labels):
    s=r[k]; v=np.array([[s['tn'],s['fp']],[s['fn'],s['tp']]])
    ax.imshow(np.zeros((2,2)),cmap='Greys',vmin=0,vmax=1)
    for (i,j),n in np.ndenumerate(v):ax.text(j,i,str(n),ha='center',va='center',fontsize=17,color='#1d1f23')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No','Flag'],yticklabels=['No','Flag'],xlabel='Prediction',title=label)
    ax.set_xticks([-.5,.5,1.5],minor=True);ax.set_yticks([-.5,.5,1.5],minor=True);ax.grid(which='minor');ax.tick_params(which='both',length=0)
axes[0].set_ylabel('Automatic target');fig.suptitle('ERROR COUNTS / THRESHOLD 0.5',y=.88,fontsize=11);fig.tight_layout();save(fig,'confusion')
fig,ax=plt.subplots(figsize=(6.8,4.3)); usable=[c for c in r['all_test']['per_class'] if c['ap'] is not None]
for i,c in enumerate(usable):
    ax.plot([0,c['ap']],[i,i],color='#d4d4d6',lw=1);ax.plot(c['ap'],i,'o',ms=4,color='#0062b3');ax.text(c['ap']+.025,i,f"{c['ap']:.3f}",va='center',fontsize=8)
names=[c['label'].replace('_',' ').replace('rom95','ROM').replace('p95','bend P95').replace('asymmetry','asym.') for c in usable]
ax.set(yticks=np.arange(len(usable)),yticklabels=names,xlim=(0,1.15),xlabel='Full-test average precision',title='POOLING / SUPPORTED CLASSES');ax.invert_yaxis();ax.grid(axis='x');fig.tight_layout();save(fig,'per_class')
fig,ax=plt.subplots(figsize=(6.8,4));old=json.loads((W/'report_results.json').read_text())
# Earlier TCN reference is preserved in page_data; read its exact stored values.
old=json.loads((W/'page_data.json').read_text())['results']['all_test']
for label,col,vals in [('Earlier TCN','#6b6e75',[old['macro_ap'],old['micro_recall']]),('Statistical pooling','#0062b3',[r['all_test']['macro_ap'],r['all_test']['micro_recall']])]:
    ax.plot([0,1],vals,'o-',color=col,lw=1,ms=5,label=label)
    for x,v in enumerate(vals):ax.annotate(f'{v:.3f}',(x,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9,color=col)
ax.set(xticks=[0,1],xticklabels=['Macro AP','Recall at 0.5'],xlim=(-.25,1.25),ylim=(.3,.85),title='RANKING / SENSITIVITY TRADEOFF',ylabel='Score');ax.grid(axis='y');ax.legend(frameon=False,loc='upper left');fig.tight_layout();save(fig,'tradeoff')
print('Four web figures generated from frozen evidence.')
rows=json.loads((W/'window_results.json').read_text()); results=json.loads((W/'report_results.json').read_text())
quality=[w for w in rows if w['quality_pass'] and w['labels']]
fig,ax=plt.subplots(figsize=(6.8,4.3)); rng=np.random.default_rng(20261115)
for who,col in [('ljmu360_P5','#1d1f23'),('ljmu360_P9','#6b6e75'),('progait_1','#b9bbc0')]:
    ww=[w for w in quality if w['subject']==who]; cc=[c for c in results['quality_clips'] if c['subject']==who]
    ax.scatter([w['min_joint_coverage']+rng.uniform(-.002,.002) for w in ww],[np.mean([l['correct'] for l in w['labels']]) for w in ww],s=13,alpha=.5,color=col,label=who)
    for c in cc:
        q=[w for w in ww if w['clip_id']==c['clip_id']];ax.scatter(np.mean([w['min_joint_coverage'] for w in q]),c['micro_accuracy'],s=55,marker='D',facecolors='none',edgecolors=col,linewidths=1)
    if ww: ax.scatter(np.mean([w['min_joint_coverage'] for w in ww]),np.mean([c['micro_accuracy'] for c in cc]),s=170,marker='*',color=col,edgecolors='white',linewidths=.6,zorder=4)
ax.set(xlabel='Minimum landmark coverage (display jitter)',ylabel='Agreement with automatic criteria',ylim=(-.05,1.08),title='TCN / 91 WINDOWS · 14 CLIPS · 3 PEOPLE');ax.grid(alpha=.5);ax.legend(loc='lower left',fontsize=8,frameon=False);fig.tight_layout();save(fig,'coverage')
fig,ax=plt.subplots(figsize=(6.8,4.3))
for arch,col in [('tcn','#0062b3'),('motionbert','#6b6e75')]:
    cc=[c for c in results['candidates'] if '_'+arch+'_' in c['run']];ax.scatter([c['validation_ap'] for c in cc],[c['quality']['macro_ap'] for c in cc],c=col,s=22,label=arch.upper())
best=next(c for c in results['candidates'] if c['run']==results['model']);ax.scatter(best['validation_ap'],best['quality']['macro_ap'],s=100,facecolors='none',edgecolors='#1d1f23',lw=1);ax.annotate('Validation selection',(best['validation_ap'],best['quality']['macro_ap']),xytext=(-130,15),textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#1d1f23'},fontsize=8)
ax.set(xlabel='Stored validation macro AP',ylabel='Quality-screened test macro AP',title='EARLIER SEARCH / 22 RUNS');ax.grid(alpha=.5);ax.legend(frameon=False);fig.tight_layout();save(fig,'models')
