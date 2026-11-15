"""Score the new pooling ensemble on the three existing, unchanged paper scopes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gait_from_scratch_evaluation import binary_metrics

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'deliverables/Theia_Selected_Gait_Paper_2026-11-15'
WEB=ROOT/'results/gait_selected_paper_20261115'
STAGE=ROOT/'results/gait_improvement_refinement_20261115'
out=PAPER/'evidence/pooling_table2';out.mkdir(exist_ok=True)
old=np.load(WEB/'evaluation_arrays.npz');new=np.load(STAGE/'geometry_set_ensemble_test.npz')
assert np.array_equal(old['indices'],new['indices'])
rows=json.loads((WEB/'window_results.json').read_text());assert len(rows)==len(old['indices'])
assert np.array_equal(old['indices'],[r['index'] for r in rows])
previous=json.loads((WEB/'report_results.json').read_text())
ensemble=next(e for e in json.loads((STAGE/'results.json').read_text())['ensembles'] if e['condition']=='geometry_set')
lookup={int(i):j for j,i in enumerate(old['indices'])}
groups={k:np.array([lookup[int(i)] for i in ids]) for k,ids in [('all_test',old['indices']),('quality',old['quality_indices']),('showcase',old['showcase_indices'])]}
common=[j for j,d in enumerate(previous['showcase']['labels']) if d['ap'] is not None]
p=new['probabilities'];y=old['y'];m=old['mask'];scores={}
for name,ix in groups.items():
    a=y[ix];b=p[ix];mask=m[ix];yy=a[mask].astype(bool);pp=b[mask]>=.5
    tp=int((yy&pp).sum());fp=int((~yy&pp).sum());fn=int((yy&~pp).sum());tn=int((~yy&~pp).sum())
    per=[dict(label=previous['all_test']['labels'][j]['label'],ap=binary_metrics(a[:,j],b[:,j],mask[:,j])['average_precision'],positive=int(((a[:,j]==1)&mask[:,j]).sum()),negative=int(((a[:,j]==0)&mask[:,j]).sum())) for j in range(a.shape[1])]
    aps=[v['ap'] for v in per if v['ap'] is not None];r=[rows[i] for i in ix]
    scores[name]=dict(windows=len(ix),clips=len({x['clip_id'] for x in r}),people=len({x['subject'] for x in r}),decisions=len(yy),positive=int(yy.sum()),negative=int((~yy).sum()),
        macro_ap=float(np.mean(aps)),ap_classes=len(aps),common_class_macro_ap=float(np.mean([per[j]['ap'] for j in common])),
        micro_accuracy=(tp+tn)/len(yy),micro_precision=tp/(tp+fp) if tp+fp else 0.,micro_recall=tp/(tp+fn),micro_f1=2*tp/(2*tp+fp+fn),balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),always_negative_accuracy=(tn+fp)/len(yy),tp=tp,fp=fp,fn=fn,tn=tn,per_class=per)
    if name!='showcase':
        for key in ['macro_ap','tp','fp','fn','tn']:assert np.allclose(scores[name][key],ensemble[name][key],atol=1e-12)
result=dict(created_at=datetime.now(timezone.utc).isoformat(),model='geometry_set three-seed ensemble',members=ensemble['members'],threshold=.5,
    selection='Highest observed original-test AP among five new families; not the validation-selected family.',
    scopes='Reuse original all-test, quality and historical TCN-selected showcase indices exactly. No new showcase selection.',
    old_arrays_sha256=hashlib.sha256((WEB/'evaluation_arrays.npz').read_bytes()).hexdigest(),new_predictions_sha256=hashlib.sha256((STAGE/'geometry_set_ensemble_test.npz').read_bytes()).hexdigest(),
    common_classes=[previous['showcase']['labels'][j]['label'] for j in common],clinical_validation=False,**scores)
(out/'results.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(out/'evaluation_arrays.npz',**{k:old[k] for k in old.files if k!='probabilities'},probabilities=p)

table=[r'\begin{table}[htbp]\centering\small',r'\caption{Statistical-pooling ensemble on the unchanged evaluation scopes, at threshold 0.5. All test includes 181 inputs from 28 clips; 124 windows from 16 clips contribute the 607 decisions. The retained showcase was selected using the earlier TCN and remains outcome-selected; it was not reselected for pooling.}\label{tab:results}',r'\begin{tabular}{lrrr}\toprule',r' & All test & Quality screened & Retained showcase\\\midrule']
specs=[('Clips / people',lambda z:f"{z['clips']} / {z['people']}"),('Windows / decisions',lambda z:f"{z['windows']} / {z['decisions']}"),('Positive / negative',lambda z:f"{z['positive']} / {z['negative']}"),('Macro AP (classes)',lambda z:f"{z['macro_ap']:.3f} ({z['ap_classes']})"),('Common-three-class AP',lambda z:f"{z['common_class_macro_ap']:.3f}"),('Accuracy',lambda z:f"{z['micro_accuracy']:.3f}"),('Always-negative accuracy',lambda z:f"{z['always_negative_accuracy']:.3f}"),('Precision / recall',lambda z:f"{z['micro_precision']:.3f} / {z['micro_recall']:.3f}"),('F1 / balanced accuracy',lambda z:f"{z['micro_f1']:.3f} / {z['balanced_accuracy']:.3f}"),('TP / FP / FN / TN',lambda z:f"{z['tp']} / {z['fp']} / {z['fn']} / {z['tn']}")]
for label,fn in specs:table.append(label+' & '+' & '.join(fn(scores[k]) for k in groups)+r'\\')
table += [r'\bottomrule\end{tabular}\end{table}']
(PAPER/'pooling_scope_table.tex').write_text('\n'.join(table)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(PAPER/'images'/f'{name}.pdf',bbox_inches='tight');fig.savefig(PAPER/'images'/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
titles=['All test','Quality screened','Retained showcase'];colors=['#72899a','#2b8290','#b26a28']
for ax,key,title in zip(axs,['macro_ap','micro_accuracy'],['Macro average precision','Decision agreement']):
    bars=ax.bar(titles,[scores[k][key] for k in groups],color=colors,width=.6);ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
    ax.set(ylim=(0,1.15),title=title,ylabel='Score');ax.tick_params(axis='x',labelsize=8)
axs[0].text(.02,.98,'8 / 7 / 3 supported classes',transform=axs[0].transAxes,va='top',fontsize=8)
axs[1].text(.02,.98,'607 / 482 / 102 decisions',transform=axs[1].transAxes,va='top',fontsize=8)
save(fig,'pooling_selection_effect')

fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained');rng=np.random.default_rng(20261115)
for who,col in [('ljmu360_P5','#248b93'),('ljmu360_P9','#805db0'),('progait_1','#c47c38')]:
    ids=[i for i in groups['quality'] if rows[i]['subject']==who]
    acc=[float(((p[i]>=.5)==y[i].astype(bool))[m[i]].mean()) for i in ids]
    cov=[rows[i]['min_joint_coverage'] for i in ids]
    ax.scatter(np.array(cov)+rng.uniform(-.002,.002,len(ids)),acc,s=17,alpha=.4,color=col,label=who+' windows')
    clips=[]
    for cid in sorted({rows[i]['clip_id'] for i in ids}):
        ii=[i for i in ids if rows[i]['clip_id']==cid];cm=m[ii];correct=((p[ii]>=.5)==y[ii].astype(bool))[cm];ca=float(correct.mean());cv=np.mean([rows[i]['min_joint_coverage'] for i in ii]);clips.append(ca)
        ax.scatter(cv,ca,s=65,marker='D',facecolors='none',edgecolors=col,linewidths=1.5)
    if ids:ax.scatter(np.mean(cov),np.mean(clips),s=190,marker='*',color=col,edgecolors='white',linewidths=.8,zorder=4)
ax.set(xlabel='Minimum landmark coverage (display jitter for windows)',ylabel='Agreement with automatic criteria',ylim=(-.05,1.08),title='Statistical pooling: 91 windows, 14 clips, 3 people');ax.legend(fontsize=8,loc='lower left')
save(fig,'pooling_coverage_scatter')
shutil.copy2(Path(__file__),PAPER/'code'/Path(__file__).name)
print(json.dumps({k:{m:v[m] for m in ['macro_ap','micro_accuracy','micro_precision','micro_recall','tp','fp','fn','tn']} for k,v in scores.items()},indent=2))
