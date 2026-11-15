"""Add immutable experiment evidence and publication figures to the manuscript."""
from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'deliverables/Theia_Selected_Gait_Paper_2026-11-15'
STAGES=[ROOT/'results/gait_improvement_20261115',ROOT/'results/gait_improvement_refinement_20261115']
E=PAPER/'evidence/improvement';E.mkdir(parents=True,exist_ok=True)
R=[json.loads((p/'results.json').read_text()) for p in STAGES]
ensembles=[e for r in R for e in r['ensembles']]
names={'raw':'Raw TCN','geometry':'Geometry TCN','geometry_reverse':'Geometry + reversal','geometry_slow':'Geometry, slower','geometry_set':'Statistical pooling'}
colors={'raw':'#5483b5','geometry':'#bc7a31','geometry_reverse':'#a36c97','geometry_slow':'#208c83','geometry_set':'#496c37'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(PAPER/'images'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(PAPER/'images'/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)

for stage,src in enumerate(STAGES,1):
    dst=E/f'stage{stage}';dst.mkdir(exist_ok=True)
    for filename in ['protocol.json','selection.json','evaluation_protocol.json','results.json','training_summary.json','latency.json']:
        shutil.copy2(src/filename,dst/filename)
    for file in src.glob('*_ensemble_*.npz'):shutil.copy2(file,dst/file.name)
    for run in R[stage-1]['runs']:
        rd=dst/'runs'/run['id'];rd.mkdir(parents=True,exist_ok=True)
        for name in ['training.json','history.json','validation_predictions.npz','test_predictions.npz','evaluation.json']:
            shutil.copy2(src/'runs'/run['id']/name,rd/name)
for name in ['selection_rule.json','combined_selection.json','summary.json']:
    shutil.copy2(ROOT/'results/gait_improvement_report_20261115'/name,E/name)
for name in ['original','fold_1','fold_2','fold_3','fold_4']:
    src=ROOT/'results/gait_experiments_20261115_0502/splits'/name;dst=E/'splits'/name;dst.mkdir(parents=True,exist_ok=True)
    for file in src.iterdir():
        if file.is_file():shutil.copy2(file,dst/file.name)

fig,axs=plt.subplots(1,2,figsize=(8,3.8),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
ax=axs[0]
for e in ensembles:
    c=e['condition'];rs=[r for report in R for r in report['runs'] if r['split']=='original' and r['condition']==c]
    ax.scatter([r['best_validation_ap'] for r in rs],[r['all_test']['macro_ap'] for r in rs],s=32,color=colors[c],alpha=.6)
    ax.scatter(e['validation']['macro_ap'],e['all_test']['macro_ap'],marker='*',s=100,color=colors[c],label=names[c])
base=R[0]['baseline'];ax.scatter(base['validation']['macro_ap'],base['all_test']['macro_ap'],marker='X',s=65,color='#142e3d',label='Existing model')
ax.plot([.2,1],[.2,1],'--',lw=.8,color='#a4acb0');ax.set(xlim=(.25,.95),ylim=(.15,.9),xlabel='Validation macro AP',ylabel='Original-test macro AP',title='(a) Individual seeds and ensembles');ax.grid(alpha=.15)
ax.legend(fontsize=9,loc='upper left')
ax=axs[1]
for c in names:
    rs=[r for report in R for r in report['runs'] if r['condition']==c]
    ax.plot(range(1,5),[next(r['all_test']['macro_ap'] for r in rs if r['split']==f'fold_{i}') for i in range(1,5)],'o-',color=colors[c],markersize=4)
ax.set(xticks=range(1,5),ylim=(0,.8),xlabel='Frozen participant fold',ylabel='Fold test macro AP',title='(b) Single-seed fold comparisons');ax.grid(alpha=.15)
save(fig,'improvement_transfer')

old=np.load(PAPER/'evidence/evaluation_arrays.npz');pool=np.load(STAGES[1]/'geometry_set_ensemble_test.npz');assert np.array_equal(old['indices'],pool['indices'])
y=old['y'];m=old['mask'];p=pool['probabilities'];bp=old['probabilities']
fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
ax=axs[0]
for target,color,label in [(0,'#5483b5','Automatic target 0'),(1,'#cf8a3b','Automatic target 1')]:
    keep=m&(y==target);ax.scatter(bp[keep],p[keep],s=9,alpha=.45,color=color,label=label,rasterized=True)
ax.plot([0,1],[0,1],'--',lw=.8,color='#8b979f');ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Existing model score',ylabel='Statistical-pooling score',title='(a) All 607 evaluable decisions');ax.legend(fontsize=9,loc='upper left');ax.grid(alpha=.15)
ax=axs[1];x=np.arange(2);width=.25;new=next(e for e in ensembles if e['condition']=='geometry_set')
for i,(key,label,color) in enumerate([('tp','True positives','#358c80'),('fp','False positives','#cf8a3b'),('fn','Missed positives','#ae6375')]):
    b=ax.bar(x+(i-1)*width,[base['all_test'][key],new['all_test'][key]],width,label=label,color=color);ax.bar_label(b,padding=3,fontsize=9)
ax.set_xticks(x,['Existing model','Statistical pooling']);ax.set(ylabel='Binary decisions',title='(b) Fixed threshold 0.5',ylim=(0,205));ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'improvement_decisions')

lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Original-split comparison against the same automatic targets. New rows average three seeds; the existing model is one previously selected checkpoint. Bold marks the validation-selected family, not the largest test AP. Precision, recall and F1 pool supported binary decisions at threshold 0.5.}\label{tab:improvement}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrrr}\toprule',r'Model & Val AP & Test AP & Quality AP & Precision & Recall & F1 & FP / FN\\\midrule']
for name,e in [('Existing model',base)]+[(names[e['condition']],e) for e in ensembles]:
    if e.get('condition')=='geometry':name=r'\textbf{'+name+'}'
    t=e['all_test'];values=[name,f"{e['validation']['macro_ap']:.3f}",f"{t['macro_ap']:.3f}",f"{e['quality']['macro_ap']:.3f}",f"{t['precision']:.3f}",f"{t['recall']:.3f}",f"{t['f1']:.3f}",f"{t['fp']} / {t['fn']}"]
    lines.append(' & '.join(values)+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}']
(PAPER/'improvement_table.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{longtable}{llrrrrr}',r'\caption{All 35 additional training runs. Selection used validation macro AP. Fold scores have different class support and are descriptive.}\label{tab:allruns}\\',r'\toprule Split & Family & Seed suffix & Epoch & Val AP & Test AP & Seconds\\\midrule\endfirsthead',r'\toprule Split & Family & Seed suffix & Epoch & Val AP & Test AP & Seconds\\\midrule\endhead',r'\bottomrule\endfoot']
for report in R:
    for r in report['runs']:
        vals=[r['split'].replace('_',' '),names[r['condition']],str(r['seed'])[-2:],str(r['best_epoch']),f"{r['best_validation_ap']:.3f}",f"{r['all_test']['macro_ap']:.3f}",f"{r['seconds']:.1f}"]
        lines.append(' & '.join(vals)+r'\\')
lines.append(r'\end{longtable}')
(PAPER/'improvement_runs.tex').write_text('\n'.join(lines)+'\n')
for name in ['gait_kinematic_model.py','gait_statistical_pool_model.py','test_gait_kinematic_model.py','run_gait_improvement.py','run_gait_improvement_refinement.py','evaluate_gait_improvement.py','gait_from_scratch_model.py','gait_from_scratch_training.py','gait_from_scratch_sampling.py','gait_from_scratch_evaluation.py','gait_assessment_data.py','build_gait_paper_improvement.py']:
    shutil.copy2(ROOT/'scripts'/name,PAPER/'code'/name)
print('Added all 35 run records, frozen predictions/splits, two publication figures and generated tables.')
