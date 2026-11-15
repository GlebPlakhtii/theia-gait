"""Create a local comparison report without altering the existing paper/demo."""
from pathlib import Path
import json,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STAGES=[ROOT/'results/gait_improvement_20261115',ROOT/'results/gait_improvement_refinement_20261115']
OUT=ROOT/'results/gait_improvement_report_20261115'
NAMES={'raw':'Raw-pose TCN','geometry':'Geometry TCN','geometry_reverse':'Geometry + reversal','geometry_slow':'Geometry · slower training','geometry_set':'Statistical pooling'}
COLORS={'raw':'#5d85c9','geometry':'#d18337','geometry_reverse':'#ab6b9b','geometry_slow':'#1b9f9a','geometry_set':'#406b3c'}
def esc(x):return html.escape(str(x))
def fmt(x):return '—' if x is None else f'{x:.3f}'

def main():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    reports=[json.loads((p/'results.json').read_text()) for p in STAGES]
    selections=[json.loads((p/'selection.json').read_text()) for p in STAGES]
    combined=json.loads((OUT/'combined_selection.json').read_text());chosen=combined['selected_condition']
    ensembles=[dict(e,stage=s) for s,r in enumerate(reports) for e in r['ensembles']]
    selected=next(e for e in ensembles if e['condition']==chosen);baseline=reports[0]['baseline']
    observed=max(ensembles,key=lambda e:e['all_test']['macro_ap'])
    runs=[dict(r,stage=s) for s,rep in enumerate(reports) for r in rep['runs']]
    plt.rcParams.update({'figure.dpi':140,'savefig.dpi':180,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.1))
    for c in NAMES:
        rr=[r for r in runs if r['condition']==c and r['split']=='original']
        ax.scatter([r['best_validation_ap'] for r in rr],[r['all_test']['macro_ap'] for r in rr],s=65,color=COLORS[c],alpha=.65,label=NAMES[c])
        e=next(e for e in ensembles if e['condition']==c)
        ax.scatter(e['validation']['macro_ap'],e['all_test']['macro_ap'],s=170,marker='*',color=COLORS[c],edgecolor='#152531')
    ax.scatter(baseline['validation']['macro_ap'],baseline['all_test']['macro_ap'],marker='X',s=110,color='#152531',label='Existing demo')
    ax.plot([0,1],[0,1],ls='--',color='#adb6be');ax.set(xlim=(.2,1),ylim=(.15,.9),xlabel='Validation macro AP',ylabel='Original test macro AP',title='Selection on validation does not ensure transfer')
    ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=2);fig.text(.13,.01,'Circles: individual seeds. Stars: three-seed ensembles. Same original split and targets.',fontsize=8);save(fig,'transfer')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    rows=[('Existing demo',baseline),('Validation-selected',selected),('Highest observed AP',observed)]
    for ax,scope,title in zip(axs,['all_test','quality'],['All test windows','Quality-screened windows']):
        x=np.arange(len(rows));width=.24
        for i,(metric,label,color) in enumerate([('tp','True positives','#348d80'),('fp','False positives','#d18337'),('fn','Missed positives','#ac5f70')]):
            bars=ax.bar(x+(i-1)*width,[r[scope][metric] for _,r in rows],width,label=label,color=color);ax.bar_label(bars,padding=3)
        ax.set_xticks(x,['Existing demo','Val-selected','Stat. pooling']);ax.set_title(title);ax.set_ylabel('Binary decisions against automatic labels');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[1].legend(fontsize=8);fig.suptitle('False-alert reduction and missed positives, together · threshold 0.5');save(fig,'decisions')
    fig,ax=plt.subplots(figsize=(8.5,4.8));folds=['fold_1','fold_2','fold_3','fold_4']
    for c in NAMES:
        values=[next(r['all_test']['macro_ap'] for r in runs if r['condition']==c and r['split']==f) for f in folds]
        ax.plot(range(1,5),values,'o-',color=COLORS[c],label=NAMES[c])
    ax.set(xticks=range(1,5),xlabel='Frozen participant fold',ylabel='Test macro AP',title='Four participant-separated folds · one matched seed per family',ylim=(0,1));ax.grid(alpha=.18);ax.legend(fontsize=8,ncol=2)
    fig.text(.13,.01,'Compare families within a fold. Enabled classes and reference fits vary between folds.',fontsize=8);save(fig,'folds')
    labels=baseline['all_test']['per_class'];fig,ax=plt.subplots(figsize=(9,5.5));supported=[j for j,c in enumerate(labels) if c['ap'] is not None];yy=np.arange(len(supported))
    ax.barh(yy-.18,[labels[j]['ap'] for j in supported],.34,label='Existing demo',color='#5d85c9')
    ax.barh(yy+.18,[selected['all_test']['per_class'][j]['ap'] for j in supported],.34,label='Validation-selected candidate',color=COLORS[chosen])
    ax.set_yticks(yy,[labels[j]['id'].replace('_',' ') for j in supported]);ax.invert_yaxis();ax.set(xlim=(0,1),xlabel='Average precision',title='Which regional criteria changed?');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15);save(fig,'per_class')
    d=np.load(ROOT/'results/gait_from_scratch_v1/dataset/dataset.npz');windows=json.loads((ROOT/'results/gait_from_scratch_v1/dataset/windows.json').read_text())
    old=np.load(ROOT/'results/gait_selected_paper_20261115/evaluation_arrays.npz');ids=old['indices'];mask=old['mask'];targets=old['y'];probs={'baseline':old['probabilities'].tolist()}
    for e in ensembles:
        a=np.load(STAGES[e['stage']]/f'{e["condition"]}_ensemble_test.npz');assert np.array_equal(ids,a['indices']);probs[e['condition']]=a['probabilities'].tolist()
    points=dict(selected=chosen,names=dict(baseline='Existing demo',**NAMES),classes=[x['id'] for x in labels],
        rows=[{k:windows[i][k] for k in ['clip_id','subject','start_time_s','view']} for i in ids],mask=mask.tolist(),targets=targets.tolist(),probabilities=probs)
    (OUT/'points.js').write_text('window.POINTS='+json.dumps(points,separators=(',',':'))+';')
    table=[]
    for name,e in [('Existing demo',baseline)]+[(NAMES[e['condition']],e) for e in ensembles]:
        m=e['all_test'];table.append('<tr'+(' class="selected"' if e is selected else '')+'>'+''.join(f'<td>{v}</td>' for v in [esc(name),fmt(e['validation']['macro_ap']),fmt(m['macro_ap']),fmt(e['quality']['macro_ap']),fmt(m['precision']),fmt(m['recall']),fmt(m['f1']),str(m['fp']),str(m['fn'])])+'</tr>')
    details=[]
    for r in runs:
        details.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['split'],NAMES[r['condition']],r['seed'],r['best_epoch'],fmt(r['best_validation_ap']),fmt(r['all_test']['macro_ap']),f"{r['seconds']:.1f}"])+f'<td><a href="stage{r["stage"]+1}/runs/{r["id"]}/training.json">Record</a> · <a href="stage{r["stage"]+1}/runs/{r["id"]}/best.pt">Weights</a></td></tr>')
    apdelta=selected['all_test']['macro_ap']-baseline['all_test']['macro_ap'];fpdelta=baseline['all_test']['fp']-selected['all_test']['fp']
    note=f"Statistical pooling gives the highest observed test AP: {baseline['all_test']['macro_ap']:.3f} → {observed['all_test']['macro_ap']:.3f}. It also misses more positives. The validation-selected model remains {NAMES[chosen].lower()}; no replacement is promoted from this reused internal benchmark."
    total_time=sum(r['seconds'] for r in runs)
    original_members=[r for r in runs if r['split']=='original' and r['condition']==chosen]
    nparameters=original_members[0]['parameters'];trained_params=nparameters*3
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Theia · Model improvement experiments</title><link rel="stylesheet" href="style.css"></head><body>
<header><b>THEIA / EXPERIMENTS</b><nav><a href="#comparison">Comparison</a><a href="#windows">Decision explorer</a><a href="#runs">Runs</a><a href="/gait_selected/#demo">Video demo ↗</a></nav></header><main>
<div class="eyebrow">15 NOVEMBER 2026 · REUSED INTERNAL BENCHMARK</div><h1>35 new gait-model<br>experiments.</h1><p class="lede">__NOTE__</p>
<div class="stats"><div><b>__RUNS__</b><span>new training runs</span></div><div><b>181 / 28 / 3</b><span>input windows / clips / people</span></div><div><b>__BASEAP__ → __OBSAP__</b><span>highest observed test AP · exploratory</span></div><div><b>__MIN__ min</b><span>recorded training + validation time</span></div></div>
<p class="scope">Targets are automatic pose-derived criteria. They are not clinician-confirmed form labels. Four existing folds test 15 known people in total; the original test has 3. Repeated windows are not independent participants.</p>
<section id="comparison"><h2>Same clips. Every outcome retained.</h2><p>All rows process the same 181 test windows from 28 clips and 3 people. Of these, 124 windows from 16 clips have at least one evaluable target: 607 binary decisions in total. The other 57 windows do not contribute to these supervised metrics. The quality subset retains the existing 91 labeled windows from 14 clips. No outcome-selected showcase is used. Stars in the chart are three-seed ensembles; individual runs remain visible.</p><div class="table"><table><thead><tr><th>Model family</th><th>Val AP</th><th>Test AP</th><th>Quality AP</th><th>Precision</th><th>Recall</th><th>F1</th><th>FP</th><th>FN</th></tr></thead><tbody>__TABLE__</tbody></table></div><p class="scope">The highlighted row was selected by validation before test scoring. Statistical pooling has the highest observed test AP, but was not selected by that rule. Its recall falls from 0.719 to 0.404; missed positives rise from 32 to 68. Its quality-subset AP improves from 0.573 to 0.662. Across four participant folds it improves AP over the raw-pose family on only two. This is a promising direction, not a confirmed general improvement.</p><p class="fine">Threshold 0.5 in this table. AP measures ranking and is threshold-independent. Validation-selected global thresholds and their recall tradeoffs are recorded in each stage’s results.</p><div class="plots"><figure><img src="figures/transfer.png" alt="Individual seeds and ensemble validation versus test AP"><figcaption>Validation selection and test transfer are shown separately.</figcaption></figure><figure><img src="figures/decisions.png" alt="True positives false positives and missed positives for existing and selected models"><figcaption>Fewer alerts can also mean more missed positives.</figcaption></figure></div></section>
<section id="windows"><h2>607 decisions, with their source windows.</h2><p>Each dot represents one available criterion in one video window. Compare any candidate with the existing model, or aggregate the 16 clips with evaluable targets. Hover a point for its clip, person and timestamp.</p><div class="controls"><label>Candidate<select id="candidate" aria-label="Candidate"></select></label><label>Criterion<select id="criterion" aria-label="Criterion"><option value="all">All supported criteria</option></select></label><label>Points<select id="unit" aria-label="Points"><option value="window">Per window + criterion</option><option value="clip">Per clip · error rate</option></select></label><span id="point-count"></span></div><div class="explorer"><svg id="scatter" viewBox="0 0 940 580" role="img" aria-label="Existing versus candidate score scatterplot"></svg><p id="point-info" aria-live="polite">Hover a dot to inspect its source.</p></div><p class="fine">Window view: blue = automatic target 0, orange = automatic target 1; the diagonal is equal score. Clip view: each dot is a video; lower error means closer agreement with the automatic targets.</p></section>
<section><h2>Check the change across people and body regions.</h2><div class="plots"><figure><img src="figures/folds.png" alt="Five model families across four participant-separated folds"><figcaption>Single-seed fold comparisons are descriptive. The same person’s recordings remain in one partition.</figcaption></figure><figure><img src="figures/per_class.png" alt="Per-criterion AP before and after for all supported original test classes"><figcaption>Classes without both positive and negative test examples have no AP estimate.</figcaption></figure></div></section>
<section><h2>What was changed</h2><div class="methods"><article><h3>Geometry</h3><p>Bone directions, relative lengths, joint sine/cosine and frame-to-frame changes are computed from the original 2D pose. No aggregate target ROM, asymmetry value, cohort label or participant ID enters the network.</p></article><article><h3>Training and pooling</h3><p>Three seeds, a reversal ablation, slower training and a smaller frame encoder with mean, standard-deviation, minimum and maximum pooling. The statistical model removes temporal direction differences.</p></article><article><h3>Fixed evaluation</h3><p>Existing participant-separated splits, reference fits, targets and quality screen were reused. Model-family selection was frozen from validation predictions before each stage’s test scoring. The follow-up began after stage one’s internal results were known.</p></article></div><p>The selected ensemble contains three __PARAM__-parameter models (__TOTALPARAM__ total). Measured latency is in the stage reports and excludes pose extraction. Translation/scale invariance checks do not establish camera rotation or 3D projection compensation.</p><p class="fine">Architectural motivation: <a href="https://arxiv.org/abs/1803.01271">Bai et al., temporal convolution (2018)</a>; <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html">Shi et al., joint and bone features (2019)</a>; <a href="https://arxiv.org/abs/1703.06114">Zaheer et al., permutation-invariant set models (2017)</a>. These motivate design choices; the implementations are not reproductions of those papers and inherit no clinical validation.</p></section>
<section id="runs"><h2>All __RUNS__ runs</h2><div class="table"><table><thead><tr><th>Split</th><th>Family</th><th>Seed</th><th>Best epoch</th><th>Val AP</th><th>Test AP</th><th>Seconds</th><th>Artifacts</th></tr></thead><tbody>__DETAILS__</tbody></table></div></section>
<section><h2>Inspect and reproduce</h2><div class="links"><a href="/gait_selected/paper/main.pdf">Revised manuscript PDF ↗</a><a href="combined_selection.json">Frozen family selection ↗</a><a href="stage1/protocol.json">Stage one protocol ↗</a><a href="stage1/results.json">Stage one results ↗</a><a href="stage2/protocol.json">Follow-up protocol ↗</a><a href="stage2/results.json">Follow-up results ↗</a><a href="summary.json">Combined summary ↗</a><a href="README.txt">Commands and limitations ↗</a></div><p class="scope">No new clinical reference was collected. This does not validate injury prediction, treatment or prosthetic adjustments, astronaut running classification, or the safety of a person’s form. The revised manuscript includes these experiments; the demo’s default model is unchanged.</p></section><footer>THEIA · MODEL IMPROVEMENT AUDIT · LOCAL RESEARCH REPORT</footer></main><script src="points.js"></script><script src="app.js"></script></body></html>'''
    for key,value in {'NOTE':esc(note),'RUNS':len(runs),'VAL':fmt(selected['validation']['macro_ap']),'TEST':fmt(selected['all_test']['macro_ap']),'BASEAP':fmt(baseline['all_test']['macro_ap']),'OBSAP':fmt(observed['all_test']['macro_ap']),'MIN':f'{total_time/60:.1f}','TABLE':''.join(table),'DETAILS':''.join(details),'PARAM':f'{nparameters:,}','TOTALPARAM':f'{trained_params:,}'}.items():page=page.replace('__'+key+'__',str(value))
    (OUT/'index.html').write_text(page)
    for source,target in [('gait_improvement_report.css','style.css'),('gait_improvement_report.js','app.js')]:
        shutil.copy2(ROOT/'scripts'/source,OUT/target)
    for s,path in enumerate(STAGES):
        link=OUT/f'stage{s+1}'
        if not link.exists():link.symlink_to(path,target_is_directory=True)
    summary=dict(coverage=dict(input_windows=181,input_clips=28,people=3,labeled_windows=124,labeled_clips=16,binary_decisions=607,quality_windows=91,quality_clips=14),selected_condition=chosen,highest_observed_test_ap_condition=observed['condition'],baseline=baseline,ensembles=ensembles,runs=len(runs),training_seconds=total_time,existing_demo_model_changed=False,clinical_validation=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'README.txt').write_text('THEIA MODEL IMPROVEMENT EXPERIMENTS\n\n35 newly trained runs, 5 model families, existing participant-separated splits.\nThe original test set has been used before; this is exploratory internal evidence.\nLabels are automatic pose measurements, not clinical diagnoses.\n\nExisting completed runs are immutable. Commands for the recorded run:\n.venv-pose/bin/python scripts/test_gait_kinematic_model.py\n.venv-pose/bin/python scripts/run_gait_improvement.py train\n.venv-pose/bin/python scripts/evaluate_gait_improvement.py\n.venv-pose/bin/python scripts/gait_statistical_pool_model.py\n.venv-pose/bin/python scripts/run_gait_improvement_refinement.py train\n.venv-pose/bin/python scripts/run_gait_improvement_refinement.py evaluate\n.venv-pose/bin/python scripts/report_gait_improvement.py\n\nTraining/evaluation commands intentionally reject already-completed runs.\nUse a newly versioned output and a frozen protocol for future experiments.\nAll best, initial and last checkpoints plus histories are under stage1/runs\nand stage2/runs. Download links refer to local artifacts, not published models.\nThe original RGB sources, poses, labels, paper numbers and default demo\ncheckpoint were preserved. No astronaut error classifier was trained.\n')
    server=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14/gait_improvement'
    if not server.exists():server.symlink_to(OUT,target_is_directory=True)
    print(json.dumps({'report':str(OUT),'selected':chosen,'runs':len(runs),'training_seconds':total_time}))

if __name__=='__main__':main()
