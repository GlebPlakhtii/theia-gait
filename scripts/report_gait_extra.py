"""Comprehensive local research report; scientific figures and interactive demos."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from gait_extra_data import OUT,BASE
from motionbert_gait_common import ROOT,write,sha,now

COLORS={'tcn':'#3976cb','motionbert':'#008c80'}
LABELS={'tcn':'Random TCN','motionbert':'MotionBERT fine-tune'}


def build():
    result=json.loads((OUT/'evaluation.json').read_text());runs=[r for r in result['runs'] if 'primary' in r]
    baseline=[r for r in runs if r['split']=='original' and r['condition']=='baseline']
    cv=[r for r in runs if r['split']!='original'];abl=[r for r in runs if r['condition']!='baseline']
    figures=[];(OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafb','axes.facecolor':'#f8fafb','axes.titleweight':'bold','savefig.facecolor':'#f8fafb'})
    def save(fig,name,title,caption):
        fig.savefig(OUT/'figures'/(name+'.png'),dpi=155,bbox_inches='tight');fig.savefig(OUT/'figures'/(name+'.svg'),bbox_inches='tight');plt.close(fig)
        figures.append(dict(id=name,title=title,caption=caption,src='figures/'+name+'.png',svg='figures/'+name+'.svg'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for ax,key,title in zip(axes,['macro_average_precision','mean_normalized_mae'],['Automatic-label average precision ↑','Normalized measurement MAE ↓']):
        for k,arch in enumerate(COLORS):
            subset=[r for r in baseline if r['architecture']==arch];ys=[r['primary'][key] for r in subset]
            ax.scatter(np.arange(len(ys))*.045+k,ys,c=COLORS[arch],s=65,zorder=3)
            ax.plot([k-.12,k+.30],[np.mean(ys)]*2,c=COLORS[arch],lw=3)
            for n,y in enumerate(ys):ax.annotate(str(subset[n]['seed'])[-2:],(k+n*.045,y),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set(xticks=[.09,1.09],xticklabels=list(LABELS.values()),title=title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Five new seeds · original participant split',fontsize=16);fig.tight_layout()
    save(fig,'seed_stability','Does the result survive a new seed?','Each dot is one predeclared seed. Bars are descriptive means; these are the same three test people, not five independent cohorts.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key,title in zip(axes,['macro_average_precision','mean_normalized_mae'],['Average precision ↑','Normalized MAE ↓']):
        for arch in COLORS:
            sub=sorted([r for r in cv if r['architecture']==arch],key=lambda r:r['split']);ax.plot(range(1,len(sub)+1),[r['primary'][key] for r in sub],'o-',label=LABELS[arch],color=COLORS[arch],lw=2)
        ax.set(xticks=range(1,5),xlabel='Participant fold',title=title);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();save(fig,'participant_folds','Held-out participant rotation','Each fold holds out a different ProGait person and 2–3 LJMU people. Each fold refits references from training people, so the label definitions and supported classes differ.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    groups=[(arch,c) for c in ['primary_only','root_centered'] for arch in COLORS]
    for ax,key,title in zip(axes,['macro_average_precision','mean_normalized_mae'],['Change in AP (positive favors ablation)','Change in normalized MAE (negative favors ablation)']):
        for k,(arch,condition) in enumerate(groups):
            for r in [r for r in abl if r['architecture']==arch and r['condition']==condition]:
                b=next(b for b in baseline if b['architecture']==arch and b['seed']==r['seed']);delta=r['primary'][key]-b['primary'][key]
                ax.scatter(k+(r['seed']-20261116)*.12,delta,color=COLORS[arch],s=65)
        ax.axhline(0,c='#444',lw=1);ax.set(xticks=range(4),xticklabels=['TCN\nPrimary only','MB\nPrimary only','TCN\nCentered','MB\nCentered'],title=title);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'paired_ablations','What changes when we remove publisher poses or center the input?','Three paired seeds per condition. Epoch draw counts match the baseline. Stopping epochs can differ because selection uses validation only.')
    ordered=baseline+cv+abl;names=[r['id'].replace('original_','').replace('motionbert','MB').replace('202611','s') for r in ordered]
    for kind,key,metric,title in [('class','labels','average_precision','Classification support and AP'),('numeric','outcomes','normalized_mae','Normalized measurement errors')]:
        matrix=np.array([[np.nan if x[metric] is None else x[metric] for x in r['primary'][key]] for r in ordered])
        fig,ax=plt.subplots(figsize=(13,max(7,len(ordered)*.26)))
        cmap=plt.get_cmap('viridis' if kind=='class' else 'magma_r').copy();cmap.set_bad('#dfe4e7')
        im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=1 if kind=='class' else max(1,float(np.nanpercentile(matrix,95))))
        ax.set(yticks=range(len(names)),yticklabels=names,xticks=range(matrix.shape[1]),xticklabels=[x['id'].replace('_',' ') for x in ordered[0]['primary'][key]])
        ax.tick_params(axis='x',labelrotation=65,labelsize=8);ax.tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=ax,shrink=.65);fig.tight_layout()
        save(fig,kind+'_heatmap',title,'Every completed run is shown. Gray cells are unavailable, disabled, or lack both signs for classification; they are not zero performance. Numeric color saturates at the 95th percentile.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,arch in zip(axes,COLORS):
        for r in [r for r in baseline if r['architecture']==arch]:
            history=json.loads((OUT/'runs'/r['id']/'training/training_history.json').read_text())
            xs=[];ys=[]
            for h in history:
                if h.get('validation'):xs.append(h['epoch']);ys.append(h['validation']['selection_score'])
            ax.plot(xs,ys,label=str(r['seed'])[-2:],alpha=.8)
        ax.set(title=LABELS[arch],xlabel='Epoch',ylabel='Validation macro AP');ax.legend(title='Seed suffix',ncol=3,fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'training_curves','Validation traces for all five seeds','Only validation scores select checkpoints. MotionBERT epoch 1 updates the assessment heads; subsequent epochs update the encoder too.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arch in COLORS:
        rs=[r for r in baseline if r['architecture']==arch]
        axes[0].scatter([r['elapsed_s'] for r in rs],[r['primary']['macro_average_precision'] for r in rs],c=COLORS[arch],s=65,label=LABELS[arch])
        axes[1].scatter([r['elapsed_s'] for r in rs],[r['primary']['mean_normalized_mae'] for r in rs],c=COLORS[arch],s=65,label=LABELS[arch])
    axes[0].set(xlabel='Measured training seconds',ylabel='Held-out AP',title='Training cost versus teacher agreement');axes[1].set(xlabel='Measured training seconds',ylabel='Normalized MAE',title='Training cost versus measurement error')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'training_cost','Compute cost and predictive agreement','Original-split seeds only. Durations include each training loop and checkpoint work, but exclude pose extraction and data preparation. Epoch counts differ under validation stopping; this is not an equal-compute benchmark.')
    for arch in COLORS:
        r=next(r for r in baseline if r['architecture']==arch and r['seed']==20261115)
        labels=[l for l in r['primary']['labels'] if l['average_precision'] is not None]
        fig,axes=plt.subplots(3,3,figsize=(11,10))
        for ax,label in zip(axes.flat,labels):
            c=label['confusion'];a=np.array([[c['tn'],c['fp']],[c['fn'],c['tp']]])
            ax.imshow(a,cmap='Blues',vmin=0)
            for i in range(2):
                for j in range(2):ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=16,color='white' if a[i,j]>a.max()*.55 else '#172332')
            ax.set(title=label['id'].replace('_',' '),xticks=[0,1],xticklabels=['Negative','Review'],yticks=[0,1],yticklabels=['Negative','Review'],xlabel='Model',ylabel='Automatic teacher');ax.title.set_fontsize(9)
        for ax in list(axes.flat)[len(labels):]:ax.set_axis_off()
        fig.suptitle(LABELS[arch]+' · fixed seed 20261115 · original test',fontsize=15);fig.tight_layout()
        save(fig,'confusion_'+arch,LABELS[arch]+' confusion counts','Fixed threshold 0.5 and the first predeclared seed. Only classes with both test signs are shown. Counts are correlated windows from three held-out people; labels are automatic review targets, not clinician judgments.')
    maps=json.loads((OUT/'protocol.json').read_text())['participant_maps']
    cvpeople=sorted({p for name,mapping in maps.items() if name!='original' for p,split in mapping.items() if split=='test'})
    outcome_ids=[o['id'] for o in runs[0]['primary']['outcomes']]
    base_np=np.load(BASE/'dataset.npz');base_values=base_np['values'];base_method=base_np['method'];base_valid=base_np['input_any_valid']
    windows=json.loads((BASE/'windows.json').read_text());clip_prediction_points=[]
    for run in cv:
        prediction=np.load(OUT/'runs'/run['id']/'test_predictions.npz');mask=np.load(OUT/'splits'/run['split']/'targets.npz')['regression_mask'];by_clip=defaultdict(list)
        for k,idx in enumerate(prediction['indices']):
            if base_method[idx]==0 and base_valid[idx]:by_clip[windows[idx]['clip_id']].append((k,idx))
        for cid,rows in by_clip.items():
            for j,oid in enumerate(outcome_ids):
                valid=[(k,i) for k,i in rows if mask[i,j]]
                if not valid:continue
                ii=[i for k,i in valid];kk=[k for k,i in valid]
                clip_prediction_points.append(dict(clip_id=cid,subject=windows[ii[0]]['subject'],architecture=run['architecture'],fold=run['split'],outcome=oid,
                    observed=float(np.mean(base_values[ii,j])),predicted=float(np.mean(prediction['values'][kk,j])),windows=len(ii)))
    fig,axes=plt.subplots(5,2,figsize=(14,21));points=[]
    for j,(ax,oid) in enumerate(zip(axes.flat,outcome_ids)):
        allvals=[]
        for arch in COLORS:
            clip_rows=[p for p in clip_prediction_points if p['outcome']==oid and p['architecture']==arch]
            ax.scatter([p['observed'] for p in clip_rows],[p['predicted'] for p in clip_rows],c=COLORS[arch],s=19,alpha=.45,label=LABELS[arch]+' clips')
            allvals.extend(v for p in clip_rows for v in [p['observed'],p['predicted']])
            for r in [r for r in cv if r['architecture']==arch]:
                for p in r['balanced']['outcomes'][j]['people']:
                    ax.scatter(p['observed'],p['predicted'],facecolors='none',edgecolors=COLORS[arch],marker='s' if p['subject'].startswith('progait') else 'o',s=100,linewidths=1.8,alpha=1)
                    points.append(dict(**p,outcome=oid,architecture=arch,fold=r['split']));allvals.extend([p['observed'],p['predicted']])
        if allvals:lo=min(allvals);hi=max(allvals);ax.plot([lo,hi],[lo,hi],'--',c='#888',lw=1)
        ax.set(title=oid.replace('_',' '),xlabel='Observed ('+runs[0]['balanced']['outcomes'][j]['unit']+')',ylabel='Predicted ('+runs[0]['balanced']['outcomes'][j]['unit']+')');ax.tick_params(labelsize=10);ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Held-out clips and participant summaries\nSmall filled = clip · large outline = person (square: ProGait; circle: LJMU)',fontsize=15);fig.tight_layout();save(fig,'participant_predictions','Out-of-fold video clips + people, on the same axes','Small filled dots = clip averages over supported windows; large outlines = the existing participant means balanced across trials. Green = MotionBERT, blue = TCN; outlined squares = ProGait, outlined circles = LJMU. Each participant is held out in one fold. Clip dots are correlated within a person and are not independent sample counts. Diagonals indicate agreement with pose-derived measurements, not clinical truth.')
    mat=np.zeros((len(cvpeople),10),int)
    for r in [r for r in cv if r['architecture']=='motionbert']:
        for j,o in enumerate(r['balanced']['outcomes']):
            for p in o['people']:mat[cvpeople.index(p['subject']),j]=p['windows']
    fig,ax=plt.subplots(figsize=(12,6));im=ax.imshow(mat,aspect='auto',cmap='YlGnBu')
    for i in range(len(cvpeople)):
        for j in range(10):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=8,color='white' if mat[i,j]>mat.max()*.55 else '#172332')
    ax.set(yticks=range(len(cvpeople)),yticklabels=cvpeople,xticks=range(10),xticklabels=[x.replace('_',' ') for x in outcome_ids]);ax.tick_params(axis='x',labelrotation=60,labelsize=8);fig.colorbar(im,ax=ax,label='Available test windows');fig.tight_layout()
    save(fig,'participant_coverage','Which people support each measurement?','Each person is held out once across the four folds. Counts are correlated windows, not independent observations. Zero means the measurement was unavailable under the frozen gates.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    stress=[r for r in runs if r['translation_stress']]
    for ax,key,title in zip(axes,['class_absolute_score_change','normalized_prediction_change'],['Mean absolute classifier score change','Normalized numeric prediction change']):
        for k,(arch,cond) in enumerate([(a,c) for c in ['baseline','root_centered'] for a in COLORS]):
            ys=[r['translation_stress'][key] for r in stress if r['architecture']==arch and r['condition']==cond]
            ax.scatter(np.arange(len(ys))*.07+k,ys,c=COLORS[arch],s=60)
        ax.set(xticks=[.07,1.07,2.07,3.07],xticklabels=['TCN raw','MB raw','TCN centered','MB centered'],title=title);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'translation_stress','Sensitivity to synthetic image translation','Fixed +0.10 x, −0.10 y normalized-coordinate shift, with unchanged masks and targets. This tests framing translation only. Root centering does not correct perspective, yaw or scale.')
    # Reuse already audited full-clip cohort statistics, without treating clips as people.
    comparison=json.loads((ROOT/'results/gait_metrics_20261114/comparison_results.json').read_text())
    cohort=[r for r in comparison['results'] if r['sensitivity']=='base' and r['status']=='descriptive_comparison_available']
    clip_values=json.loads((ROOT/'results/gait_metrics_20261114/comparison_clip_values.json').read_text())['rows']
    clip_values=[{k:r[k] for k in ['clip_id','subject','source','cohort','trial_block','value','retained_duration_s','mode','outcome','activity_group','environment','view','pose_method']} for r in clip_values if r['sensitivity']=='base']
    # Retain audited aggregates, but show their contributing clips alongside them.
    def cohort_rows(comp,group):
        allowed={cid for u in comp['groups'][group]['units'] for cid in u['clips']}
        return [r for r in clip_values if r['mode']==comp['mode'] and r['outcome']==comp['outcome']['id'] and r['cohort']==group and r['clip_id'] in allowed and all(r[k]==v for k,v in comp['stratum'].items())]
    group_colors={'with_amputation':'#bd7428','without_amputation':'#008c80'}
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,oid in zip(axes.flat,['knee_rom95_mean','trunk_side_rom95','hip_line_rom95_proxy','shoulder_line_rom95']):
        comp=next(c for c in cohort if c['mode']=='guarded_known_subject_sensitivity' and c['outcome']['id']==oid)
        labels=[]
        for k,(group,color) in enumerate(group_colors.items()):
            rows=cohort_rows(comp,group);units=comp['groups'][group]['units']
            jitter=np.array([((i*17)%23-11)/65 for i in range(len(rows))]);ax.scatter(k+jitter,[r['value'] for r in rows],s=22,c=color,alpha=.45)
            jitter=np.array([((i*7)%13-6)/50 for i in range(len(units))]);ax.scatter(k+jitter,[u['value'] for u in units],s=95,facecolors='none',edgecolors=color,linewidths=1.8)
            labels.append(('ProGait' if k==0 else 'LJMU')+f"\n{len(rows)} clips / {len(units)} people")
        ax.set(xticks=[0,1],xticklabels=labels,ylabel=comp['outcome']['unit'],title=comp['outcome']['label']);ax.set_xlim(-.45,1.45);ax.title.set_fontsize(11);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Whole-clip observations with participant summaries\nSmall filled dots = clips · large outlined dots = people',fontsize=16);fig.tight_layout()
    save(fig,'participant_distributions','Whole-clip distributions + people','Small filled dots are individual clip values; large outlines are the existing trial-balanced person summaries. Horizontal jitter is display-only. Each panel keeps the same camera-view, environment and pose-method stratum. ProGait guarding differs from reference walking; these are descriptive measurements, not clinical or causal effects.')
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,mode in zip(axes,['guarded_known_subject_sensitivity','unsupported_context_descriptive']):
        cx=next(c for c in cohort if c['mode']==mode and c['outcome']['id']=='trunk_frontal_rom95');cy=next(c for c in cohort if c['mode']==mode and c['outcome']['id']=='shoulder_line_rom95')
        for group,color in group_colors.items():
            yrows={r['clip_id']:r for r in cohort_rows(cy,group)};rows=[r for r in cohort_rows(cx,group) if r['clip_id'] in yrows]
            yu={u['unit_id']:u for u in cy['groups'][group]['units']};units=[u for u in cx['groups'][group]['units'] if u['unit_id'] in yu]
            ax.scatter([r['value'] for r in rows],[yrows[r['clip_id']]['value'] for r in rows],s=25,c=color,alpha=.5,label=group.replace('_',' ')+f' · {len(rows)} clips')
            ax.scatter([u['value'] for u in units],[yu[u['unit_id']]['value'] for u in units],s=110,facecolors='none',edgecolors=color,linewidths=2)
        ax.scatter([],[],s=110,facecolors='none',edgecolors='#555',linewidths=2,label='Outline = person / source summary')
        ax.set(xlabel='Frontal trunk ROM (°)',ylabel='Shoulder-line ROM (°)',title='Known participants · guarding differs' if mode.startswith('guarded') else 'Unresolved source identities');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'trunk_shoulder_scatter','Trunk and shoulder: clips + summaries','Filled dots show clips with both measurements; outlined dots show existing person summaries, or explicitly unresolved source-block summaries in the right panel. Multiple clips from one person are correlated. Source blocks are not independent people. View and pose method remain matched within each panel.')
    oldfigs=[('source_coverage','Full corpus source coverage'),('method_sensitivity','Pose-method sensitivity')]
    for name,title in oldfigs:
        figures.append(dict(id=name,title=title,src='/gait_metrics/figures/'+name+'.png',svg='/gait_metrics/figures/'+name+'.svg',caption='Reused completed full-clip comparison. Additional-input identities remain unresolved; ProGait guarding differs from reference walking. These are descriptive source comparisons, not causal effects of amputation.'))
    squat=json.loads((ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14/data.json').read_text())
    astronaut=[c for c in squat['clips'] if c['source_id'] not in ['GYM_01','GYM_02','GYM_03','GYM_04','GYM_05']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));sids=sorted({c['source_id'] for c in astronaut});palette=plt.get_cmap('tab10')
    for k,sid in enumerate(sids):
        cs=[c for c in astronaut if c['source_id']==sid and c['features']['completeRep']]
        if not cs:continue
        axes[0].scatter([c['features']['rawDepth'] for c in cs],[c['features']['depth'] for c in cs],label=sid,color=palette(k),alpha=.75)
        axes[1].scatter([c['features']['asymmetry'] for c in cs],[c['features']['ikResidual']*100 for c in cs],color=palette(k),alpha=.75)
    axes[0].plot([0,150],[0,150],'--',c='#777');axes[0].set(xlabel='Projected 2D peak knee flexion (°)',ylabel='OpenSim-fit peak flexion estimate (°)',title='Projection / model sensitivity');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Estimated bilateral angle difference (°)',ylabel='IK residual (% nominal leg)',title='Asymmetry versus fitting residual');fig.tight_layout()
    save(fig,'astronaut_projection','Astronaut and ARED ground-training estimates','Previously computed candidate squat repetitions, including withheld ones. OpenSim fit consistency does not establish anatomical accuracy. Ground and in-mission sources remain distinct; no gait classifier is applied to these squats.')
    old=json.loads((ROOT/'results/motionbert_gait_finetune_v1/completed_run/report_data.json').read_text())
    intake=json.loads((ROOT/'data/manifests/exercise_intake_20261114.json').read_text())
    selections=json.loads((ROOT/'data/prepared/exercise_intake_20261114/selections.json').read_text())
    # Intake clip indices refer to the prepared segment, not the full recording.
    # Follow its existing frame lineage instead of relabeling prepared frames as raw.
    lineages={}
    for sel in selections:
        if sel['id'] not in sids:continue
        path=Path(sel['prepared_path']).with_suffix('')/'original_frame_lineage.csv'
        with path.open() as f:lineages[sel['id']]=list(csv.DictReader(f))
    for clip in astronaut:
        first=clip['source_first_frame'];count=clip['frames']
        if clip['source_id'] in lineages:
            rows=lineages[clip['source_id']][first:first+count];assert len(rows)==count
            clip['raw_source_frames']=[int(r['original_frame_index']) for r in rows]
            clip['raw_source_times_s']=[float(r['original_timestamp_s']) for r in rows]
            clip['raw_clock_provenance']='Existing original_frame_lineage.csv, indexed by prepared native frame'
        else:
            clip['raw_source_frames']=list(range(first,first+count))
            clip['raw_source_times_s']=[clip['source_start_s']+i/clip['fps'] for i in range(count)]
            clip['raw_clock_provenance']='Original native recording indices retained by the existing doctor clip manifest'
        assert len(clip['raw_source_frames'])==clip['frames']
    prior_sources=json.loads((ROOT/'data/manifests/sources.json').read_text())['sources']
    old_source_map={s['dataset_id']:s for s in prior_sources}
    source_context={
        'ARED':dict(title='Adenot ARED / EasyMotion-2 development footage',url=old_source_map['T1']['source_page'],note='Prior local development footage. Catalog describes EasyMotion-2; EMS activation, exercise load and prescription in these intervals are unknown. Catalog URL was previously unavailable; source media was verified separately.'),
        'GERST_01':dict(title='Gerst archival ARED · first interval',url=old_source_map['T2']['source_page'],note='2014 mission footage in a later archive path. Same crew and recording as GERST_02; not independent people.'),
        'GERST_02':dict(title='Gerst archival ARED · second interval',url=old_source_map['T2']['source_page'],note='Same crew and recording as GERST_01; development-exposed, no independent clinician labels.')}
    for sid in ['P21_ARED_01','CSA_ARED_01','HOP_GROUND_01','WIL_GROUND_01']:
        sel=next(s for s in selections if s['id']==sid)
        source_context[sid]=dict(title=sid.replace('_',' '),url=sel['source_page'],note=sel['notes'])
    discovery=[dict(title='Exercise in space with Samantha Cristoforetti',url='https://www.esa.int/media/exercise-in-space-with-samantha-cristoforetti',
        video='https://media.esa.int/legacy/delivery/public/videos/2024/01/024/orig-2401_024_AR_EN.mp4?dl=',status='Previously catalogued source, now reverified for this report; full unchanged remote video. No new pose extraction or training use.',
        note='ESA/NASA; page publication date 20 February 2015. Previously development-exposed, not a fresh holdout. ESA notes third-party content and restrictions on modification and reuse.'),
        dict(title='Matthias Maurer — Workout in 360°',url='https://www.esa.int/ESA_Multimedia/Videos/2021/12/Workout_in_3601_Cosmic_Kiss/(lang)/en',status='Context only; excluded from timing metrics.',note='Official ESA/NASA timelapse: 53 minutes captured at one frame every two seconds. Not continuous native-rate motion.')]
    groups=[]
    for arch in COLORS:
        rs=[r for r in baseline if r['architecture']==arch]
        groups.append(dict(architecture=arch,count=len(rs),AP=dict(mean=np.mean([r['primary']['macro_average_precision'] for r in rs]),min=min(r['primary']['macro_average_precision'] for r in rs),max=max(r['primary']['macro_average_precision'] for r in rs)),
            normalized_MAE=dict(mean=np.mean([r['primary']['mean_normalized_mae'] for r in rs]),min=min(r['primary']['mean_normalized_mae'] for r in rs),max=max(r['primary']['mean_normalized_mae'] for r in rs))))
    paired=[]
    for condition in ['primary_only','root_centered']:
        for arch in COLORS:
            rows=[]
            for r in [r for r in abl if r['condition']==condition and r['architecture']==arch]:
                b=next(b for b in baseline if b['architecture']==arch and b['seed']==r['seed'])
                rows.append(dict(seed=r['seed'],delta_AP=r['primary']['macro_average_precision']-b['primary']['macro_average_precision'],delta_normalized_MAE=r['primary']['mean_normalized_mae']-b['primary']['mean_normalized_mae']))
            paired.append(dict(condition=condition,architecture=arch,pairs=rows,mean_delta_AP=np.mean([r['delta_AP'] for r in rows]),mean_delta_normalized_MAE=np.mean([r['delta_normalized_MAE'] for r in rows])))
    write(OUT/'experiment_summary.json',dict(created_at=now(),seed_summary=groups,paired_ablations=paired,
        interpretation='Descriptive paired seeds on reused internal test people. No confidence interval or clinical validation claim.'))
    coverage=dict(clips=len(old['inventory']),windows=old['coverage']['windows'],cv_people=len(cvpeople),
        cv_people_with_any_primary_measurement=len({p['subject'] for r in cv for o in r['balanced']['outcomes'] for p in o['people']}),source_clips=dict(Counter(c['source'] for c in old['inventory'])),
        cv_people_without_primary_measurements=sorted(set(cvpeople)-{p['subject'] for r in cv for o in r['balanced']['outcomes'] for p in o['people']}),
        astronaut_clips=len(astronaut),astronaut_sources=len(sids),astronaut_reps=sum(c['features']['completeRep'] for c in astronaut),
        catalogued_intake_sources=10,known_people=16,inventory=old['inventory'])
    coverage['progait_knee_people']=sorted({p['subject'] for r in cv for p in r['balanced']['outcomes'][0]['people'] if p['subject'].startswith('progait_')})
    for f in figures:
        path=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14'/f['src'].lstrip('/') if f['src'].startswith('/') else OUT/f['src']
        with Image.open(path) as im:f['width'],f['height']=im.size
    data=dict(created_at=now(),runs=runs,all_run_statuses=[{k:r[k] for k in ['id','status']} for r in result['runs']],seed_summary=groups,paired_summary=paired,
        figures=figures,examples=result['examples'],videos=old['videos'],coverage=coverage,participant_points=points,clip_prediction_points=clip_prediction_points,cohort_comparisons=cohort,cohort_clip_values=clip_values,
        astronaut_clips=astronaut,astronaut_sources=[s for s in squat['sources'] if s['id'] in sids],intake_sources=intake['sources'],selections=selections,
        astronaut_discovery=discovery,astronaut_source_context=source_context,protocol=json.loads((OUT/'protocol.json').read_text()),preflight=json.loads((OUT/'preflight.json').read_text()),
        squat_defaults=squat['defaults'],edges3d=squat['edges3d'],source_issues=comparison['limitations'],clinical_validation=False)
    write(OUT/'report_data.json',data)
    for source,target in [('report.html','index.html'),('report.css','report.css'),('report.js','report.js'),('anatomy.js','anatomy.js')]:shutil.copyfile(ROOT/'scripts/gait_extra_report'/source,OUT/target)
    route=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14/gait_experiments'
    if not route.exists():route.symlink_to(OUT,target_is_directory=True)
    intake_route=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14/astronaut_intake'
    if not intake_route.exists():intake_route.symlink_to(ROOT/'data/prepared/exercise_intake_20261114',target_is_directory=True)
    write(OUT/'report_manifest.json',dict(created_at=now(),url='http://localhost:8006/gait_experiments/',figures=len(figures),runs=len(runs),
        sources={'evaluation':sha(OUT/'evaluation.json'),'report_data':sha(OUT/'report_data.json'),'builder':sha(__file__)},
        browser_verification='pending',astronaut_new_inference=False,prior_astronaut_results_reused=True))
    print(json.dumps(dict(figures=len(figures),runs=len(runs),video_examples=len(result['examples']),astronaut_clips=len(astronaut))))


if __name__=='__main__':build()
