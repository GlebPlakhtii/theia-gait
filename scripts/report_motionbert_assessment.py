"""Build a local, per-clip-loaded assessment report from real model results."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
from motionbert_gait_common import ROOT, now, sha, write
from report_motionbert_gait import figures
from plot_gait_training_history import plot_history


def report(folder):
    folder=Path(folder);read=lambda name:json.loads((folder/name).read_text())
    for name in ['normalization.json','dataset.npz']:
        path=folder/name
        if not path.exists():path.symlink_to(Path('dataset')/name)
    evaluation=read('evaluation/results.json');history=read('training/training_history.json')
    protocol=read('dataset/label_protocol.json');inventory=read('dataset/clip_inventory.json')
    model=read('training/resolved_training.json');training=read('training/training_result.json')
    manifest=json.loads((ROOT/'data/prepared/gait_curated_20261114/manifest.json').read_text())['clips']
    predictions=read('predictions.json');by_clip=defaultdict(list)
    for m in predictions['windows']:by_clip[m['clip_id']].append(m)
    for cid,windows in by_clip.items():write(folder/'video_predictions'/f'{cid}.json',dict(windows=windows))
    data=dict(created_at=now(),model_name='MotionBERT fine-tuned encoder',training=training,
        coverage=read('dataset/label_coverage.json'),protocol=protocol,inventory=inventory,
        evaluation=evaluation,predictions={'windows':[]},per_clip_predictions=True,
        history=[dict(epoch=h['epoch'],loss=h['training_loss']['total'],validation={}) for h in history],
        videos={c['clip_id']:{k:c[k] for k in ['clip_id','video','width','height','source_first_frame','fps']} for c in manifest},
        latency=read('latency.json'),configuration=model,
        artifacts=['evaluation/results.json','dataset/label_protocol.json','dataset/label_coverage.json',
            'dataset_audit.json','dataset/dataset_manifest.json','training/training_result.json','training/best.pt','latency.json','backbone_update_audit.json','device_pilot.json','architecture_comparison.json'])
    # Reuse the plot definitions; detailed training curves have their own verified renderer.
    plot_data=dict(data,history=[])
    figures(folder,plot_data)
    plot_inputs=folder/'plot_history_inputs';plot_inputs.mkdir(exist_ok=True)
    for name in ['training_history.json','training_result.json']:
        path=plot_inputs/name
        if not path.exists():path.symlink_to(Path('../training')/name)
    plot_config=dict(model,input_provenance=dict(dataset_sha256=model['dataset_sha256'],
        initial_checkpoint_sha256=model['initial_checkpoint_sha256'],resolved_config_sha256=sha(folder/'training/resolved_training.json')))
    write(plot_inputs/'resolved_training.json',plot_config)
    plot_history(plot_inputs,folder/'figures',title='MotionBERT fine-tuned encoder · automatic targets')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(folder/'predictions.npz') as p,np.load(folder/'dataset/dataset.npz') as d:
        keep=(d['split']==2)&(d['method']==0)
        fig,axes=plt.subplots(2,5,figsize=(17,7))
        for j,(ax,o) in enumerate(zip(axes.flat,protocol['outcomes'])):
            ok=keep&d['measurement_mask'][:,j]&np.isfinite(p['values'][:,j])
            x=d['values'][ok,j];y=p['values'][ok,j]
            ax.scatter(x,y,s=12,alpha=.65)
            if len(x):
                lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],ls='--',c='#777',lw=1)
            ax.set(title=o['id'].replace('_',' '),xlabel='Measured ('+o['unit']+')',ylabel='Predicted ('+o['unit']+')')
            ax.grid(alpha=.15)
        fig.suptitle('Held-out Sapiens windows · one-second measurements · repeated windows are correlated')
        fig.tight_layout();fig.savefig(folder/'figures/heldout_scatter.png',dpi=150);plt.close(fig)
    windows=read('dataset/windows.json')
    sources=sorted({m['source'] for m in windows});views=['front_rear','side','oblique','mixed','unknown']
    total=np.zeros((len(sources),len(views)),int);eligible=total.copy()
    for m in windows:
        i=sources.index(m['source']);j=views.index(m['view'] if m['view'] in views else 'unknown');total[i,j]+=1
        eligible[i,j]+=int(any(v is not None for v in m['value']))
    fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(np.divide(eligible,total,out=np.zeros_like(total,float),where=total>0),vmin=0,vmax=1,cmap='Blues')
    for (i,j),count in np.ndenumerate(total):ax.text(j,i,f'{eligible[i,j]}/{count}',ha='center',va='center',color='black')
    ax.set(xticks=range(len(views)),xticklabels=views,yticks=range(len(sources)),yticklabels=sources,
        title='Numeric target coverage by source/view · eligible / all one-second windows')
    fig.colorbar(im,ax=ax,label='Eligible fraction');fig.tight_layout();fig.savefig(folder/'figures/coverage_by_view.png',dpi=150);plt.close(fig)
    test=evaluation['reports']['test_sapiens'];fig,axes=plt.subplots(1,2,figsize=(13,5))
    labels=[l for l in test['labels'] if l['average_precision'] is not None];y=np.arange(len(labels))
    axes[0].barh(y-.18,[l['baseline_prevalence']['average_precision'] for l in labels],height=.35,label='Train-prevalence baseline',color='#bec9d8')
    axes[0].barh(y+.18,[l['average_precision'] for l in labels],height=.35,label='Fine-tuned MotionBERT',color='#4878a8');axes[0].set(yticks=y,yticklabels=[l['id'] for l in labels],xlabel='Average precision',xlim=(0,1))
    rows=[o for o in test['outcomes'] if o['mae'] is not None];y=np.arange(len(rows));scales=read('dataset/normalization.json')['sapiens']['scale'];mapping={o['id']:scales[i] for i,o in enumerate(protocol['outcomes'])}
    axes[1].barh(y-.18,[o['baseline_median']['mae']/mapping[o['id']] for o in rows],height=.35,label='Train-median baseline',color='#bec9d8')
    axes[1].barh(y+.18,[o['normalized_mae'] for o in rows],height=.35,label='Fine-tuned MotionBERT',color='#4878a8');axes[1].set(yticks=y,yticklabels=[o['id'] for o in rows],xlabel='MAE / training standard deviation')
    for ax in axes:ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Held-out participants · baselines fitted only on training participants');fig.tight_layout();fig.savefig(folder/'figures/baseline_comparison.png',dpi=150);plt.close(fig)
    comparison=read('architecture_comparison.json');models=comparison['models']
    names_model=['scratch_tcn','motionbert_warmup','motionbert_finetuned'];colors=['#8a9bac','#d1a269','#4878a8']
    supported=[l['id'] for l in models['motionbert_finetuned']['labels'] if l['average_precision'] is not None]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5));yy=np.arange(len(supported))
    for k,(name,color) in enumerate(zip(names_model,colors)):
        lookup={l['id']:l for l in models[name]['labels']}
        axes[0].barh(yy+(k-1)*.24,[lookup[label]['average_precision'] for label in supported],height=.23,label=name.replace('_',' '),color=color)
        lookup={o['id']:o for o in models[name]['outcomes']}
        axes[1].barh(np.arange(10)+(k-1)*.24,[lookup[o['id']]['normalized_mae'] for o in protocol['outcomes']],height=.23,label=name.replace('_',' '),color=color)
    axes[0].set(yticks=yy,yticklabels=supported,xlabel='Average precision',xlim=(0,1))
    axes[1].set(yticks=np.arange(10),yticklabels=[o['id'] for o in protocol['outcomes']],xlabel='MAE / training standard deviation')
    for ax in axes:ax.legend(fontsize=7);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Same held-out people, windows and targets · warm-up is a one-epoch baseline')
    fig.tight_layout();fig.savefig(folder/'figures/architecture_comparison.png',dpi=150);plt.close(fig)
    data['architecture_comparison']={k:dict(average_precision=v['macro_average_precision'],normalized_mae=v['mean_normalized_mae']) for k,v in models.items()}
    names=['architecture_comparison','class_support','baseline_comparison','training_curves','confusion_matrices','precision_recall','heldout_scatter','coverage_by_source','coverage_by_view']
    data['figures']=[name for name in names if (folder/'figures'/f'{name}.png').exists()]
    write(folder/'report_data.json',data)
    for name in ['report.html','report.js','report.css']:shutil.copyfile(ROOT/'scripts/gait_assessment_report'/name,folder/name)
    html_path=folder/'report.html';html_path.write_text(html_path.read_text().replace('TCN trained from scratch','MotionBERT fine-tuned encoder'))
    (folder/'index.html').write_text('<!doctype html><meta http-equiv="refresh" content="0;url=report.html"><a href="report.html">Open model report</a>')
    write(folder/'report_status.json',dict(status='results_available',created_at=now(),clips=596,model='fine-tuned MotionBERT',
        predictions=len(predictions['windows']),clinical_validation=False,
        files_sha256={name:sha(folder/name) for name in ['report.html','report.js','report.css','report_data.json','evaluation/results.json']}))
    from summarize_gait_review_coverage import summarize
    summarize(folder)
    exposed=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14/motionbert_assessment'
    if not exposed.exists():exposed.symlink_to(folder,target_is_directory=True)
    print('Report ready: http://localhost:8006/motionbert_assessment/report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/motionbert_gait_finetune_v1/completed_run')
    args=p.parse_args();report(args.output)
