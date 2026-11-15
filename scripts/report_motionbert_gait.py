"""Static local report; real output only, including explicit pending status."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from motionbert_gait_common import ROOT,OUT,now,write


def figures(folder,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=folder/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.facecolor':'#f7f9fc','axes.facecolor':'#ffffff'})
    labels=data.get('protocol',{}).get('labels',[])
    if labels:
        fig,ax=plt.subplots(figsize=(10,5))
        y=np.arange(len(labels));ax.barh(y-.18,[len(l['negative_training_subjects'])for l in labels],height=.35,label='Negative')
        ax.barh(y+.18,[len(l['positive_training_subjects'])for l in labels],height=.35,label='Positive',color='#d28b38')
        ax.set(yticks=y,yticklabels=[l['id']for l in labels],xlabel='Independent training participants',title='Automatic class support; people can have both types of windows')
        ax.axvline(2,color='#888',ls='--',lw=1);ax.legend();fig.tight_layout();fig.savefig(dest/'class_support.png',dpi=150);plt.close(fig)
    inventory=data.get('inventory',[])
    if inventory:
        sources=sorted({c['source']for c in inventory});total=Counter(c['source']for c in inventory);available=Counter(c['source']for c in inventory if c['windows']>0)
        fig,ax=plt.subplots(figsize=(9,3.5));x=np.arange(len(sources))
        ax.bar(x,[total[s]for s in sources],label='Accounted clips',color='#bec9d8');ax.bar(x,[available[s]for s in sources],label='Clips with pose windows',color='#4878a8')
        ax.set(xticks=x,xticklabels=sources,ylabel='Clips',title='Corpus coverage; a pose window does not imply an available anomaly label')
        ax.legend();fig.tight_layout();fig.savefig(dest/'coverage_by_source.png',dpi=150);plt.close(fig)
    history=data.get('history',[])
    if history:
        fig,(ax,other)=plt.subplots(1,2,figsize=(11,3.5))
        ax.plot([h['epoch']for h in history],[h['loss']for h in history],marker='o')
        ax.set(xlabel='Epoch',ylabel='Training loss',title='Fine-tuning loss; teacher supervision')
        other.plot([h['epoch']for h in history],[h.get('validation',{}).get('mean_normalized_mae') for h in history],marker='o',color='#4878a8')
        other.set(xlabel='Epoch',ylabel='Normalized measurement MAE',title='Validation; participant split')
        ax.grid(alpha=.2);other.grid(alpha=.2);fig.tight_layout();fig.savefig(dest/'training_loss.png',dpi=150);plt.close(fig)
    evaluation=data.get('evaluation')
    if evaluation and evaluation.get('status')=='complete':
        report=evaluation['reports']['test_sapiens']
        supported=[label for label in report['labels'] if label['enabled'] and label['n']]
        if supported:
            columns=min(3,len(supported));rows=(len(supported)+columns-1)//columns
            fig,axes=plt.subplots(rows,columns,figsize=(4*columns,3.5*rows),squeeze=False)
            for ax,label in zip(axes.flat,supported):
                counts=np.array([[label['tn'],label['fp']],[label['fn'],label['tp']]])
                ax.imshow(counts,cmap='Blues')
                for (i,j),value in np.ndenumerate(counts):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>counts.max()/2 else 'black')
                ax.set(title=label['id'],xticks=[0,1],xticklabels=['Negative','Positive'],yticks=[0,1],yticklabels=['Negative','Positive'],xlabel='Model (score ≥ 0.5)',ylabel='Automatic teacher')
            for ax in list(axes.flat)[len(supported):]:ax.axis('off')
            fig.suptitle('Held-out window counts; overlapping windows are correlated',fontsize=10)
            fig.tight_layout();fig.savefig(dest/'confusion_matrices.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4))
        rows=[o for o in report['outcomes'] if o['mae'] is not None]
        if rows:
            # Standardized error avoids mixing degrees and L0 on one value axis.
            scales=json.loads((folder/'normalization.json').read_text())['sapiens']['scale']
            outcomes=data['protocol']['outcomes'];mapping={o['id']:scales[i]for i,o in enumerate(outcomes)}
            ax.barh([r['id']for r in rows],[r['mae']/mapping[r['id']]for r in rows],color='#4878a8')
            ax.set_xlabel('MAE / training target standard deviation')
        else:ax.text(.5,.5,'No eligible held-out measurements',ha='center')
        ax.set_title('Held-out Sapiens measurement agreement');fig.tight_layout();fig.savefig(dest/'measurement_error.png',dpi=150);plt.close(fig)
        prediction_file=folder/'predictions.npz';dataset=folder/'dataset.npz'
        if prediction_file.exists() and dataset.exists():
            with np.load(prediction_file)as p,np.load(dataset)as a:
                ids=p['indices'];valid=(a['split'][ids]==2)&(a['method'][ids]==0)
                fig,axes=plt.subplots(1,3,figsize=(12,3.5))
                for ax,j in zip(axes,[4,6,7]):
                    keep=valid&a['measurement_mask'][ids,j]
                    xx=a['values'][ids,j][keep];yy=p['values'][:,j][keep]
                    ax.scatter(xx,yy,s=14,alpha=.65)
                    if len(xx):
                        lo=min(xx.min(),yy.min());hi=max(xx.max(),yy.max());ax.plot([lo,hi],[lo,hi],color='#777',ls='--')
                    ax.set(title=data['protocol']['outcomes'][j]['id'],xlabel='Measured (degrees)',ylabel='Predicted (degrees)')
                fig.suptitle('Held-out windows; repeated windows are not independent people',fontsize=10)
                fig.tight_layout();fig.savefig(dest/'heldout_scatter.png',dpi=150);plt.close(fig)
                fig,ax=plt.subplots(figsize=(8,4))
                lines=0
                for j,label in enumerate(data['protocol']['labels']):
                    keep=valid&(a['labels'][ids,j]>=0)
                    if not label['classification_enabled'] or len(np.unique(a['labels'][ids,j][keep]))<2:continue
                    target=a['labels'][ids,j][keep];score=p['probabilities'][:,j][keep]
                    order=np.argsort(-score,kind='stable');target=target[order];score=score[order]
                    ends=np.r_[np.flatnonzero(np.diff(score)),len(score)-1];tp=np.cumsum(target)[ends]
                    ax.step(tp/target.sum(),tp/(ends+1),where='post',label=label['id']);lines+=1
                if lines:ax.legend(fontsize=7,loc='best')
                else:ax.text(.5,.5,'No supported held-out class has both labels',ha='center')
                ax.set(xlabel='Recall against automatic teacher',ylabel='Precision',xlim=(0,1),ylim=(0,1.02),title='Held-out precision–recall; not clinical validation')
                fig.tight_layout();fig.savefig(dest/'precision_recall.png',dpi=150);plt.close(fig)


def report(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def read(name,default=None):
        path=folder/name
        return json.loads(path.read_text()) if path.exists() else default
    manifest=json.loads((ROOT/'data/prepared/gait_curated_20261114/manifest.json').read_text())['clips']
    data=dict(created_at=now(),coverage=read('label_coverage.json',{}),protocol=read('label_protocol.json',{}),
        training=read('training_status.json',{}),history=read('training_history.json',[]),
        evaluation=read('evaluation/results.json'),predictions=read('predictions.json',{'windows':[]}),
        inventory=read('clip_inventory.json',[]),
        videos={c['clip_id']:{k:c[k]for k in ['clip_id','video','width','height','source_first_frame']}for c in manifest})
    data['artifacts']=[name for name in ['evaluation/results.json','label_protocol.json','label_coverage.json',
        'backbone_update_audit.json','checkpoints/best.pt'] if (folder/name).exists()]
    figures(folder,data)
    names=(['class_support'] if data['protocol'].get('labels') else [])+(['coverage_by_source'] if data['inventory'] else [])
    if data['history']:names+=['training_loss']
    if data['evaluation'] and data['evaluation'].get('status')=='complete':
        names+=['measurement_error','heldout_scatter','precision_recall','confusion_matrices']
    data['figures']=[name for name in names if (folder/'figures'/f'{name}.png').exists()]
    write(folder/'report_data.json',data)
    assets=ROOT/'scripts/motionbert_report'
    import shutil
    for name in ['report.html','report.js','report.css','index.html']:
        shutil.copyfile(assets/name,folder/name)
    complete=data['evaluation'] and data['evaluation'].get('status')=='complete'
    write(folder/'report_status.json',dict(updated_at=now(),status='results_available' if complete else 'training_pending',
        clinical_validation=False,clips_accounted_for=len(data['inventory']),prediction_windows=len(data['predictions']['windows'])))
    print('Report written: '+str(folder/'report.html'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=OUT);args=p.parse_args();report(args.data)
