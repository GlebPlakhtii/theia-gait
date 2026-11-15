"""Export publication figures, provenance, and curated video-page data."""
import json,shutil,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_gait_selected import ROOT,OUT,EX,BASE,write,evaluate
from gait_from_scratch_evaluation import binary_metrics

PAPER=ROOT/'deliverables/Theia_Selected_Gait_Paper_2026-11-15'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True);(PAPER/'images').mkdir(exist_ok=True)
r=json.loads((OUT/'results.json').read_text());rows=json.loads((OUT/'window_results.json').read_text())
a=np.load(OUT/'evaluation_arrays.npz');y=a['y'];p=a['probabilities'];mask=a['mask'];ii=a['indices'];lookup={int(i):j for j,i in enumerate(ii)}
groups={'all_test':np.arange(len(ii)),'quality':np.array([lookup[int(i)] for i in a['quality_indices']]),'showcase':np.array([lookup[int(i)] for i in a['showcase_indices']])}
names=[d['label'] for d in r['all_test']['labels']];common=[j for j,n in enumerate(names) if r['showcase']['labels'][j]['ap'] is not None]
for k,ix in groups.items():
 s=r[k];s['common_class_macro_ap']=float(np.mean([binary_metrics(y[ix,j],p[ix,j],mask[ix,j])['average_precision'] for j in common]))
 s['common_classes']=[names[j] for j in common];s['always_negative_accuracy']=s['negative']/s['decisions']
 s['constant_score_macro_ap']=float(np.mean([d['positive']/d['n'] for d in s['labels'] if d['ap'] is not None]))
 c=s['confusion'];s['balanced_accuracy']=.5*(c[1][1]/sum(c[1])+c[0][0]/sum(c[0]))
sc=groups['showcase'];r['selection_detail']=dict(showcase_positive_people=sorted({rows[j]['subject'] for j in sc if any(l['target']==1 for l in rows[j]['labels'])}),
 showcase_positive_clips=sorted({rows[j]['clip_id'] for j in sc if any(l['target']==1 for l in rows[j]['labels'])}),
 ap_common_class_note='All three showcase AP classes each contain one positive and nine negatives, all within the same side-view clip.')
write(OUT/'report_results.json',r)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#213345','text.color':'#213345','xtick.color':'#536879','ytick.color':'#536879','savefig.facecolor':'white'})
C=['#72899a','#2b8290','#b26a28'];titles=['All test','Quality screened','Curated showcase']
def save(name):
 plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close();shutil.copy2(FIG/(name+'.pdf'),PAPER/'images'/(name+'.pdf'))
fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for n,(metric,title) in enumerate([('macro_ap','Macro average precision'),('micro_accuracy','Decision agreement')]):
 vals=[r[k][metric] for k in groups];ax[n].bar(titles,vals,color=C,width=.6);ax[n].set_ylim(0,1.13);ax[n].set_title(title)
 for j,val in enumerate(vals):ax[n].text(j,val+.035,f'{val:.3f}',ha='center',weight='bold')
 ax[n].set_ylabel('Score');ax[n].tick_params(axis='x',labelsize=9)
ax[0].text(.03,.96,'8 / 7 / 3 AP classes respectively',transform=ax[0].transAxes,va='top',fontsize=9);ax[1].text(.03,.96,'607 / 482 / 102 decisions',transform=ax[1].transAxes,va='top',fontsize=9)
save('selection_effect')
fig,axs=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
for ax,k,title in zip(axs,groups,titles):
 cm=np.array(r[k]['confusion']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(1,cm.max()))
 for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',fontsize=15,color='white' if v>cm.max()*.5 else '#193448')
 ax.set_xticks([0,1],['Negative','Positive']);ax.set_yticks([0,1],['Negative','Positive']);ax.set_xlabel('Predicted');ax.set_ylabel('Automatic target');ax.set_title(title)
save('confusion')
short=['Knee ROM high','Knee asymmetry','Knee bend high','Knee SD high','Trunk frontal ROM','Trunk side ROM','Hip-line ROM','Shoulder-line ROM','Arm ROM high','Arm asymmetry','Knee ROM low','Knee bend low','Arm ROM low']
js=[j for j in range(13) if r['all_test']['labels'][j]['ap'] is not None]
fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(len(js))
for c,(k,title,col) in enumerate(zip(groups,titles,C)):
 vals=[r[k]['labels'][j]['ap'] for j in js];ax.bar(x+(c-1)*.24,[v or 0 for v in vals],width=.22,label=title,color=col)
 for z,v in zip(x+(c-1)*.24,vals):
  if v is None:ax.text(z,.02,'NA',ha='center',rotation=90,fontsize=7)
ax.set_xticks(x,[short[j] for j in js],rotation=25,ha='right');ax.set_ylim(0,1.1);ax.set_ylabel('Average precision');ax.legend(loc='upper left',ncol=3,fontsize=8);save('per_class')
fig,ax=plt.subplots(figsize=(8,3.9),layout='constrained');cols={'ljmu360_P5':'#248b93','ljmu360_P9':'#805db0','progait_1':'#c47c38'}
quality=[w for w in rows if w['quality_pass'] and w['labels']]
rng=np.random.default_rng(20261115)
for who,col in cols.items():
 ww=[w for w in quality if w['subject']==who]
 ax.scatter([w['min_joint_coverage']+rng.uniform(-.002,.002) for w in ww],[np.mean([l['correct'] for l in w['labels']]) for w in ww],s=17,alpha=.4,color=col,label=who+' windows')
 cc=[c for c in r['quality_clips'] if c['subject']==who]
 for c in cc:
  q=[w for w in ww if w['clip_id']==c['clip_id']];ax.scatter(np.mean([w['min_joint_coverage'] for w in q]),c['micro_accuracy'],s=65,marker='D',facecolors='none',edgecolors=col,linewidths=1.6)
 if ww:ax.scatter(np.mean([w['min_joint_coverage'] for w in ww]),np.mean([c['micro_accuracy'] for c in cc]),s=230,marker='*',color=col,edgecolors='white',linewidths=.8,zorder=4)
ax.set_xlabel('Minimum landmark coverage (small display jitter for windows)');ax.set_ylabel('Agreement with automatic criteria');ax.set_ylim(-.05,1.08);ax.legend(loc='lower left',fontsize=8);ax.set_title('91 windows · 14 clip diamonds · 3 person stars',loc='left');save('coverage_scatter')
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for arch,marker,col in [('tcn','o','#248b93'),('motionbert','s','#805db0')]:
 cc=[c for c in r['candidates'] if '_'+arch+'_' in c['run']]
 ax.scatter([c['validation_ap'] for c in cc],[c['quality']['macro_ap'] for c in cc],marker=marker,c=col,s=55,alpha=.8,label=arch.upper())
best=r['candidates'][0];ax.scatter(best['validation_ap'],best['quality']['macro_ap'],s=220,facecolors='none',edgecolors='#c4772a',lw=2);ax.annotate('Selected by validation',(best['validation_ap'],best['quality']['macro_ap']),xytext=(-150,12),textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#c4772a'},fontsize=9)
ax.set_xlabel('Stored validation macro AP');ax.set_ylabel('Quality-screened test macro AP');ax.legend();save('model_comparison')
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for k,title,col in zip(groups,titles,C):
 ix=groups[k];yy=y[ix][mask[ix]];pp=p[ix][mask[ix]];z=binary_metrics(yy,pp);curve=z['precision_recall_curve'];ax.step([d['recall'] for d in curve],[d['precision'] for d in curve],where='post',color=col,label=title)
ax.set_xlim(0,1);ax.set_ylim(0,1.05);ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_title('Pooled decisions · confidence is uncalibrated',loc='left');ax.legend();save('precision_recall')
# Bibliography with frozen provenance; journal issue year supersedes online date.
refs=[]
for key in ['winter1995','nolan2003','delp2007','seth2018','hicks2015','kanko2021','uhlrich2023','saito2015','topham2023']:
 z=json.loads((OUT/'references'/(key+'.json')).read_text());o=z['openalex'];b=o['biblio'];authors=[a['author']['display_name'] for a in o['authorships']]
 # Prefer publisher-deposited author forms over OpenAlex identity display names.
 cr=z.get('crossref') or {};cr=cr.get('message',cr)
 if cr.get('author'):authors=[' '.join(filter(None,[a.get('given'),a.get('family')])) for a in cr['author']]
 refs.append(dict(key=key,title=o['display_name'],authors=authors,year=2015 if key=='hicks2015' else o['publication_year'],journal=o['primary_location']['source']['display_name'],volume=b['volume'],pages=(b['first_page'] or '')+('–'+b['last_page'] if b['last_page'] and b['last_page']!=b['first_page'] else ''),url='https://doi.org/'+z['doi'],citation_count=o['cited_by_count'],count_source=o['id'],retrieved_at=z['retrieved_at']))
refs.extend([
 dict(key='motionbert2023',title='MotionBERT: A Unified Perspective on Learning Human Motion Representations',authors=['Wentao Zhu','Xiaoxuan Ma','Zhaoyang Liu','Libin Liu','Wayne Wu','Yizhou Wang'],year=2023,journal='ICCV',pages='15085–15099',url='https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html'),
 dict(key='progait2025',title='ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users',authors=['Xiangyu Yin','Boyuan Yang','Weichen Liu','Qiyao Xue','Abrar Alamri','Goeran Fiedler','Wei Gao'],year=2025,journal='ICCV',pages='8984–8993',url='https://openaccess.thecvf.com/content/ICCV2025/html/Yin_ProGait_A_Multi-Purpose_Video_Dataset_and_Benchmark_for_Transfemoral_Prosthesis_ICCV_2025_paper.html'),
 dict(key='sapiens2026',title='Sapiens2',authors=['Rawal Khirodkar','He Wen','Julieta Martinez','Yuan Dong','Su Zhaoen','Shunsuke Saito'],year=2026,journal='arXiv:2604.21681',url='https://arxiv.org/abs/2604.21681'),
 dict(key='hirschorn2023',title='Normalizing Flows for Human Pose Anomaly Detection',authors=['Or Hirschorn','Shai Avidan'],year=2023,journal='ICCV',url='https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html')])
write(OUT/'bibliography.json',refs)
def tex(s):return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('–','--').replace('Ł','L').replace('ł','l').replace('ń','n').replace('í','i').replace('é','e').replace('ö','o').replace('ñ','n')
bib=[]
for z in refs:
 author=' and '.join(z['authors']).replace(' and and colleagues',' and others');fields=dict(title='{'+tex(z['title'])+'}',author=tex(author),year=z['year'],journal=tex(z['journal']),url=z['url'])
 if z.get('pages'):fields['pages']=tex(z['pages'])
 if z.get('volume'):fields['volume']=z['volume']
 bib.append('@article{'+z['key']+',\n'+',\n'.join('  '+k+' = {'+str(v)+'}' for k,v in fields.items())+'\n}')
(PAPER/'references.bib').write_text('\n\n'.join(bib))
# Lightweight page payload and local media paths. No remote upload or video copies.
vmap=json.loads((ROOT/'results/motionbert_gait_finetune_v1/completed_run/report_data.json').read_text())['videos']
density=json.loads((ROOT/'results/gait_distribution_20261115/data.json').read_text())
pgids=['PG_outside_1_2_1_f_2_616_691','PG_outside_2_1_2_f_603_726','PG_outside_3_1_4_f_333_475','ljmu360_P4_E3T1_Rear_117_232']
clips=[]
ordered=[r['showcase_clips'][-1]]+r['showcase_clips'][:-1]+[c['clip_id'] for c in r['quality_clips'] if c['clip_id'] not in r['showcase_clips']]
for cid in ordered:
 c=next(c for c in r['quality_clips'] if c['clip_id']==cid);ww=[w for w in rows if w['clip_id']==cid];v=vmap[cid]
 clips.append(dict(**{k:v for k,v in c.items() if k!='windows'},window_count=c['windows'],kind='tcn',curated=cid in r['showcase_clips'],video=v,windows=ww,pose_url='/gait_metrics/'+('harmonized/' if '/harmonized/' in ww[0]['metric_path'] else '')+'clips/'+cid+'/viewer.json'))
for cid in pgids:
 c=next(c for c in density['results']['clips'] if c['clip_id']==cid);ww=json.loads((ROOT/'results/gait_distribution_20261115'/c['predictions']).read_text())['windows']
 clips.append(dict(clip_id=cid,subject=c['subject'],source='progait' if cid.startswith('PG_') else 'ljmu360',view=c['view'],kind='density',curated=True,video=vmap[cid],windows=ww,pose_url='/gait_metrics/'+c['pose_prefix']+'clips/'+cid+'/viewer.json'))
(OUT/'posters').mkdir(exist_ok=True)
for c in clips:
 video=ROOT/'results/gait_metrics_20261114/videos'/c['video']['video'];poster=OUT/'posters'/(c['clip_id']+'.jpg')
 if not poster.exists():subprocess.run(['ffmpeg','-loglevel','error','-ss','0.6','-i',str(video),'-frames:v','1','-vf','scale=480:-2','-q:v','3','-y',str(poster)],check=True)
 c['id']=c['kind']+':'+c['clip_id'];c['poster']='posters/'+poster.name;c['video_url']='/gait_metrics/videos/'+c['video']['video']
write(OUT/'page_data.json',dict(results=r,clips=clips,rows=rows,references=refs,inference=json.loads((OUT/'inference_verification.json').read_text())))
for name,target in [('anatomy_assets',ROOT/'results/gait_experiments_20261115_0502/anatomy'),('paper',PAPER)]:
 if not (OUT/name).exists():(OUT/name).symlink_to(target,target_is_directory=True)
shutil.copy2(ROOT/'scripts/gait_distribution_demo/anatomy.js',OUT/'anatomy.js')
link=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14/gait_selected'
if not link.exists():link.symlink_to(OUT,target_is_directory=True)
print(json.dumps(dict(common_class_ap={k:r[k]['common_class_macro_ap'] for k in groups},baseline_accuracy={k:r[k]['always_negative_accuracy'] for k in groups},clips=len(clips),figures=len(list(FIG.glob('*.png'))),references=len(refs)),indent=2))
