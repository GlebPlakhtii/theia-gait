"""Publication architecture figure and exact paired-case provenance."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'deliverables/Theia_Selected_Gait_Paper_2026-11-15'
REPORT=ROOT/'results/gait_selected_paper_20261115'
DENSITY=ROOT/'results/gait_distribution_20261115'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9,7.5));fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
ax.set_xlim(0,9);ax.set_ylim(0,7.5);ax.axis('off')
ink='#173246';blue='#266f98';green='#187d72';grey='#63798b'
def box(x,y,w,h,title,body,edge=grey,fill='#f3f6f8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.09',lw=1.1,ec=edge,fc=fill))
 ax.text(x+w/2,y+h-.12,title,ha='center',va='top',fontsize=11,weight='bold',color=ink)
 ax.text(x+w/2,y+h-.39,body,ha='center',va='top',fontsize=10,color=ink,linespacing=1.3)
def arrow(a,b,color=grey,style='-'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color,linestyle=style,connectionstyle='angle3,angleA=90,angleB=0' if a[0]!=b[0] and a[1]!=b[1] else 'arc3'))
box(.14,6.52,2.55,.87,'RGB walking video','Original frames + timestamps')
box(3.22,6.52,2.55,.87,'Sapiens2-1B','Cached 2D pose + confidence')
box(6.30,6.52,2.55,.87,'17-joint adapter','Explicit validity; 1 s windows')
arrow((2.75,6.96),(3.16,6.96));arrow((5.83,6.96),(6.24,6.96))
# One shared pose input forks into two independent analysis branches.
ax.plot([7.58,7.58,2.20],[6.47,6.20,6.20],color=grey,lw=1.2)
arrow((2.20,6.20),(2.20,5.60),blue)
arrow((6.80,6.20),(6.80,5.60),green)
ax.text(2.20,5.91,'A   Supervised model variants',ha='center',va='center',color=blue,fontsize=12,weight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=5))
ax.text(6.80,5.91,'B   Regional density model',ha='center',va='center',color=green,fontsize=12,weight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=5))
boxesA=[('Pose + optional geometry','30 × 17 × 4; 8 context indicators\nGeometry: 220/frame; pooling: 169/frame'),('Alternative encoders','Raw/geometry TCN: 4 temporal blocks\nStatistical model: 96D frame embeddings'),('Pooling + task heads','TCN: mean/max; statistical: mean/SD/min/max\n128D readout → class/numeric/evaluability'),('Regional scores + evaluation','Three-seed original-split ensembles\nThreshold 0.5; unsupported labels masked')]
boxesB=[('View and quality gates','Seven regions; confidence and coverage'),('Motion features','Median angle; P95–P05 range\nTemporal SD; median angular speed'),('Weighted 2-component GMM','Separate region/view fits\nlog(1 + feature); training-set scaling'),('Calibrated deviation rank','Likelihood rank ≥95th percentile\nSeparate calibration people; qualitative cases')]
for x,xx,boxes,col,fill in [(.14,2.20,boxesA,blue,'#edf5fa'),(4.82,6.80,boxesB,green,'#edf7f3')]:
 for i,(t,b) in enumerate(boxes):
  y=4.70-i*1.04;box(x,y,4.03,.84,t,b,col,fill)
  if i<3:arrow((xx,y-.04),(xx,y-.16),col)
 arrow((xx,1.53),(xx,1.10),col)
box(.14,.19,8.71,.84,'Synchronized regional review','RGB + pose overlay  |  2D skeleton  |  generic OpenSim anatomy\nRegion colors, scores and original-source frames',grey,'#f2f4f7')
ax.text(4.5,.04,'Amputation status is not an input. Anatomy colors localize signals; they do not estimate muscle force.',ha='center',fontsize=9.3,color=grey)
fig.savefig(PAPER/'images/architecture.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(PAPER/'images/architecture.png',dpi=200,bbox_inches='tight',pad_inches=.03)
plt.close(fig)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((DENSITY/'data.json').read_text());models=json.loads((DENSITY/'models.json').read_text())
specs=[('A','PG_outside_1_2_1_f_2_616_691',1.2,'with amputation'),('B','ljmu360_P4_E3T1_Rear_117_232',3.3,'without amputation')]
cases=[]
for panel,cid,time,cohort in specs:
 c=next(c for c in d['results']['clips'] if c['clip_id']==cid);pred=DENSITY/c['predictions'];ws=json.loads(pred.read_text())['windows'];w=max((w for w in ws if w['start_time_s']<=time<w['end_time_s']),key=lambda w:w['start_time_s'])
 assert all(e['percentile'] is not None for e in w['events'])
 flags=[e['region'] for e in w['events'] if e['percentile']>=95]
 assert flags==(['left_knee','left_arm'] if panel=='A' else [])
 v=d['videos'][cid]
 cases.append(dict(panel=panel,clip_id=cid,subject=c['subject'],cohort=cohort,fold_index=c['fold'],display_fold=c['fold']+1,
  view=c['view'],requested_frame_time_s=time,expected_source_frame=v['source_first_frame']+int(time*v['fps']),
  window_key=w['key'],window_start_s=w['start_time_s'],window_end_s=w['end_time_s'],
  original_source_first_frame=w['original_source_first_frame'],original_source_end_frame_exclusive=w['original_source_end_frame_exclusive'],
  flags=flags,regions=[{k:e[k] for k in ['region','percentile','model_key','coverage','training_people','calibration_people']} for e in w['events']],
  prediction_file=str(pred.relative_to(ROOT)),prediction_sha256=sha(pred),video=v,clinical_form_label=None))
assert cases[0]['fold_index']==cases[1]['fold_index']==3
assert [e['model_key'] for e in cases[0]['regions']]==[e['model_key'] for e in cases[1]['regions']]
result=dict(selection='Post hoc illustrative flagged/unflagged pair; same view context, fold, trained distributions, calibration people and threshold.',
 threshold_percentile=95,clinical_validation=False,models_sha256=sha(DENSITY/'models.json'),fold=d['results']['folds'][3]['split'],cases=cases,
 interpretation='The LJMU example is a walking reference from the non-amputee source cohort; no model flag does not independently certify correct form. Cohort is a caption, never an input.')
capture_path=PAPER/'evidence/paired_case_capture.json'
if capture_path.exists():
 captures=json.loads(capture_path.read_text())
 for case in cases:
  capture=captures[case['panel']]
  assert capture['clip']=='density:'+case['clip_id']
  assert case['window_start_s']<=capture['time']<case['window_end_s']
  case['captured_time_s']=capture['time']
  case['captured_source_frame']=case['video']['source_first_frame']+int(capture['time']*case['video']['fps'])
  case['capture_metadata']='paired_case_capture.json'
 result['capture_metadata_sha256']=sha(capture_path)
(PAPER/'evidence/paired_case_provenance.json').write_text(json.dumps(result,indent=2));(REPORT/'paired_case_provenance.json').write_text(json.dumps(result,indent=2))
print(json.dumps({c['panel']:{'clip':c['clip_id'],'window':[c['window_start_s'],c['window_end_s']],'max_rank':max(e['percentile'] for e in c['regions']),'flags':c['flags']} for c in cases},indent=2))
