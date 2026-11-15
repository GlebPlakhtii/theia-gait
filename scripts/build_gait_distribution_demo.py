from pathlib import Path
import json,shutil
from motionbert_gait_common import write,now,sha
from gait_distribution_model import REGIONS,TITLES,FEATURES
R=Path(__file__).resolve().parents[1];P=R/'results/gait_distribution_20261115'
result=json.load(open(P/'results.json'));audit=json.load(open(P/'audit.json'));old=json.load(open(R/'results/motionbert_gait_finetune_v1/completed_run/report_data.json'))
# Display selection only; dataset names and identities never enter the fitted distributions.
default='PG_outside_1_2_1_f_2_616_691';assert any(c['clip_id']==default for c in result['clips'])
write(P/'data.json',dict(created_at=now(),results=result,audit=audit,models=json.load(open(P/'models.json')),regions=REGIONS,titles=TITLES,features=FEATURES,default_clip=default,
 default_selection='Previously used ProGait participant-1 frontal example, fixed for continuity; not selected by maximum anomaly score.',videos={c['clip_id']:old['videos'][c['clip_id']] for c in result['clips']}))
for name in ['index.html','style.css','app.js','anatomy.js']:shutil.copyfile(R/'scripts/gait_distribution_demo'/name,P/name)
assets=P/'anatomy_assets'
if not assets.exists():assets.symlink_to(R/'results/gait_experiments_20261115_0502/anatomy',target_is_directory=True)
route=R/'deliverables/Theia_Squat_Form_Lab_2026-11-14/gait_distribution'
if not route.exists():route.symlink_to(P,target_is_directory=True)
write(P/'demo_manifest.json',dict(created_at=now(),url='http://localhost:8006/gait_distribution/',additional_demo=True,previous_report_untouched=True,
 clinical_validation=False,files={n:sha(P/n) for n in ['index.html','style.css','app.js','anatomy.js','data.json','models.json','protocol.json','audit.json']},browser_verification='pending'))
(P/'README.txt').write_text('THEIA — ADDITIONAL MOVEMENT DISTRIBUTION DEMO\n\nOpen http://localhost:8006/gait_distribution/ using the existing port-8006 server.\n\nPooled, cohort-blind weighted regional Gaussian-mixture model. Old teacher targets, healthy-only thresholds and amputation status are not used. Five participant folds; separate training, calibration and test people. This model estimates rarity, not clinical wrongness.\n\n106 matched overground-walking clips: 93 identified-person test clips and 13 exploratory external clips with unresolved identity. Other views and activities are outside this reference population.\n\nThe anatomy map uses generic OpenSim Gait2392 bones and muscle paths with schematic arm extensions. Colors are regional deviation scores, not estimated muscle activation.\n\nTraining and audit scripts: scripts/train_gait_distribution.py, scripts/gait_distribution_model.py, scripts/audit_gait_distribution.py.\nBuilder: scripts/build_gait_distribution_demo.py; UI: scripts/gait_distribution_demo/.\n\nThe demo reuses existing local videos/poses and anatomy assets through symlinks. This folder alone is not a portable video package. No data were uploaded.\n')
print('http://localhost:8006/gait_distribution/')
