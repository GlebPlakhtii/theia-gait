"""Register supplementary ROM/SD outcomes for every numeric projected family."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gait_metrics_20261114'
FIELDS={
 'M02':('front_rear',['trunk_image_inclination_deg','trunk_horizontal_offset_L0']),
 'M03':('side',['trunk_image_inclination_deg','trunk_horizontal_offset_L0']),
 'M05':('side',[s+'_unsigned_projected_knee_bend_deg' for s in ['left','right']]),
 'M15':('front_rear',['left_up_shoulder_line_deg']+[s+'_shoulder_minus_hip_height_L0' for s in ['left','right']]),
 'M16':('side',[s+'_wrist_minus_shoulder_horizontal_L0' for s in ['left','right']]),
 'M17':('side',[s+'_ear_minus_shoulder_horizontal_L0' for s in ['left','right']]),
}
p=OUT/'extended_outcomes_frozen.json'
if p.exists():raise SystemExit('Supplementary definitions already frozen; not overwritten.')
outcomes=[]
for family,(view,fields) in FIELDS.items():
 for field in fields:
  for statistic in ['rom95','sd']:
   outcomes.append(dict(id=family+'_'+field+'_'+statistic,family=family,view=view,source=f'observation:{field}:{statistic}',
      unit='deg' if field.endswith('_deg') else 'L0',label=family+' · '+field.replace('_',' ')+' · '+statistic.upper(),free_arm_only=family=='M16'))
d=dict(created_at=datetime.now(timezone.utc).isoformat(),registration='Supplementary coverage extension after initial primary interim results; defined before these supplementary group differences.',
 selection_rule='All observation components whose family can return numeric uncertain 2D measurements in gait_geometry.py; report ROM95 and within-segment temporal SD. No selection by observed group effect.',
 matching_and_quality='Unchanged comparison_manifest_frozen.json candidates, exclusions, aggregation and base QC rules.',
 distinction='Supplementary image-plane changes, not new clinical metrics. Absolute posture offsets are not compared because camera roll/orientation and head rotation are uncalibrated.',
 primary_manifest_sha256=hashlib.sha256((OUT/'comparison_manifest_frozen.json').read_bytes()).hexdigest(),outcomes=outcomes)
p.write_text(json.dumps(d,indent=2));(OUT/'extended_outcomes_frozen.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
print('Registered',len(outcomes),'supplementary outcomes')
