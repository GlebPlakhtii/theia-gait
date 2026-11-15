"""Materialize reviewed shots, preserving rational cadence and source-frame lineage.

Rotations are rigid image orientation changes, not perspective/3D correction.
No infilling, stabilization, temporal interpolation or generated pixels.
"""
import csv
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from prepare_videos import inspect
from common import write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/new_video_intake_20261114'
OUT = ROOT / 'data/prepared/exercise_intake_20261114'

SELECTIONS = [
    dict(id='P21_ARED_01', source='space_workout', start=1., end=25., domain='in_mission_ared', rotation='ccw90', roi=[.20,.22,.86,.91], pose=True,
         notes='Continuous ARED movement; single person; small ESA logo away from body. Equipment occlusion and oblique view remain. Task/cycle predictions require review.'),
    dict(id='HOP_GROUND_01', source='hopkins_ground', start=60.5, end=69.5, domain='ground_ared', roi=[.37,.12,.81,.99], pose=True,
         notes='Wide squat view before cut at 69.67s. Foreground exerciser; coach behind/left. Auxiliary ground footage, not mission target.'),
    dict(id='WIL_GROUND_01', source='williams_ground', start=77., end=91.5, domain='ground_ared', roi=[.28,.15,.72,.99], pose=True,
         notes='Wide rear-oblique squat view before camera moves closer. Coach/background people excluded from intended track. Auxiliary ground footage.'),
    dict(id='P17_INSET_01', source='fitness_2017', start=127., end=130.4, domain='in_mission_ared', crop=[.68,0,1,1], roi=[0,0,1,1], pose=True,
         notes='Only exercise inset retained; speaking-person panel removed. Low spatial resolution; short fragment, possibly incomplete cycle; secondary quality tier.'),
    dict(id='CSA_ARED_01', source='saint_jacques_2019', start=76.3, end=79.4, domain='in_mission_ared', roi=[.49,.03,.84,.99], pose=True,
         notes='Short full-body squat b-roll after fade-in, before cut. Exerciser identity not independently verified; do not assume narrator Saint-Jacques. Feet near frame edge, oblique view, secondary quality tier.'),
    dict(id='THIRSK_PARTIAL_01', source='thirsk_ared_2009', start=0., end=54., domain='partial_view_ared', pose=False,
         notes='Continuous in-mission squats, but feet/ankles outside image and hip partially occluded by ARED. Excluded from full-chain knee-angle queue; useful for limited upper-body review.'),
    dict(id='GLOVER_PARTIAL_01', source='glover_ground', start=52., end=60., domain='partial_view_ared', pose=False,
         notes='Ground ARED squat view. Legs/ankles cropped; do not invent complete hip-knee-ankle measurements.'),
    dict(id='RUBIO_TREADMILL_01', source='rubio_treadmill', start=123., end=129.8, domain='in_mission_treadmill', rotation='cw90', pose=False,
         notes='Treadmill b-roll between edits at 122.155s and 130.230s. Separate task; runner visibility and exact task identity require task-specific tracking.'),
    dict(id='GERST18_TREADMILL_01', source='gerst_2018', start=21.5, end=27.3, domain='in_mission_treadmill', pose=False,
         notes='Treadmill shot before edit at 27.928s. Different mission from Gerst 2014; same crew identity, not crew-disjoint.'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records=[]; queue=[]; rois={'clips':{}}
    for spec in SELECTIONS:
        src=json.loads((AUDIT/(spec['source']+'_acquisition.json')).read_text())
        meta=json.loads(Path(src['metadata']).read_text())
        stream=next(s for s in meta['ffprobe']['streams'] if s['codec_type']=='video')
        mapping=list(csv.DictReader(open(meta['frame_map'])))
        selected=[r for r in mapping if spec['start']<=float(r['timestamp_s'])<spec['end']]
        first=int(selected[0]['source_frame_index']); stop=int(selected[-1]['source_frame_index'])+1
        folder=OUT/spec['domain'];folder.mkdir(exist_ok=True)
        video=folder/(spec['id']+'.mp4');filt=f'trim=start_frame={first}:end_frame={stop},setpts=PTS-STARTPTS'
        transform=[]
        if 'crop' in spec:
            x0,y0,x1,y1=spec['crop'];w,h=stream['width'],stream['height']
            x0=int(x0*w)//2*2;y0=int(y0*h)//2*2;x1=int(x1*w)//2*2;y1=int(y1*h)//2*2
            filt+=f',crop={x1-x0}:{y1-y0}:{x0}:{y0}'
            transform.append(dict(operation='crop',xyxy_pixels=[x0,y0,x1,y1],inverse='source_x=prepared_x+x0; source_y=prepared_y+y0'))
        if spec.get('rotation'):
            direction=2 if spec['rotation']=='ccw90' else 1;filt+=f',transpose={direction}'
            transform.append(dict(operation=spec['rotation'],source_width=stream['width'],source_height=stream['height'],
                                  inverse=('source_x=source_width-1-prepared_y; source_y=prepared_x' if direction==2 else 'source_x=prepared_y; source_y=source_height-1-prepared_x')))
        if not video.exists():
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','2',
                '-i',src['local_path'],'-map','0:v:0','-vf',filt,'-frames:v',str(len(selected)),
                '-an','-c:v','libx264','-crf','18','-preset','fast','-threads','2','-fps_mode','passthrough',
                '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video)],check=True)
        saved_meta=folder/spec['id']/'metadata.json'
        prepared=json.loads(saved_meta.read_text()) if saved_meta.exists() else inspect(video,folder/spec['id'])
        assert prepared['decoded_frames']==len(selected)
        fps=Fraction(stream['avg_frame_rate']);end=len(selected)/float(fps)
        assert abs(prepared['median_frame_interval_s']-1/float(fps))<1e-6
        pmap=list(csv.DictReader(open(prepared['frame_map'])))
        assert all(abs(float(p['timestamp_s'])-(float(o['timestamp_s'])-float(selected[0]['timestamp_s'])))<2e-6 for p,o in zip(pmap,selected))
        lineage=[dict(prepared_frame_index=i,prepared_timestamp_s=pmap[i]['timestamp_s'],
                      original_frame_index=r['source_frame_index'],original_timestamp_s=r['timestamp_s'],original_pts_s=r['source_pts_s']) for i,r in enumerate(selected)]
        write_csv(folder/spec['id']/'original_frame_lineage.csv',list(lineage[0]),lineage)
        record=dict(spec,original_sha256=src['sha256'],original_path=src['local_path'],source_page=src['page'],
                    original_first_frame=first,original_last_frame=stop-1,frames=len(selected),duration_s=end,
                    native_rate=str(fps),prepared_path=str(video),prepared_sha256=prepared['sha256'],
                    transforms=transform,clinical_label=None)
        records.append(record);write_json(folder/spec['id']/'provenance.json',record)
        if spec['pose']:
            queue.append(dict(clip_id=spec['id'],local_path=str(video),sha256=prepared['sha256'],start_s=0,end_s=end,
                              frame_map=prepared['frame_map'],status='prepared_for_pilot',transformations=json.dumps(transform)))
            rois['clips'][spec['id']]={'bbox_normalized_xyxy':spec['roi']}
        print(spec['id'],len(selected),float(fps),flush=True)
    write_json(OUT/'selections.json',records)
    write_csv(OUT/'pose_queue.csv',list(queue[0]),queue)
    write_json(OUT/'pose_rois.json',rois)


if __name__=='__main__':main()
