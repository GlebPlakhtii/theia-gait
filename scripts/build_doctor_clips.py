"""Cut every eligible pose recording into prelabelled clips for form review.

No human is asked to annotate timings. Pose measurements are prefilled; only
clinical conclusions and optional corrections are requested. Unsegmented footage
is retained as short context clips, so detection failures are not hidden.
"""
import argparse,csv,json,math,hashlib,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/Theia_Doctor_Clips_2026-11-14'
RUN=ROOT/'results/local_execution_20261114'
SOURCE_MAPPING=ROOT/'audit/doctor_clip_source_mapping_20261114.json'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def tracks(path):
    rows={}
    for r in csv.DictReader(open(path)):
        if r['joint_id'] not in {s+'_'+j for s in ['left','right'] for j in ['hip','knee','ankle']}:continue
        f=int(r['source_frame_index']);q=rows.setdefault(f,{'time':float(r['timestamp_s']),'joints':{}})
        q['joints'][r['joint_id']]=[float(r['x']) if r['x'] else np.nan,float(r['y']) if r['y'] else np.nan,float(r['confidence'] or 0)]
    frames=np.array(sorted(rows));times=np.array([rows[f]['time'] for f in frames])
    angles={};scores={}
    for side in ['left','right']:
        p=np.array([[rows[f]['joints'][side+'_'+joint] for joint in ['hip','knee','ankle']] for f in frames])
        u=p[:,0,:2]-p[:,1,:2];v=p[:,2,:2]-p[:,1,:2]
        den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        angles[side]=np.degrees(np.arccos(np.clip(np.divide((u*v).sum(1),den,out=np.full(len(den),np.nan),where=den>1e-8),-1,1)))
        scores[side]=p[:,:,2].min(1)
    return frames,times,angles,scores


def plan(custom_sources=None, cycle_results=None):
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['clips_original','clips_pose','automatic_labels','provenance']: (OUT/name).mkdir(exist_ok=True)
    allcycles=cycle_results if cycle_results is not None else json.load(open(RUN/'squat_candidates.json'))
    queue={r['clip_id']:r for r in csv.DictReader(open(ROOT/'data/manifests/pose_queue_ready.csv'))}
    records=[];sources=[]
    for source_index,c in enumerate(allcycles):
        cid=c['clip_id'];ared=source_index==0;prefix='ARED' if ared else f'GYM_{source_index:02d}'
        if custom_sources is not None:
            spec=custom_sources[cid];prefix=spec['id'];ared=spec['domain']=='in_mission_ared'
            pose_csv=Path(spec['pose_csv']);raw=Path(spec['raw']);replay=Path(spec['replay'])
        elif ared:
            pose_csv=ROOT/'data/poses/ared_continuous_development/poses.csv'
            raw=ROOT/'deliverables/Theia_Doctor_Review_2026-11-14/videos/01_ARED_development_109s.mp4'
            replay=ROOT/'data/poses/ared_continuous_development/ared_670_779_pose_review.mp4'
        else:
            pose_csv=ROOT/'data/poses/sapiens2_5b_full'/cid/'poses.csv'
            raw=Path(queue[cid]['local_path'])
            replay=ROOT/'data/pose_replays/initial_provisional'/(cid+'_pose_review.mp4')
        frames,times,angles,scores=tracks(pose_csv)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=avg_frame_rate','-of','json',str(raw)]))['streams'][0]
        rate=Fraction(probe['avg_frame_rate']);fps=int(rate) if rate.denominator==1 else float(rate)
        assert np.max(abs(np.diff(times)-1/fps))<1e-5
        source={'id':prefix,'domain':spec['domain'] if custom_sources is not None else ('in_mission_ared' if ared else 'general_fitness_health_status_undocumented'),
                'source_recording_internal':cid,'frames':len(frames),'fps':fps,'source_start_s':float(times[0]),
                'source_frame_first':int(frames[0]),'pose_csv_sha256':digest(pose_csv),
                'original_cut_input':str(raw),'original_cut_input_sha256':digest(raw),
                'pose_cut_input':str(replay),'pose_cut_input_sha256':digest(replay)}
        sources.append(source)
        events=c['parameter_results'][0]['candidates'];segments=[];cursor=0
        def gap(a,b):
            if b<=a:return
            chunk=max(1,int(8*fps))
            for start in range(a,b,chunk):segments.append({'kind':'unsegmented_context','a':start,'b':min(b,start+chunk)})
        for e in events:
            a=e['start_index'];b=e['end_index']
            gap(cursor,a)
            segments.append({'kind':'candidate_repetition','a':a,'b':b,'bottom':e['bottom_index'],'duration_s':e['duration_s']})
            cursor=b
        gap(cursor,len(times))
        coverage=np.zeros(len(times),int)
        for seq,seg in enumerate(segments,1):
            a,b=seg['a'],seg['b'];coverage[a:b]+=1
            # Include the completion frame and 0.32 s context on either side.
            first=max(0,a-round(.32*fps));stop=min(len(times),b+1+round(.32*fps))
            rid=f'{prefix}_{seq:03d}'
            sample=slice(a,min(len(times),b+1));valid=(scores['left'][sample]>=.3)&(scores['right'][sample]>=.3)
            measurement={}
            for side in ['left','right']:
                vals=angles[side][sample];vals=vals[(scores[side][sample]>=.3)&np.isfinite(vals)]
                measurement[side+'_projected_knee_min_deg']=round(float(vals.min()),1) if len(vals) else None
                measurement[side+'_projected_knee_max_deg']=round(float(vals.max()),1) if len(vals) else None
                measurement[side+'_projected_knee_excursion_deg']=round(float(np.ptp(vals)),1) if len(vals) else None
            differences=np.abs(angles['left'][sample]-angles['right'][sample])[valid]
            measurement['median_projected_left_right_difference_deg']=round(float(np.median(differences)),1) if len(differences) else None
            measurement['score_gate_coverage']=round(float(valid.mean()),4)
            event_time=lambda i:round((i-first)/fps,6)
            record={'clip_id':rid,'source_id':prefix,'domain':source['domain'],'automatic_type':seg['kind'],
                    'source_start_s':float(times[first]),'source_last_s':float(times[stop-1]),
                    'source_first_frame':int(frames[first]),'source_last_frame':int(frames[stop-1]),
                    'frames':stop-first,'fps':fps,'video_duration_s':(stop-first)/fps,
                    'core_start_s':event_time(a),'core_end_s':event_time(b),
                    'candidate_bottom_s':event_time(seg['bottom']) if 'bottom' in seg else None,
                    'cycle_duration_s':seg.get('duration_s'),
                    'descent_s':(seg['bottom']-a)/fps if 'bottom' in seg else None,
                    'ascent_s':(b-seg['bottom'])/fps if 'bottom' in seg else None,
                    'automatic_measurements':measurement,
                    'pose_video':f'clips_pose/{rid}.mp4','original_video':f'clips_original/{rid}.mp4',
                    'doctor_conclusion':None,'doctor_notes':'',
                    'source_subset_first_index':first,'source_subset_end_index_exclusive':stop}
            records.append(record)
        assert np.all(coverage==1),'Core segments must cover every source frame exactly once'
        print(prefix,'planned',len(segments),'clips',flush=True)
    write(OUT/'automatic_labels/clips.json',records)
    write(SOURCE_MAPPING,sources)
    write(OUT/'provenance/plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'clips':len(records),
          'candidate_repetitions':sum(r['automatic_type']=='candidate_repetition' for r in records),
          'context_clips':sum(r['automatic_type']=='unsegmented_context' for r in records),
          'unique_source_frames':sum(s['frames'] for s in sources),'sources':[{
              k:v for k,v in s.items() if k not in ['source_recording_internal','original_cut_input','pose_cut_input']}
              for s in sources],
          'method':'Use primary precomputed candidate cycles; split all remaining frames into <=8s cores; add up to 0.32s boundary context. No dropped source frames. Context overlap is not independent data.',
          'clinical_status':'Automatic geometry and segmentation only. No automatic good/bad form or safety judgments.',
          'source_prediction':'Sapiens2-5B, existing completed native-frame outputs',
          'measurements':'2D projected knee angles, viewpoint-dependent; model scores are not calibrated visibility; this does not measure joint loading, pain, fatigue or injury risk.',
          'human_task':'Watch each short cut and write a form conclusion; timestamps and pose labels are already filled.',
          'user_correction':'Replaces full-video timestamp-annotation packet. Gerst reserved recording is not included or inferred.'})


def render_one(r,s,mode):
    dst=OUT/r[mode+'_video']
    first=r.get('source_subset_first_index',r['source_first_frame']-s['source_frame_first'])
    stop=r.get('source_subset_end_index_exclusive',first+r['frames'])
    src=s['pose_cut_input' if mode=='pose' else 'original_cut_input']
    filt=f'trim=start_frame={first}:end_frame={stop},setpts=PTS-STARTPTS'
    if mode=='pose':
        font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        title=r['clip_id']+' | '+('POSE CANDIDATE' if r['automatic_type']=='candidate_repetition' else 'UNSEGMENTED CONTEXT')
        filt+=f",pad=iw:ih+40:0:40:color=0x11232b,drawtext=fontfile={font}:text='{title}':x=8:y=9:fontsize=16:fontcolor=white"
        if r['candidate_bottom_s'] is not None:
            for text,start,end in [('DESCENT',r['core_start_s'],r['candidate_bottom_s']),('ASCENT',r['candidate_bottom_s'],r['core_end_s'])]:
                # The shared bottom frame belongs to ascent only; inclusive
                # bounds on both phases draw two labels on top of each other.
                upper='lt' if text=='DESCENT' else 'lte'
                filt+=f",drawtext=fontfile={font}:text='{text}':x=w-150:y=9:fontsize=16:fontcolor=0xffd47c:enable='gte(t,{start:.6f})*{upper}(t,{end:.6f})'"
    part=dst.with_suffix('.part.mp4')
    if dst.exists():return (r['clip_id'],mode,'already_complete')
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',src,'-map','0:v:0','-vf',filt,
         '-frames:v',str(r['frames']),'-an','-c:v','libx264','-crf','20','-preset','fast','-threads','1',
         '-pix_fmt','yuv420p','-fps_mode','passthrough','-movflags','+faststart','-y',str(part)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(r['clip_id']+' '+mode+' '+result.stderr[-1500:])
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
       'stream=nb_frames,avg_frame_rate,duration','-of','json',str(part)]))['streams'][0]
    assert int(probe['nb_frames'])==r['frames'],(r['clip_id'],probe)
    n,d=map(int,probe['avg_frame_rate'].split('/'));assert abs(n/d-r['fps'])<1e-6
    assert abs(float(probe['duration'])-r['video_duration_s'])<1e-4
    part.rename(dst)
    return (r['clip_id'],mode,'complete')


def render():
    records=json.load(open(OUT/'automatic_labels/clips.json'));sources={s['id']:s for s in json.load(open(SOURCE_MAPPING))}
    count=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(render_one,r,sources[r['source_id']],mode) for r in records for mode in ['original','pose']]
        for job in as_completed(jobs):
            job.result();count+=1
            if count%10==0:print(count,'/',len(jobs),'video files rendered and frame/rate checked',flush=True)
    write(OUT/'provenance/render_complete.json',{'completed_at':datetime.now(timezone.utc).isoformat(),
        'videos':count,'unique_review_clips':len(records),'all_frame_counts_rates_durations_checked':True,
        'script_sha256':digest(__file__),'audio':'Derived clips are silent',
        'video_hashes':{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.glob('clips*/*.mp4'))}})
    print('RENDER COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','render']);a=p.parse_args()
    plan() if a.action=='plan' else render()
