"""Include the five limited-view/treadmill intake excerpts without invented poses."""
import concurrent.futures as cf
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/Theia_Doctor_Limited_Clips_2026-11-14'
INCLUDED={'P17_INSET_01','THIRSK_PARTIAL_01','GLOVER_PARTIAL_01','RUBIO_TREADMILL_01','GERST18_TREADMILL_01'}

def process(s):
    fps=Fraction(s['native_rate']);n=s['frames'];count=math.ceil(n/math.floor(float(fps)*8))
    edges=[i*n//count for i in range(count+1)];rows=[]
    for number,(start,end) in enumerate(zip(edges,edges[1:]),1):
        cid=s['id']+f'_CONTEXT_{number:03d}';target=OUT/'clips_original'/(cid+'.mp4');target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',s['prepared_path'],
            '-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS','-frames:v',str(end-start),
            '-an','-c:v','libx264','-crf','20','-preset','veryfast','-threads','1','-fps_mode','passthrough',
            '-movflags','+faststart','-n',str(target)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,avg_frame_rate','-of','json',str(target)]))['streams'][0]
        assert int(probe['nb_read_frames'])==end-start and Fraction(probe['avg_frame_rate'])==fps
        rows.append(dict(clip_id=cid,source_id=s['id'],domain='ground_ared_limited' if s['id']=='GLOVER_PARTIAL_01' else s['domain'],
            exercise='walking' if 'treadmill' in s['domain'] else 'squat',automatic_type='unsegmented_context',
            source_first_frame=s['original_first_frame']+start,source_last_frame=s['original_first_frame']+end-1,
            source_start_s=(s['original_first_frame']+start)/float(fps),source_last_s=(s['original_first_frame']+end-1)/float(fps),
            frames=end-start,fps=float(fps),video_duration_s=(end-start)/float(fps),core_start_s=0,core_end_s=(end-start-1)/float(fps),
            candidate_bottom_s=None,cycle_duration_s=None,descent_s=None,ascent_s=None,automatic_measurements={},
            original_video=str(target.relative_to(OUT)),pose_video=None,doctor_conclusion=None,doctor_notes='',
            context=s['notes'],pose_provenance='No accepted full-chain pose output. Limited view or separate treadmill task.',source_url=s['source_page']))
    return rows

def main():
    selections=[s for s in json.load(open(ROOT/'data/prepared/exercise_intake_20261114/selections.json')) if s['id'] in INCLUDED]
    assert len(selections)==5
    with cf.ThreadPoolExecutor(max_workers=2) as pool:rows=[r for group in pool.map(process,selections) for r in group]
    (OUT/'automatic_labels').mkdir(parents=True,exist_ok=True)
    (OUT/'automatic_labels/clips.json').write_text(json.dumps(rows,indent=2))
    (OUT/'START_HERE.txt').write_text('Limited-view and treadmill excerpts for optional expert review.\nCuts cover every selected source frame, in windows of at most 8 seconds.\nNo complete lower-body pose is claimed. Cropped anatomy, low resolution or the separate treadmill task limits interpretation.\nUse Cannot judge where appropriate. Source descriptions and credits are retained per clip.\nNASA / ESA / CSA source terms apply.\n')
    print('Prepared',len(rows),'limited-view review clips',flush=True)

if __name__=='__main__':main()
