"""Register two acquired lunar development candidates, preserving all other rows."""
import json
from pathlib import Path
from common import read_csv, write_csv, sha256

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=ROOT/'data/manifests/videos.csv'
    old=read_csv(manifest); fields=list(old[0]); rows=[r for r in old if r['dataset_id']!='apollo17_alsj']
    queuepath=ROOT/'data/manifests/pose_queue.csv'
    queue=read_csv(queuepath); qfields=list(queue[0]); queue=[q for q in queue if not q['clip_id'].startswith('apollo17_')]
    for stem in ['a17v.1201208','a17v.1201329']:
        meta=json.loads((ROOT/f'data/metadata/apollo17/{stem}/metadata.json').read_text())
        stream=meta['ffprobe']['streams'][0]
        assert sha256(meta['local_path'])==meta['sha256']
        cid='apollo17_'+stem.replace('.','_')+'_development'
        r={k:'unknown' for k in fields}
        r.update(dataset_id='apollo17_alsj',dataset_version='archive_checked_2026-11-13',source_recording_id='apollo17_EVA1_surface_TV',
            clip_id=cid,participant_id='apollo17_crew_Cernan_Schmitt_unresolved_per_frame',session_id='apollo17_EVA1',cohort='astronaut',
            diagnosis_evidence='not_applicable',domain='in_mission',exercise='lunar_core_sampling_kneel_rise_task',equipment='Apollo_A7LB_suit_and_lunar_tools',
            camera_id='apollo17_surface_TV',view='moving_camera_oblique',simultaneous_capture_group='apollo17_EVA1_surface_TV',
            source_page='https://apollojournals.org/alsj/a17/a17.deepcore.html',download_url=f'https://apollojournals.org/alsj/a17/{stem}.mpg',
            access_checked_at='2026-11-13',access_status='downloaded_tail_decode_warning',license_url='https://apollojournals.org/alsj/a17/video17.html',
            permitted_use_notes='Video library describes public-domain clips; credit NASA and digitization Ken Glover; transcript/editorial rights separate',
            local_path=meta['local_path'],sha256=meta['sha256'],width=stream['width'],height=stream['height'],duration_s=stream['duration'],
            fps_nominal=stream['avg_frame_rate'],time_base=stream['time_base'],variable_fps=str(meta['variable_fps']).lower(),original_start_s=0,original_end_s=180,
            transformation_history='Unmodified MPEG; virtual first-180-second prefix decoded cleanly; full-stream corrupt tail retained; native PTS map; no mission clock alignment',
            annotation_path='',annotation_type='none',annotator_provenance='Codex contact-sheet source screening only',prior_exposure='development_inspected_2026-11-13',
            split='unassigned',exclusion_reason='hold: full-stream corrupt tail; mission clock and overlapping renditions unresolved; no decision reference',
            duplicate_group='apollo17_EVA1_surface_TV',duplicate_review_status='pending; conservatively grouped entire recording',frame_map=meta['frame_map'])
        rows.append(r)
        queue.append(dict(clip_id=cid,local_path=meta['local_path'],sha256=meta['sha256'],start_s=0,end_s=180,frame_map=meta['frame_map'],
            status='hold_corrupt_tail_and_mission_clock_review',transformations=r['transformation_history'],purpose='Lunar scenario development candidate; not squat or held-out evaluation'))
    write_csv(manifest,fields,rows);write_csv(queuepath,qfields,queue)
    print(f'{len(rows)} canonical videos; Apollo candidates held outside ready pose queue')

if __name__=='__main__':main()
