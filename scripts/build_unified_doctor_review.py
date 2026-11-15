"""Build one offline review page without changing prior packets or human labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/Theia_All_Videos_Review'
AUDIT=ROOT/'audit/unified_review_20261114'
PACKETS=[
 ('original','Theia_Doctor_Clips_2026-11-14','theia-doctor-short-clips-20261114-v2'),
 ('gerst','Theia_Doctor_Gerst_Clips_2026-11-14','theia-doctor-gerst-short-clips-20261114-v1'),
 ('extra','Theia_Doctor_Extra_Clips_2026-11-14','theia-doctor-extra-clips-20261114-v1'),
 ('limited','Theia_Doctor_Limited_Clips_2026-11-14','theia-doctor-limited-clips-20261114-v1'),
 ('progait','Theia_Doctor_ProGait_2026-11-14','theia-doctor-progait-20261114-v1'),
]
GROUPS={'in_mission_ared':'In-mission ARED','ground_ared':'Ground ARED',
        'gym':'Gym · health undocumented','terrestrial_gym':'Gym · health undocumented',
        'gym_squat':'Gym · health undocumented','in_mission_treadmill':'In-mission treadmill',
        'partial_view_ared':'ARED · limited view','ground_ared_limited':'Ground ARED · limited view',
        'clinical_prosthetic_gait':'ProGait · clinical prosthetic gait'}


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()==text:return
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text);temp.replace(path)


def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if source.stat().st_ino==target.stat().st_ino:return
        if source.stat().st_size==target.stat().st_size and hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest():return
        raise ValueError('Changed media at existing stable clip ID: '+str(target))
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def snapshot():
    return [ROOT/'deliverables'/folder/'automatic_labels/clips.json' for _,folder,_ in PACKETS]+[
        ROOT/'audit/new_video_intake_20261114/pipeline_status.json',ROOT/'audit/progait_20261114/preparation_status.json']


def build():
    clips=[];packages=[]
    for key,folder,package_id in PACKETS:
        source=ROOT/'deliverables'/folder;manifest=source/'automatic_labels/clips.json'
        if not manifest.exists():continue
        rows=json.loads(manifest.read_text())
        # Extra packet publishes its manifest before rendering. Include only verified existing media.
        ready=[]
        for original in rows:
            c=dict(original);media=[c[f] for f in ['original_video','pose_video','pose_data'] if c.get(f)]
            if not all((source/f).is_file() for f in media):continue
            for field in ['original_video','pose_video','pose_data']:
                if c.get(field):
                    relative=Path(c[field]);assert not relative.is_absolute() and '..' not in relative.parts
                    destination=OUT/'media'/key/relative;link(source/relative,destination)
                    c[field]=destination.relative_to(OUT).as_posix()
            c.update(uid=key+':'+c['clip_id'],package_id=package_id,packet_key=key)
            if c['source_id'].startswith('GYM'):c['domain']='gym'
            c.setdefault('exercise','walking' if 'treadmill' in c['domain'] else 'squat')
            if key in ['original','gerst','extra']:
                c['pose_provenance']='Sapiens2-5B predictions; native frames; automatic cycle boundaries and angles.'
            ready.append(c)
        clips+=ready;packages.append(dict(key=key,package_id=package_id,ready_clips=len(ready)))
        guide=source/'START_HERE.txt'
        if guide.exists():write(OUT/'provenance'/(key+'_guide.txt'),guide.read_text())
        for provenance in (source/'provenance').glob('*.json'):
            # Source audit only; no publisher gait conclusions are copied to the doctor's page.
            if key!='progait':write(OUT/'provenance'/key/provenance.name,provenance.read_text())
        if key!='progait':
            for provenance in (source/'provenance').rglob('*.csv'):
                write(OUT/'provenance'/key/provenance.relative_to(source/'provenance'),provenance.read_text())
    source_audit=ROOT/'audit/progait_20261114/final_source_audit.json'
    if source_audit.exists():write(OUT/'provenance/progait_source_audit.json',source_audit.read_text())
    if not clips:raise ValueError('No review-ready media')
    assert len({c['uid'] for c in clips})==len(clips)
    groups={**GROUPS,**{c['domain']:GROUPS.get(c['domain'],c['domain'].replace('_',' ')) for c in clips}}
    statuses={}
    for name,path in [('additional_ared',snapshot()[-2]),('progait',snapshot()[-1])]:
        if path.exists():statuses[name]=json.loads(path.read_text())
    status_text=f'{len(clips)} review clips available. Each item is reviewed once, using original or pose view.'
    if statuses.get('additional_ared',{}).get('status')!='complete':status_text+=' Additional ARED pose processing is still running.'
    ps=statuses.get('progait',{})
    if ps.get('status')!='complete':status_text+=f' ProGait preparation: {ps.get("complete_sources",0)}/412 sources ready; this page adds completed clips when refreshed.'
    else:status_text+=' All 412 ProGait sources are prepared using publisher pose annotations.'
    catalog=dict(schema_version=3,generated_at=datetime.now(timezone.utc).isoformat(),groups=groups,packages=packages,
                 clips=clips,status_text=status_text,pipeline_status=statuses)
    write(OUT/'catalog.js','window.THEIA_CATALOG='+json.dumps(catalog,separators=(',',':'),allow_nan=False)+';\n')
    write(OUT/'catalog.json',json.dumps(catalog,indent=2,allow_nan=False))
    for src,dest in [('doctor_review.html','index.html'),('doctor_review.js','review.js')]:write(OUT/dest,(ROOT/'scripts'/src).read_text())
    write(OUT/'REVIEW_ALL_VIDEOS.html','<!doctype html><meta http-equiv="refresh" content="0;url=index.html"><a href="index.html">Open all video reviews</a>')
    write(OUT/'START_HERE.txt',f'''THEIA — ONE PAGE FOR ALL VIDEO REVIEWS
Extract the complete folder and open index.html in Chrome, Edge or Firefox.
{len(clips)} review items are currently included. Original and pose views belong
to one item; enter one conclusion for that item. No timestamps or skeleton
annotation are needed. Choose a conclusion, add optional observations, then
Save & next. Export one JSON regularly and return it to Hlib Plakhtii / Theia.

Import saved review accepts the previous ARED/gym, Gerst and extra-packet JSON
exports. Existing conclusions are merged. Conflicting conclusions are preserved
and shown for explicit resolution; exports retain unresolved conflicts.
Browser autosave depends on browser and folder location. Moving the folder or
sending a ZIP does not transfer browser storage: always export the review JSON.

In-mission ARED, ground ARED, limited views, treadmill, gym and ProGait walking
are separate groups. Gym health status is undocumented. ProGait subjects have
transfemoral prostheses; this is clinical gait footage, not matched healthy squat
footage. Fixed gait windows are not stride detections. Context cuts can divide
movements. Use Cannot judge where visible context does not support a conclusion.

ARED/gym overlays: Sapiens2-5B estimates. ProGait overlays: publisher-provided
RTMW-assisted annotations, with manual correction, NOT our Sapiens2 predictions.
The metadata names the pose source. Published ProGait gait descriptions are kept
in the separate research data folder, not displayed here or prefilled as doctor
conclusions. Model-assisted review is not a blinded pose-accuracy benchmark.

All views preserve source privacy masks. No form, diagnosis, pain, injury or
medical-safety verdict is inferred. Resistance and prescriptions are unknown.

CREDITS / SOURCE TERMS
ARED and station footage: NASA, ESA/NASA, CSA/NASA. Gym: original source authors;
see provenance/original_guide.txt and packet provenance. Source terms still apply.
ProGait: Yin et al., ProGait: A Multi-Purpose Video Dataset and Benchmark for
Transfemoral Prosthesis Users, ICCV 2025. CC BY-NC-SA 4.0. These silent cuts and
browser overlays are derived review copies; attribution and ShareAlike retained.
https://huggingface.co/datasets/ericyxy98/ProGait
https://creativecommons.org/licenses/by-nc-sa/4.0/

This folder works offline. Nothing is uploaded or sent automatically.
''')
    AUDIT.mkdir(parents=True,exist_ok=True)
    report=dict(review_items=len(clips),source_videos=len({c['source_id'] for c in clips}),
                groups={g:sum(c['domain']==g for c in clips) for g in groups if any(c['domain']==g for c in clips)},
                packages=packages,media_paths_valid=True,duplicate_uids=0,
                watch_time_seconds=sum(c['video_duration_s'] for c in clips),built_at=catalog['generated_at'])
    write(AUDIT/'build_report.json',json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return statuses


def archive():
    target=OUT.with_suffix('.zip');tmp=target.with_suffix('.zip.part')
    with zipfile.ZipFile(tmp,'w',allowZip64=True) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)),compress_type=zipfile.ZIP_STORED if p.suffix=='.mp4' else zipfile.ZIP_DEFLATED)
    tmp.replace(target)
    with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    write(AUDIT/'archive.json',json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=sha),indent=2))
    print('Archive ready:',target,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--zip',action='store_true');a=parser.parse_args()
    fingerprint=None
    while True:
        now=tuple((str(p),p.stat().st_mtime_ns) for p in snapshot() if p.exists())
        if now!=fingerprint:
            statuses=build();fingerprint=now
        finished=all(statuses.get(k,{}).get('status')=='complete' for k in ['additional_ared','progait'])
        if a.zip and (finished or not a.watch):archive()
        if not a.watch or finished:break
        time.sleep(30)


if __name__=='__main__':main()
