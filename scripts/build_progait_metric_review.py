"""Build the standalone ProGait metric-review app; retain the all-video packet."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from scripts.build_unified_doctor_review import link, write

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'deliverables/Theia_All_Videos_Review'
OUT=ROOT/'deliverables/Theia_ProGait_Metric_Review'

def build():
    original=json.loads((SOURCE/'catalog.json').read_text())
    clips=[c for c in original['clips'] if c['packet_key']=='progait']
    assert clips and all(c['exercise']=='walking' for c in clips)
    assert len({c['uid'] for c in clips})==len(clips)
    media=set()
    for clip in clips:
        for key in ('original_video','pose_video','pose_data'):
            if not clip.get(key):continue
            relative=Path(clip[key])
            assert not relative.is_absolute() and '..' not in relative.parts
            link(SOURCE/relative,OUT/relative)
            media.add(str(relative))
    catalog=dict(schema_version=4,generated_at=datetime.now(timezone.utc).isoformat(),
        groups={'clinical_prosthetic_gait':'ProGait · clinical prosthetic gait'},
        packages=[p for p in original['packages'] if p['key']=='progait'],clips=clips,
        status_text=f"{len(clips)} ProGait walking clips · {len({c['source_id'] for c in clips})} source videos · annotate each metric independently; partial reviews are saved.")
    write(OUT/'catalog.json',json.dumps(catalog,indent=2,allow_nan=False))
    write(OUT/'catalog.js','window.THEIA_CATALOG='+json.dumps(catalog,separators=(',',':'),allow_nan=False)+';\n')
    for source,target in [('progait_review.html','index.html'),('progait_review.js','review.js'),('progait_metrics.js','metrics.js')]:
        write(OUT/target,(ROOT/'scripts'/source).read_text())
    write(OUT/'START_HERE.txt',f'''THEIA — PROGAIT MOVEMENT AND METRIC REVIEW
{len(clips)} walking clips from 412 ProGait source videos. Open index.html or use the local server.
Choose an overall visible-form conclusion. Expand each measurement under
Pose and automatic measurements and add an assessment, optional corrected
value in the displayed units, and/or notes. Original measurements are preserved.
Only projected left/right knee minimum and maximum are shown and annotated.
Minimum and maximum are grouped in one section per knee. Other source metrics
remain in source metadata but have no annotation fields in this app.
Invalid numeric corrections are preserved as text drafts with an error; they
are not exported as valid corrected values. Correct them before final submission.

Autosave uses a separate browser key from the all-video app. Export annotations
to JSON to share or back up. Import restores these annotations and accepts
older ProGait/unified review exports; only ProGait entries from unified reviews
are imported. Conflicting entries are kept for explicit resolution.
Moving this folder does not move browser autosave. Export before changing browsers.
Next autoplays. Original RGB and Show pose are two views of the same review item.
No information is sent automatically. Published gait assessments are not prefilled.
The 2D measurements do not correct for camera perspective or prosthetic geometry.

CREDITS AND LICENSE
Yin et al., ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral
Prosthesis Users, ICCV 2025. CC BY-NC-SA 4.0.
https://huggingface.co/datasets/ericyxy98/ProGait
https://creativecommons.org/licenses/by-nc-sa/4.0/
Derived review videos are silent time-window cuts. Source privacy masks retained.
Poses are publisher RTMW-assisted annotations with manual corrections, not
new Sapiens2 output. Derived ProGait review media retain CC BY-NC-SA 4.0.
''')
    audit=ROOT/'audit/progait_metric_review';audit.mkdir(parents=True,exist_ok=True)
    report=dict(clips=len(clips),sources=len({c['source_id'] for c in clips}),media_assets=len(media),
        output=str(OUT),built_at=catalog['generated_at'],storage_key='theia-progait-metric-review-v1',
        files={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['index.html','review.js','metrics.js','catalog.json']})
    write(audit/'build_report.json',json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
