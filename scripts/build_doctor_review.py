"""Build a portable, prediction-blinded review packet from local mission RGB."""
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/Theia_Doctor_Review_2026-11-14'
PACKAGE='theia-doctor-review-20261114-v1'
CLIPS=[
 {'id':'ARED_DEV_01','title':'01 · ARED development segment · 1 min 49 s',
  'path':'videos/01_ARED_development_109s.mp4','duration_s':109.0,'frames':2725,
  'source_start_s':670.0,'source_first_frame':16750,'nominal_fps':25,
  'role':'Development / rubric calibration; previously used for model development.',
  'source_person':'Sophie Adenot','mission_year':2026,
  'source_context':'ESA catalog describes EasyMotion-2 exercise on ARED. Electrical stimulation status, resistance and personal prescription are unknown.',
  'source_url':'https://dlmultimedia.esa.int/download/public/videos/2026/02/039/2602_039_BR_118.mp4',
  'source_sha256':'f5a6355bb87ee367caa2c3ba9b3e7803f6f538e71c41627025ab42788b480fb1',
  'transformation':'Frame-preserving selection of source frames 16750–19474 inclusive; H.264 CRF18 re-encode, no crop/scale/overlay; corresponding source audio retained and re-encoded.',
  'credit':'ESA/NASA'},
 {'id':'ARED_RESERVE_02','title':'02 · ARED reserved recording · 12 min 29 s',
  'path':'videos/02_ARED_reserved_full.mp4','duration_s':749.44,'frames':18736,
  'source_start_s':0.0,'source_first_frame':0,'nominal_fps':25,
  'role':'Reserved candidate; no pose inference or RGB review by the current modeling workflow. Not yet an eligible frozen test set; doctor must first identify tasks and assess suitability.',
  'source_person':'Alexander Gerst','mission_year':2014,
  'source_context':'ESA catalog describes Blue Dot mission 2014 ARED exercise. Catalog upload is 2024, not mission date. Task boundaries, load and prescription have not been assigned.',
  'source_url':'https://dlmultimedia.esa.int/download/public/videos/2024/01/026/2401_026_BR_004.mp4',
  'source_sha256':'d7bec40ee47691bd04692e4d8d99984f8b386aa3ad06944d62442b8f8841d33c',
  'transformation':'Unmodified publisher MP4. Container lists 18737 frames; actual decode/native map has 18736 frames. Use supplied frame map; do not infer a missing annotation from container frame count.',
  'credit':'ESA/NASA'},
]

GUIDE='''THEIA — DOCTOR REVIEW PACKET
Project contact: Hlib Plakhtii / Theia
Prepared: 2026-11-14

START
1. Extract the entire ZIP; keep the videos folder beside LABEL_VIDEOS.html.
2. Open LABEL_VIDEOS.html in Chrome, Edge or Firefox. No installation, account,
   server or network connection is needed. If embedded playback is unavailable,
   play videos in VLC and enter the video-relative times in the form.
3. Work on video 01 first to establish your review criteria. Then write and lock
   those criteria before opening video 02. Do not consult model predictions.
4. Export your labels JSON before closing. Send that JSON back to Hlib.
   CSV export is also available. You do not need to return the videos.

WHAT IS INCLUDED
01: 109 seconds of ARED footage, used for development/rubric calibration.
02: 12 min 29 seconds of a different crew member's ARED recording, reserved from
    the current modeling workflow. Its actual exercise segments have not yet
    been identified. It may contain non-squat exercise or unusable views.
Both are real in-mission ESA/NASA media. No model skeletons, proposed event times,
confidence scores, classifications or performance results are included.
Total video duration is approximately 14 min 18 s. Detailed annotation will take
longer than playback. A partial review is useful; mark completion honestly.

PURPOSE
Create an independent expert reference for visible movement and whether a
crew-support technique review would be warranted. This is retrospective research,
not medical care for the people depicted. We do not ask you to diagnose a person
or infer pain, fatigue, injury, muscle mass or safe resistance from video.

FIRST: DEFINE WHAT CAN BE JUDGED
In the rubric box, write specific visible observations that would justify a
technique review, and conditions in which you cannot judge. State relevant
expertise and limits (for example, no ARED/microgravity-specific experience).
Earth-based assumptions about gravity, balance, depth and support may not apply.
The equipment, restraint, camera angle and individual prescription matter.
If this footage cannot support a defensible review decision, say so. That is a
valid result. Do not force a safe/unsafe decision.

RECOMMENDED LABELING ORDER FOR EACH VIDEO
A. TASK INTERVALS: cover the whole video, including non-exercise transitions,
   other exercise and uncertain periods. Mark squat only when identifiable.
B. VISIBILITY INTERVALS: for squat intervals, say whether the body parts needed
   by your rubric are assessable. Describe missing/occluded parts and camera cuts.
C. REVIEW INTERVALS: cover the same squat intervals with one of:
   - No concern visible: no visible reason for technique review under your rubric.
     This does NOT mean medically safe or appropriate load.
   - Technique review warranted: state the visible reason and relevant body region.
   - Cannot assess: view, task context or domain expertise is insufficient.
   - Outside scope: the interval does not meet the agreed assessment task.
   Include ordinary/no-concern and uncertain intervals, not just apparent errors.
D. REPETITIONS (optional supporting labels): mark a complete visible squat from
   descent onset to return to the individual's starting posture. Optionally record
   deepest point. Mark truncated/partial cycles separately; never guess through
   an occlusion. Do not label repositioning as a squat because the knees bend.

TIMING
All entered seconds are relative to the supplied review video, starting at zero.
Do NOT add 670 seconds to video 01. Source offsets are already in the manifest.
Mark start/end using the buttons or type seconds. Intervals are [start, end).
For a repetition, end is the return/completion time. For a review interval,
optionally record the earliest time its evidence justified review. Judge that
time from information available up to then; if that cannot be done, leave blank.
Browser seeking is approximate. Native-frame mappings are in provenance/.

INDEPENDENCE AND RETURN
Record whether you have already seen model outputs. Do not view the project's
results while labeling. Video 01 was used for development, so expert labels for
it remain development labels. Lock the rubric before reviewing video 02; any
later change needs a new rubric version and explicit disclosure to Hlib.
This packet alone does not designate a valid independent benchmark. Eligibility,
label coverage and agreement still need checking before an experiment.

Save frequently with EXPORT JSON. Browser storage is only a convenience and may
be unavailable. IMPORT JSON restores a saved review. Keep the two video IDs and
times unchanged. For another doctor, use a fresh packet and independent labels;
do not share the first doctor's answers before independent review.

MEDIA AND CREDITS
Credit: ESA/NASA. Source URLs, hashes, mission context, transformations and native
frame maps are supplied in provenance/. This is a collaborator-review copy, not
a public dataset release. Preserve original attribution; consult source terms
before wider publication or redistribution. This package has not been sent to
anyone automatically.
'''


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for clip in CLIPS:
        path=OUT/clip['path']
        clip['review_video_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        clip['bytes']=path.stat().st_size
    manifest={'package_id':PACKAGE,'created_at':datetime.now(timezone.utc).isoformat(),
              'purpose':'Independent clinician labeling; no model outputs included',
              'clips':CLIPS,'annotation_timebase':'Seconds relative to each supplied review video',
              'source_credit':'ESA/NASA','automatic_external_transmission':False}
    (OUT/'provenance/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'START_HERE.txt').write_text(GUIDE)
    blank={'package_id':PACKAGE,'schema_version':1,'reviewer':{'name_or_id':'','specialty':'','ared_or_microgravity_expertise':'','previously_seen_model_outputs':'not_answered'},
           'rubric':{'text':'','locked_at':None},'clip_completion':{},'annotations':[]}
    (OUT/'blank_labels.json').write_text(json.dumps(blank,indent=2)+'\n')
    fields=['clip_id','kind','start_s','end_s','task','visibility','action','confidence','bottom_s','earliest_evidence_s','visible_reason','notes']
    with (OUT/'blank_labels.csv').open('w') as f:csv.writer(f).writerow(fields)
    page=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Theia — independent video labeling</title><style>
*{box-sizing:border-box}body{font:16px/1.45 system-ui,sans-serif;color:#18313d;background:#f3f6f8;max-width:1180px;margin:24px auto;padding:0 18px}
h1,h2{line-height:1.2}section{background:white;border:1px solid #d7e1e6;border-radius:9px;padding:20px;margin:18px 0}
label{display:block;font-size:14px;font-weight:600;margin:8px 0}input,select,textarea,button{font:inherit}input,select,textarea{max-width:100%;width:100%;padding:8px;border:1px solid #9bb1bc;border-radius:4px;background:white;color:#18313d}
textarea{min-height:88px}button{padding:8px 12px;border:1px solid #436d7f;border-radius:5px;background:#edf4f7;color:#12465b;cursor:pointer;margin:4px 4px 4px 0}button.primary{background:#155d79;color:white}button:disabled{opacity:.5;cursor:not-allowed}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.note{padding:12px;background:#edf3f6;border-left:4px solid #7195a5}.warn{background:#fff1dc;border-color:#cc922f}.small{font-size:14px}.bar{position:sticky;top:0;background:#f3f6f8;z-index:1;border-bottom:1px solid #d7e1e6;padding:8px 0}
video{display:block;width:100%;max-height:650px;background:#111}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border-bottom:1px solid #d7e1e6;text-align:left;vertical-align:top}th{background:#eef4f6}.scroll{overflow:auto}input[type=checkbox]{width:auto}a{color:#146681}.hidden{display:none}#status{white-space:pre-line}details{margin:12px 0}
</style><h1>Theia — independent video labeling</h1>
<p>Project: Hlib Plakhtii / Theia · ESA/NASA source footage · <a href="START_HERE.txt" target="_blank">Read the labeling guide</a></p>
<p class="note">Please label what is visible and when a technique review would be warranted. “No concern visible” is not a medical safety judgment. No model predictions are provided.</p>
<div class="bar"><button class="primary" id="export">Export labels JSON</button><button id="csv">Export CSV</button><button id="import">Import saved JSON</button><input class="hidden" type="file" id="importFile" accept=".json"><span id="saved" class="small">Export before closing.</span></div>
<section><h2>1. Reviewer and assessment rule</h2><div class="grid">
<label>Name or reviewer ID (optional)<input id="reviewer"></label><label>Specialty / relevant expertise<input id="specialty"></label>
<label>ARED or microgravity experience<input id="expertise" placeholder="State experience or limits"></label>
<label>Already seen model predictions/results?<select id="exposure"><option value="not_answered">Please select</option><option value="no">No</option><option value="yes">Yes — describe in the rule/notes</option><option value="uncertain">Uncertain</option></select></label></div>
<label>Visible criteria that justify a review, and when you cannot assess<textarea id="rubric" placeholder="Develop this using video 01; lock before opening video 02. State task, visible observations, and limitations."></textarea></label>
<button id="lock">Lock rule before reserved video</button><span id="lockStatus" class="small"></span>
<p class="small">Locking preserves the rule for this review. If it needs revision later, export first and discuss a new rule version with Hlib; do not silently revise it after seeing the reserved recording.</p></section>
<section><h2>2. Watch and mark intervals</h2><label>Recording<select id="clip"></select></label><p id="role" class="note"></p><p id="context" class="small"></p>
<video id="video" controls preload="metadata"></video><p id="clock">Video time 0.00 s</p>
<div><button id="back">−0.04 s</button><button id="forward">+0.04 s</button><button id="play">Play / pause</button><label style="display:inline-block">Speed <select id="speed" style="width:100px"><option>0.25</option><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label></div>
<p class="small">Time-step buttons seek approximately one nominal frame. Use native-frame maps for later exact alignment. Label all task intervals first, then visibility/review intervals over eligible squats. Repetition labels are optional.</p>
<div class="grid"><label>Annotation kind<select id="kind"><option value="task_interval">Task interval</option><option value="visibility_interval">Visibility interval</option><option value="review_interval">Review decision interval</option><option value="repetition">Complete repetition</option><option value="partial_repetition">Partial / truncated repetition</option></select></label>
<label>Start seconds<input id="start" type="number" step="0.01" min="0" value="0"><button id="setStart">Set from video</button></label>
<label>End seconds<input id="end" type="number" step="0.01" min="0"><button id="setEnd">Set from video</button></label></div>
<div class="grid">
<label>Task<select id="task"><option value="">Choose when relevant</option><option value="squat">Squat</option><option value="other_exercise">Other exercise</option><option value="transition_or_nonexercise">Transition / non-exercise</option><option value="uncertain">Uncertain</option></select></label>
<label>Visibility under your rule<select id="visibility"><option value="">Choose when relevant</option><option value="adequate">Adequate to judge</option><option value="partial">Partial / limited</option><option value="unassessable">Cannot assess</option></select></label>
<label>Review decision<select id="action"><option value="">Choose for review interval</option><option value="no_concern_visible">No concern visible under rule</option><option value="technique_review_warranted">Technique review warranted</option><option value="cannot_assess">Cannot assess</option><option value="outside_scope">Outside assessment scope</option></select></label>
<label>Confidence<select id="confidence"><option value="">Please select</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></label>
<label>Deepest point, seconds (optional for repetition)<input id="bottom" type="number" step="0.01" min="0"></label>
<label>Earliest justified review time (optional)<input id="earliest" type="number" step="0.01" min="0"></label></div>
<label>Visible evidence, relevant body parts, or reason you cannot assess<textarea id="reason" placeholder="Required for a review or visibility interval. Describe observations; do not infer diagnosis, pain or fatigue."></textarea></label>
<label>Other notes<textarea id="notes"></textarea></label><button class="primary" id="add">Add annotation</button><button id="cancelEdit" class="hidden">Cancel edit</button>
<p id="status" role="status"></p><label><input id="completed" type="checkbox"> I have finished my intended review of this recording</label><p class="small">This is your completion statement, not automated validation of the labels. Export keeps partial/incomplete work.</p></section>
<section><h2>3. Review and export</h2><p id="counts"></p><div class="scroll"><table><thead><tr><th>Video</th><th>Kind</th><th>Start–end</th><th>Task / visibility / action</th><th>Evidence</th><th>Edit</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="note">Return the exported JSON to Hlib. Do not send browser cache files. Data stays on this computer until you choose to share the exported file. The packet has no network service.</p></section>
<script>
'use strict';
const PACKAGE=PACKAGE_JSON,clips=CLIPS_JSON,blank=BLANK_JSON;
const $=id=>document.getElementById(id),video=$('video');
let state=structuredClone(blank),editing=null,active=clips[0].id;
function clip(){return clips.find(c=>c.id===active)}
function capture(){state.reviewer={name_or_id:$('reviewer').value,specialty:$('specialty').value,ared_or_microgravity_expertise:$('expertise').value,previously_seen_model_outputs:$('exposure').value};if(!state.rubric.locked_at)state.rubric.text=$('rubric').value;state.updated_at=new Date().toISOString()}
function save(){capture();try{localStorage.setItem(PACKAGE,JSON.stringify(state));$('saved').textContent='Saved in this browser; export a backup.'}catch(e){$('saved').textContent='Browser storage unavailable: export JSON frequently.'}}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function render(){
 $('rows').innerHTML=state.annotations.map(a=>`<tr><td>${esc(a.clip_id)}</td><td>${esc(a.kind)}</td><td><button data-seek="${a.id}">${a.start_s.toFixed(2)}–${a.end_s.toFixed(2)}</button></td><td>${esc([a.task,a.visibility,a.action].filter(Boolean).join(' / '))}</td><td>${esc(a.visible_reason)}</td><td><button data-edit="${a.id}">Edit</button><button data-delete="${a.id}">Delete</button></td></tr>`).join('');
 $('counts').textContent=clips.map(c=>c.id+': '+state.annotations.filter(a=>a.clip_id===c.id).length+' annotations; '+(state.clip_completion[c.id]?'reviewer marked complete':'incomplete')).join(' · ');
 $('rubric').disabled=Boolean(state.rubric.locked_at);$('lock').disabled=Boolean(state.rubric.locked_at);$('lockStatus').textContent=state.rubric.locked_at?'Locked '+state.rubric.locked_at:'Not locked';
 $('completed').checked=Boolean(state.clip_completion[active]);
}
function loadClip(id){
 if(id==='ARED_RESERVE_02'&&!state.rubric.locked_at){alert('Please write and lock your visible assessment rule before opening the reserved recording.');$('clip').value=active;return false}
 active=id;$('clip').value=id;video.src=clip().path;video.load();$('role').textContent=clip().role;$('context').textContent=clip().source_context;
 $('start').value='0';$('end').value='';render();return true;
}
function hydrate(){
 $('reviewer').value=state.reviewer.name_or_id||'';$('specialty').value=state.reviewer.specialty||'';$('expertise').value=state.reviewer.ared_or_microgravity_expertise||'';$('exposure').value=state.reviewer.previously_seen_model_outputs||'not_answered';$('rubric').value=state.rubric.text||'';render();
}
function resetEdit(){editing=null;$('add').textContent='Add annotation';$('cancelEdit').classList.add('hidden')}
function download(name,type,text){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function coverageWarnings(){
 const warnings=[];
 for(const c of clips){
  const tasks=state.annotations.filter(a=>a.clip_id===c.id&&a.kind==='task_interval').sort((a,b)=>a.start_s-b.start_s);
  let cursor=0;for(const a of tasks){if(a.start_s>cursor+.08)warnings.push(c.id+': task coverage gap '+cursor.toFixed(2)+'–'+a.start_s.toFixed(2)+'s');if(a.start_s<cursor-.08)warnings.push(c.id+': overlapping task intervals');cursor=Math.max(cursor,a.end_s)}
  if(cursor<c.duration_s-.08)warnings.push(c.id+': task coverage ends at '+cursor.toFixed(2)+' of '+c.duration_s+'s');
  for(const kind of ['visibility_interval','review_interval']){
   const intervals=state.annotations.filter(a=>a.clip_id===c.id&&a.kind===kind).sort((a,b)=>a.start_s-b.start_s);
   for(let i=1;i<intervals.length;i++)if(intervals[i].start_s<intervals[i-1].end_s-.01)warnings.push(c.id+': overlapping '+kind+' entries');
   for(const task of tasks.filter(a=>a.task==='squat')){let at=task.start_s;for(const a of intervals){if(a.end_s<=at)continue;if(a.start_s>at+.08)break;at=Math.max(at,a.end_s)}if(at<task.end_s-.08)warnings.push(c.id+': '+kind+' does not cover squat '+task.start_s.toFixed(2)+'–'+task.end_s.toFixed(2)+'s')}
  }
 }
 return warnings;
}
function payload(){save();return {...state,exported_at:new Date().toISOString(),clip_manifest:clips,automated_coverage_warnings:coverageWarnings(),status:'expert_review_submission_not_yet_adjudicated'}}
$('export').onclick=()=>{const p=payload();download('theia_doctor_labels_'+new Date().toISOString().slice(0,10)+'.json','application/json',JSON.stringify(p,null,2));$('status').textContent=p.automated_coverage_warnings.length?'Exported with coverage warnings (partial work retained):\n'+p.automated_coverage_warnings.join('\n'):'Exported. No automated coverage gaps found; clinical validity is not automatically checked.'};
$('csv').onclick=()=>{save();const fields=['clip_id','kind','start_s','end_s','task','visibility','action','confidence','bottom_s','earliest_evidence_s','visible_reason','notes'];const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';download('theia_doctor_labels.csv','text/csv;charset=utf-8','\uFEFF'+[fields,...state.annotations.map(a=>fields.map(k=>a[k]))].map(row=>row.map(quote).join(',')).join('\r\n'));$('status').textContent='CSV exported. Also export JSON to retain your rubric and provenance.'};
$('import').onclick=()=>$('importFile').click();
$('importFile').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const p=JSON.parse(await file.text());if(p.package_id!==PACKAGE||p.schema_version!==1||!Array.isArray(p.annotations)||!p.reviewer||!p.rubric||!p.clip_completion)throw Error('Wrong or incomplete review format');
for(const a of p.annotations){const c=clips.find(c=>c.id===a.clip_id);if(!c||!Number.isFinite(a.start_s)||!Number.isFinite(a.end_s)||a.start_s<0||a.end_s<=a.start_s||a.end_s>c.duration_s+.001)throw Error('Invalid annotation time or video ID')}
if(state.annotations.length&&!confirm('Replace the current browser review with this saved file? Export current work first if needed.'))return;state=p;resetEdit();hydrate();save();loadClip(clips[0].id);$('status').textContent='Saved review imported.'}catch(err){alert('Import failed: '+err.message)}e.target.value=''};
$('lock').onclick=()=>{capture();if(!state.rubric.text.trim()){alert('Write your assessment rule and limitations first.');return}if(!confirm('Lock this rule before reserved-video review? Export first if you need to preserve another version.'))return;state.rubric.locked_at=new Date().toISOString();save();render()};
$('clip').innerHTML=clips.map(c=>`<option value="${c.id}">${esc(c.title)}</option>`).join('');$('clip').onchange=e=>{resetEdit();loadClip(e.target.value)};
video.ontimeupdate=()=>{$('clock').textContent='Video time '+Math.min(video.currentTime,clip().duration_s).toFixed(2)+' s / '+clip().duration_s.toFixed(2)+' s'};
video.onplay=()=>{state.first_played_at=state.first_played_at||{};state.first_played_at[active]=state.first_played_at[active]||new Date().toISOString();save()};
video.onerror=()=>{$('status').textContent='Video could not be loaded. Extract the whole ZIP and keep the videos folder beside this file. You can also use VLC and enter video-relative times manually.'};
$('back').onclick=()=>{video.pause();video.currentTime=Math.max(0,video.currentTime-.04)};$('forward').onclick=()=>{video.pause();video.currentTime=Math.min(clip().duration_s-.04,video.currentTime+.04)};
$('play').onclick=()=>{video.paused?video.play().catch(e=>$('status').textContent=e.message):video.pause()};$('speed').onchange=e=>video.playbackRate=Number(e.target.value);
$('setStart').onclick=()=>$('start').value=Math.min(video.currentTime,clip().duration_s).toFixed(2);$('setEnd').onclick=()=>$('end').value=Math.min(video.currentTime,clip().duration_s).toFixed(2);
$('add').onclick=()=>{
 const s=$('start').value.trim(),e=$('end').value.trim(),kind=$('kind').value;let start=Number(s),end=Number(e);
 if(!s||!e||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>clip().duration_s+.001){$('status').textContent='Enter valid start < end within this video.';return}
 if(kind==='task_interval'&&!$('task').value){$('status').textContent='Select a task.';return}
 if(kind==='visibility_interval'&&!$('visibility').value){$('status').textContent='Select visibility.';return}
 if(kind==='review_interval'&&!$('action').value){$('status').textContent='Select a review decision.';return}
 if(!$('confidence').value){$('status').textContent='Select confidence.';return}
 if(['review_interval','visibility_interval'].includes(kind)&&!$('reason').value.trim()){$('status').textContent='Describe visible evidence or why you cannot assess.';return}
 const bottom=$('bottom').value===''?null:Number($('bottom').value),earliest=$('earliest').value===''?null:Number($('earliest').value);
 if([bottom,earliest].some(t=>t!==null&&(!Number.isFinite(t)||t<start||t>end))){$('status').textContent='Optional event times must lie inside the interval.';return}
 const a={id:editing||('annotation_'+Date.now()+'_'+Math.random().toString(36).slice(2,7)),clip_id:active,kind,start_s:start,end_s:end,task:$('task').value,visibility:$('visibility').value,action:$('action').value,confidence:$('confidence').value,bottom_s:bottom,earliest_evidence_s:earliest,visible_reason:$('reason').value.trim(),notes:$('notes').value.trim(),recorded_at:new Date().toISOString()};
 if(editing)state.annotations=state.annotations.map(r=>r.id===editing?a:r);else state.annotations.push(a);state.clip_completion[active]=false;resetEdit();save();render();$('status').textContent='Saved annotation. Export JSON periodically.';
};
$('rows').onclick=e=>{const b=e.target.closest('button');if(!b)return;const id=b.dataset.edit||b.dataset.delete||b.dataset.seek,a=state.annotations.find(a=>a.id===id);if(!a)return;
 if(b.dataset.delete){if(confirm('Delete this annotation?')){state.annotations=state.annotations.filter(a=>a.id!==id);state.clip_completion[a.clip_id]=false;save();render()}return}
 if(active!==a.clip_id&&!loadClip(a.clip_id))return;
 const seek=()=>video.currentTime=a.start_s;if(video.readyState>=1)seek();else video.addEventListener('loadedmetadata',seek,{once:true});
 if(b.dataset.edit){editing=id;for(const [field,key] of [['kind','kind'],['start','start_s'],['end','end_s'],['task','task'],['visibility','visibility'],['action','action'],['confidence','confidence'],['bottom','bottom_s'],['earliest','earliest_evidence_s'],['reason','visible_reason'],['notes','notes']])$(field).value=a[key]??'';$('add').textContent='Update annotation';$('cancelEdit').classList.remove('hidden');$('kind').scrollIntoView({behavior:'smooth',block:'center'})}
};
$('cancelEdit').onclick=resetEdit;$('completed').onchange=()=>{state.clip_completion[active]=$('completed').checked;save();render()};
for(const id of ['reviewer','specialty','expertise','exposure','rubric'])$(id).addEventListener('change',save);
try{const raw=localStorage.getItem(PACKAGE);if(raw){const cached=JSON.parse(raw);if(cached.package_id===PACKAGE&&Array.isArray(cached.annotations))state=cached}}catch(e){}
hydrate();loadClip(clips[0].id);
</script></html>'''
    page=page.replace('PACKAGE_JSON',json.dumps(PACKAGE)).replace('CLIPS_JSON',json.dumps(CLIPS)).replace('BLANK_JSON',json.dumps(blank))
    (OUT/'LABEL_VIDEOS.html').write_text(page)
    print(OUT)


if __name__=='__main__':main()
