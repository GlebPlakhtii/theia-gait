"""Package short pose-prelabelled clips; the doctor's only task is form review."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from build_doctor_clips import ROOT,OUT,digest,write


def main():
    clips=json.load(open(OUT/'automatic_labels/clips.json'))
    # No absolute workspace paths, publisher good/bad labels or model form verdicts.
    public=[{k:v for k,v in c.items() if k not in ['source_subset_first_index','source_subset_end_index_exclusive']} for c in clips]
    write(OUT/'automatic_labels/clips.json',public)
    fields=['clip_id','automatic_type','cycle_duration_s','descent_s','ascent_s',
            'left_projected_knee_min_deg','left_projected_knee_max_deg','left_projected_knee_excursion_deg',
            'right_projected_knee_min_deg','right_projected_knee_max_deg','right_projected_knee_excursion_deg',
            'median_projected_left_right_difference_deg','doctor_conclusion','doctor_notes']
    with (OUT/'DOCTOR_CONCLUSIONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in public:
            r={**c,**c['automatic_measurements']};w.writerow({k:r.get(k,'') for k in fields})
    guide='''THEIA — SHORT CLIPS FOR DOCTOR FORM REVIEW
Prepared for Hlib Plakhtii / Theia, 2026-11-14

YOUR TASK: watch each short clip and give a form conclusion.
The videos have already been cut. Repetition boundaries, movement phases,
timing and projected knee angles have already been estimated automatically.
You do NOT need to label timestamps, count repetitions or draw a skeleton.

HOW TO USE
1. Extract the whole ZIP.
2. Open REVIEW_CLIPS.html in Chrome, Edge or Firefox. No installation needed.
3. Start with the ARED group. Watch the short original/pose comparison video.
   You can switch to the original-only version using the view dropdown.
4. Choose one conclusion:
   - Form acceptable in the visible segment
   - Form needs review
   - Cannot judge from this view/context
   - Not a squat: setup, transition or other movement
   Add your observations, concerns, acceptable adaptations or recommendations
   for further review in the notes box. The timing is already provided.
5. Click SAVE & NEXT. Export JSON regularly and at the end. Return that JSON
   to Hlib. Partial reviews are welcome; unreviewed items remain blank.

PRIORITY AND CONTENTS
Priority 1: 32 short in-mission ARED clips: 23 pose-detected repetition candidates
and nine context clips covering the remaining movement.
Optional supporting review: 80 gym clips: 69 repetition candidates and 11 context
clips. Their health status is not documented. They are not a clinical cohort.
There are 112 review items total; every item has two video versions. Review an
item once, using whichever version is useful. No need to review both separately.
All 5,068 eligible source frames are represented. Context clips are included to
allow rejection of transitions and discovery of missed repetitions. Adjacent
clips contain a small amount of overlapping context; they are not independent
people or sessions. No 12-minute unprocessed recording is included.

WHAT THE AUTOMATIC LABELS MEAN
The repetition label is a candidate, not verified truth. Reject it if the clip
shows repositioning, a partial movement, a missed boundary or another exercise.
The pose video shows source RGB on the left and predicted joints on the right.
The added banner identifies the clip and predicted descent/ascent phase.
Measurements are 2D projected hip-knee-ankle angles. Camera viewpoint and hidden
joints can change them; they are not calibrated anatomical flexion or loading.
Left/right differences are also viewpoint-dependent, not a diagnosis of asymmetry.
Automatic good/bad form, injury, pain, fatigue and safe-load labels are NOT supplied.
Your form conclusion and notes are the missing expert judgment.

ASSESSMENT CONTEXT
ARED footage is in microgravity with equipment/restraint support; normal Earth
balance and gravity assumptions may not apply. Individual prescriptions and
resistance are unknown. The ESA catalog associates this recording with EasyMotion-2;
stimulation status is unknown. It is acceptable to say that you cannot judge.
“Form acceptable” refers only to what is visible; it does not certify medical
safety or the appropriate resistance. Please note any expertise limits.
These clips and their model estimates are development evidence. Seeing pose
outputs means this is an assisted expert review, not a pose-blinded benchmark.

ALTERNATIVES AND RETURN
If the browser form does not work, play files in clips_original/ or clips_pose/
and enter conclusions into DOCTOR_CONCLUSIONS.csv. Automatic measurements and
clip IDs are already filled in. Return the filled CSV, preserving IDs.
JSON export includes all automatic labels plus your conclusions and overall notes.
Import saved JSON resumes a review. Browser autosave is a convenience; always
export a file before closing. The packet works offline and sends nothing itself.

CREDITS
ARED source footage: ESA/NASA. Gym footage: original multiview fitness dataset,
preserving publisher privacy masks. Clips are silent derived research-review
copies. Exact source-frame intervals and hashes are retained. This packet is
prepared for collaborator review, not a public dataset release. Preserve source
attribution; source reuse terms still apply to wider redistribution.
'''
    (OUT/'START_HERE.txt').write_text(guide)
    data={'package_id':'theia-doctor-short-clips-20261114-v2','schema_version':2,'reviewer_name_or_id':'',
          'specialty_and_limits':'','overall_conclusions':'','reviews':{}}
    write(OUT/'automatic_labels/blank_doctor_review.json',data)
    page=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Theia — short clips for form review</title><style>
*{box-sizing:border-box}body{font:16px/1.45 system-ui,sans-serif;color:#18313d;background:#f1f5f7;max-width:1180px;margin:24px auto;padding:0 18px}
h1,h2{line-height:1.2}section{padding:20px;background:white;border:1px solid #d4e1e7;border-radius:9px;margin:18px 0}button,input,select,textarea{font:inherit}button{border:1px solid #527e91;border-radius:5px;padding:9px 13px;color:#174a61;background:#edf5f8;cursor:pointer;margin:4px 5px 4px 0}button.primary{background:#155d79;color:white}input,select,textarea{width:100%;padding:8px;border:1px solid #9cb2bd;border-radius:4px;background:white}label{display:block;font-size:14px;font-weight:600;margin:8px 0}textarea{min-height:86px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.note{padding:12px;border-left:4px solid #6b96a9;background:#edf4f7}.small{font-size:14px}.toolbar{position:sticky;top:0;background:#f1f5f7;z-index:2;padding:8px 0;border-bottom:1px solid #d5e1e6}video{display:block;width:100%;max-height:650px;background:#14252c}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px}.metric{padding:10px;background:#f0f5f7;border-radius:5px}.metric strong{display:block;font-size:19px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #d8e3e8}a{color:#13647e}.hidden{display:none}.scroll{overflow:auto}#status{white-space:pre-line}
.conclusion-group{border:0;padding:0;margin:18px 0}.conclusion-group legend{font-size:16px;font-weight:650;margin-bottom:8px}.conclusion-choices{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.conclusion-choice{display:flex;align-items:center;gap:10px;margin:0;padding:14px;border:2px solid #c6d7df;border-radius:7px;cursor:pointer;min-height:60px}.conclusion-choice input{width:auto;flex:none;accent-color:#155d79}.conclusion-choice:has(input:checked){background:#e6f3f8;border-color:#155d79}.conclusion-choice:focus-within{outline:2px solid #155d79;outline-offset:2px}@media(max-width:600px){.conclusion-choices{grid-template-columns:1fr}}
</style><h1>Theia — watch the clip, judge the form</h1>
<p>The cuts and pose measurements are ready. Your task is the form conclusion and notes.</p>
<div class="toolbar"><button id="export" class="primary">Export doctor conclusions</button><button id="import">Resume from saved JSON</button><input id="importFile" class="hidden" type="file" accept=".json"><span id="progress" class="small"></span></div>
<details><summary>Reviewer details and instructions</summary><div class="grid"><label>Name or reviewer ID (optional)<input id="reviewer"></label><label>Specialty / relevant limits (optional)<input id="specialty"></label></div><p><a href="START_HERE.txt" target="_blank">Short guide</a> · <a href="DOCTOR_CONCLUSIONS.csv">Prefilled CSV alternative</a></p><p class="small">ARED is in microgravity; load and prescription are unknown. “Acceptable” concerns the visible form only. These are development clips with model assistance; this is not a blinded validation study.</p></details>
<section><div class="grid"><label>Review group<select id="group"><option value="ARED">Priority 1: ARED — 32 short clips</option><option value="GYM">Optional: gym — 80 short clips</option><option value="ALL">All 112 clips</option></select></label><label>Clip<select id="clip"></select></label><label>Video view<select id="view"><option value="pose_video">Original + estimated pose</option><option value="original_video">Original only</option></select></label></div>
<h2 id="title"></h2><p id="autoLabel" class="note"></p><video id="video" controls preload="metadata" loop></video>
<p id="phase" class="small">Predicted phase</p><div><button id="previous">Previous clip</button><button id="next">Next clip</button><label style="display:inline-block">Speed<select id="speed" style="width:100px"><option>0.25</option><option>0.5</option><option selected>1</option></select></label></div>
<details open><summary>Automatic measurements — already filled in</summary><div id="metrics" class="metrics"></div><p class="small">Projected 2D angles depend on viewpoint and occlusion. These measurements do not determine good/bad form or medical safety. Boundaries and phases may be wrong.</p></details>
<fieldset class="conclusion-group"><legend>Your form conclusion</legend><input type="hidden" id="conclusion" value=""><div class="conclusion-choices">
<label class="conclusion-choice"><input type="radio" name="formConclusion" value="acceptable_visible_form"><span>Form acceptable in this visible segment</span></label>
<label class="conclusion-choice"><input type="radio" name="formConclusion" value="form_needs_review"><span>Form needs review</span></label>
<label class="conclusion-choice"><input type="radio" name="formConclusion" value="cannot_judge"><span>Cannot judge from this view/context</span></label>
<label class="conclusion-choice"><input type="radio" name="formConclusion" value="not_a_squat"><span>Not a squat / setup or transition</span></label>
</div><button type="button" id="clearConclusion">Clear selection</button></fieldset>
<label>Your observations / conclusions<textarea id="notes" placeholder="What is good, what needs attention, acceptable adaptations, or why you cannot judge. No timestamps needed."></textarea></label>
<button id="saveNext" class="primary">Save & next clip</button><button id="save">Save this conclusion</button><p id="status" role="status"></p></section>
<section><label>Overall conclusions (optional)<textarea id="overall" placeholder="General findings, limitations and recommendations for further review."></textarea></label><details><summary>Review progress / jump to any clip</summary><div class="scroll"><table><thead><tr><th>Clip</th><th>Automatic label</th><th>Doctor conclusion</th></tr></thead><tbody id="summary"></tbody></table></div></details></section>
<p class="small">Works offline. Export the JSON and return it to Hlib Plakhtii / Theia. Nothing is sent automatically. ARED footage: ESA/NASA.</p>
<script>
'use strict';const clips=CLIPS_DATA,blank=BLANK_DATA,$=id=>document.getElementById(id),video=$('video');
let state=structuredClone(blank),active=clips[0].clip_id;
const current=()=>clips.find(c=>c.clip_id===active);
const filtered=()=>clips.filter(c=>$('group').value==='ALL'||($('group').value==='ARED'?c.source_id==='ARED':c.source_id!=='ARED'));
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=(v,suffix='')=>v===null||v===undefined?'—':Number(v).toFixed(1)+suffix;
function saveDraft(){state.reviewer_name_or_id=$('reviewer').value;state.specialty_and_limits=$('specialty').value;state.overall_conclusions=$('overall').value;state.updated_at=new Date().toISOString();try{localStorage.setItem(blank.package_id,JSON.stringify(state))}catch(e){$('status').textContent='Browser storage is unavailable. Export JSON to save your work.'}}
function setConclusion(value){$('conclusion').value=value;document.querySelectorAll('input[name="formConclusion"]').forEach(input=>input.checked=input.value===value)}
function capture(){state.reviews[active]={conclusion:$('conclusion').value||null,notes:$('notes').value,recorded_at:new Date().toISOString()};saveDraft()}
function progress(){const reviewed=clips.filter(c=>state.reviews[c.clip_id]?.conclusion).length;$('progress').textContent=reviewed+' / '+clips.length+' reviewed';$('summary').innerHTML=clips.map(c=>`<tr><td><button data-clip="${c.clip_id}">${c.clip_id}</button></td><td>${c.automatic_type==='candidate_repetition'?'Pose repetition candidate':'Context / unsegmented'}</td><td>${esc(state.reviews[c.clip_id]?.conclusion||'Not reviewed')}</td></tr>`).join('')}
function options(){const list=filtered();$('clip').innerHTML=list.map(c=>`<option value="${c.clip_id}">${c.clip_id} · ${c.automatic_type==='candidate_repetition'?'repetition candidate':'context'} · ${c.video_duration_s.toFixed(1)}s</option>`).join('');if(!list.some(c=>c.clip_id===active))active=list[0].clip_id;$('clip').value=active}
function load(id){active=id;const c=current();$('clip').value=id;video.src=c[$('view').value];video.load();video.playbackRate=Number($('speed').value);$('title').textContent=c.clip_id+' · '+c.video_duration_s.toFixed(1)+' seconds';$('autoLabel').textContent=c.automatic_type==='candidate_repetition'?'Automatic label: candidate squat repetition. Start, bottom, completion and phases are already estimated. Please judge the form and reject the label if needed.':'Automatic label: unsegmented context. This may be setup, transition, another exercise or a missed repetition. Please identify it in your conclusion.';
 const m=c.automatic_measurements;const cells=[['Cycle duration',number(c.cycle_duration_s,' s')],['Descent / ascent',number(c.descent_s,' s')+' / '+number(c.ascent_s,' s')],['Left knee projected range',number(m.left_projected_knee_min_deg,'°')+'–'+number(m.left_projected_knee_max_deg,'°')],['Right knee projected range',number(m.right_projected_knee_min_deg,'°')+'–'+number(m.right_projected_knee_max_deg,'°')],['Left / right excursion',number(m.left_projected_knee_excursion_deg,'°')+' / '+number(m.right_projected_knee_excursion_deg,'°')],['Median left/right difference',number(m.median_projected_left_right_difference_deg,'°')]];
 $('metrics').innerHTML=cells.map(([label,value])=>`<div class="metric">${label}<strong>${value}</strong></div>`).join('');setConclusion(state.reviews[id]?.conclusion||'');$('notes').value=state.reviews[id]?.notes||'';$('phase').textContent='Pose labels ready; play to inspect phases.';$('status').textContent='';progress();}
function advance(delta){const list=filtered(),i=list.findIndex(c=>c.clip_id===active),next=i+delta;if(next<0||next>=list.length){$('status').textContent='End of this group. Export your conclusions or select another group.';return}load(list[next].clip_id);if(delta>0){const requestedClip=active;video.play().catch(error=>{if(active===requestedClip&&error.name!=='AbortError')$('status').textContent='Playback did not start. Press Play to continue.'})}}
function exportLabels(){capture();const payload={...state,exported_at:new Date().toISOString(),review_mode:'expert_form_review_assisted_by_pose; not blinded',clips:clips.map(c=>({...c,doctor_conclusion:state.reviews[c.clip_id]?.conclusion||null,doctor_notes:state.reviews[c.clip_id]?.notes||''}))};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='theia_doctor_conclusions_'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('status').textContent='Exported. Return the JSON file to Hlib. Unreviewed clips remain blank.'}
$('group').onchange=()=>{capture();options();load(active)};$('clip').onchange=e=>{capture();load(e.target.value)};$('view').onchange=()=>{const t=video.currentTime;capture();load(active);video.addEventListener('loadedmetadata',()=>video.currentTime=Math.min(t,current().video_duration_s-.04),{once:true})};
$('previous').onclick=()=>{capture();advance(-1)};$('next').onclick=()=>{capture();advance(1)};
$('save').onclick=()=>{capture();progress();$('status').textContent='Saved. Export JSON to keep a backup.'};$('saveNext').onclick=()=>{if(!$('conclusion').value){$('status').textContent='Choose a conclusion, or use Next clip to leave it unreviewed.';return}capture();progress();advance(1)};
$('speed').onchange=e=>video.playbackRate=Number(e.target.value);video.ontimeupdate=()=>{const c=current(),t=video.currentTime;let phase='Context';if(c.candidate_bottom_s!==null&&t>=c.core_start_s&&t<=c.core_end_s)phase=t<c.candidate_bottom_s?'Predicted descent':'Predicted ascent';$('phase').textContent='Clip '+t.toFixed(2)+' s · '+phase};video.onerror=()=>$('status').textContent='Extract the whole ZIP and keep both clips folders beside this HTML. You can also use the CSV and a video player.';
$('export').onclick=exportLabels;$('import').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;const p=JSON.parse(await f.text());if(p.package_id!==blank.package_id||p.schema_version!==2||!p.reviews)throw Error('Wrong review package');state=p;$('reviewer').value=state.reviewer_name_or_id||'';$('specialty').value=state.specialty_and_limits||'';$('overall').value=state.overall_conclusions||'';saveDraft();load(active);$('status').textContent='Saved conclusions imported.'}catch(err){$('status').textContent='Import failed: '+err.message}e.target.value=''};
$('summary').onclick=e=>{const b=e.target.closest('[data-clip]');if(!b)return;capture();$('group').value='ALL';options();load(b.dataset.clip);$('title').scrollIntoView({behavior:'smooth',block:'start'})};
for(const id of ['reviewer','specialty','overall'])$(id).onchange=saveDraft;$('notes').onchange=()=>{capture();progress()};
document.querySelectorAll('input[name="formConclusion"]').forEach(input=>input.onchange=()=>{setConclusion(input.value);capture();progress()});
$('clearConclusion').onclick=()=>{setConclusion('');capture();progress()};
try{const p=JSON.parse(localStorage.getItem(blank.package_id)||'null');if(p?.package_id===blank.package_id&&p.reviews)state=p}catch(e){}
$('reviewer').value=state.reviewer_name_or_id||'';$('specialty').value=state.specialty_and_limits||'';$('overall').value=state.overall_conclusions||'';options();load(active);
</script></html>'''
    page=page.replace('CLIPS_DATA',json.dumps(public)).replace('BLANK_DATA',json.dumps(data))
    (OUT/'REVIEW_CLIPS.html').write_text(page)
    print(OUT/'REVIEW_CLIPS.html')


if __name__=='__main__':main()
