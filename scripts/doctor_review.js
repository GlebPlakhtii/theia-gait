'use strict';
const $ = id => document.getElementById(id);
const KEY = 'theia-unified-doctor-review-v1';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let catalog=window.THEIA_CATALOG, clips=catalog.clips, active=null, pose=null, poseRequest=0;
let mediaRequest=0, metadataHandler=null, pendingPlayback=null, frameHandle=null, frameGeneration=0;
let state={package_id:KEY,schema_version:3,reviews:{},conflicts:[],unmapped_reviews:[],imported_reviewer_details:[],reviewer_name_or_id:'',specialty_and_limits:'',overall_conclusions:''};
const labels={acceptable_visible_form:'Acceptable visible form',form_needs_review:'Needs review',cannot_judge:'Cannot judge',not_expected_exercise:'Other movement / transition',not_a_squat:'Other movement / transition'};
const video=$('video'), canvas=$('overlay'), ctx=canvas.getContext('2d');
const radio=()=>document.querySelector('input[name="conclusion"]:checked')?.value||null;
const current=()=>clips.find(c=>c.uid===active);
const meaningful=r=>!!(r&&(r.conclusion||r.notes));
function message(text){$('status').textContent=text}
function persist(){
 state.updated_at=new Date().toISOString();
 try{localStorage.setItem(KEY,JSON.stringify(state));$('storageWarning').textContent=''}
 catch(e){$('storageWarning').textContent='Browser storage is unavailable or full. Export JSON now to preserve your conclusions.'}
}
function capture(){
 if(active){const old=state.reviews[active]||{},next={conclusion:radio(),notes:$('notes').value};
  if((old.conclusion||null)!==next.conclusion||(old.notes||'')!==next.notes)state.reviews[active]={...next,recorded_at:new Date().toISOString()};}
 for(const [field,id] of [['reviewer_name_or_id','reviewer'],['specialty_and_limits','specialty'],['overall_conclusions','overall']])state[field]=$(id).value;
 persist();
}
function filtered(){const term=$('search').value.toLowerCase().trim();return clips.filter(c=>{
 const r=state.reviews[c.uid],status=$('reviewStatus').value;
 return ($('group').value==='all'||c.domain===$('group').value)&&($('source').value==='all'||c.source_id===$('source').value)&&
 (status==='all'||(status==='unreviewed'&&!r?.conclusion)||(status==='reviewed'&&r?.conclusion)||r?.conclusion===status)&&
 (!term||[c.clip_id,c.source_id,c.subject_id,c.domain,c.view].join(' ').toLowerCase().includes(term));});}
function groups(){
 const oldGroup=$('group').value,oldSource=$('source').value;
 $('group').innerHTML='<option value="all">All exercise groups</option>'+Object.entries(catalog.groups).filter(([k])=>clips.some(c=>c.domain===k)).map(([k,v])=>`<option value="${esc(k)}">${esc(v)} (${clips.filter(c=>c.domain===k).length})</option>`).join('');
 if([...$('group').options].some(o=>o.value===oldGroup))$('group').value=oldGroup;
 const sources=[...new Set(clips.filter(c=>$('group').value==='all'||c.domain===$('group').value).map(c=>c.source_id))];
 $('source').innerHTML='<option value="all">All sources</option>'+sources.map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('');
 if(sources.includes(oldSource))$('source').value=oldSource;
}
function progress(){
 const count=clips.filter(c=>state.reviews[c.uid]?.conclusion).length;
 $('progress').textContent=`${count} / ${clips.length} reviewed`;$('bar').max=clips.length||1;$('bar').value=count;
 $('summary').innerHTML=clips.map(c=>`<tr><td><button data-uid="${esc(c.uid)}">${esc(c.clip_id)}</button></td><td>${esc(catalog.groups[c.domain]||c.domain)}</td><td>${c.video_duration_s.toFixed(1)}s</td><td>${esc(labels[state.reviews[c.uid]?.conclusion]||'Not reviewed')}</td></tr>`).join('');
 $('pipeline').textContent=catalog.status_text;
}
function options(){
 const list=filtered();$('clipSelect').innerHTML=list.map(c=>`<option value="${esc(c.uid)}">${esc(c.clip_id)} · ${c.video_duration_s.toFixed(1)}s${state.reviews[c.uid]?.conclusion?' · reviewed':''}</option>`).join('');
 $('workspace').classList.toggle('hidden',!list.length);$('empty').classList.toggle('hidden',!!list.length);
 return list;
}
const number=(n,suffix='')=>n===null||n===undefined?'—':Number(n).toFixed(1)+suffix;
function load(uid,autoplay=false,restoreTime=0){
 const list=options();const c=clips.find(c=>c.uid===uid&&list.some(x=>x.uid===uid))||list[0];
 if(!c){active=null;pose=null;++poseRequest;++mediaRequest;stopFrames();video.pause();video.removeAttribute('src');video.load();draw();progress();return}
 active=c.uid;$('clipSelect').value=active;$('position').textContent=`(${list.findIndex(x=>x.uid===active)+1} of ${list.length})`;
 $('title').textContent=c.clip_id;
 $('badges').innerHTML=[catalog.groups[c.domain]||c.domain,c.view,c.subject_id?'Subject '+c.subject_id:null,c.video_duration_s.toFixed(1)+' seconds'].filter(Boolean).map(x=>`<span class="pill">${esc(x)}</span>`).join('');
 const type=c.automatic_type==='candidate_repetition'?'Pose-detected knee-flexion cycle candidate':c.automatic_type==='gait_window'?'Walking window · boundaries are time-based':'Context / unsegmented movement';
 $('context').textContent=type+(c.context?' — '+c.context:'');
 const m=c.automatic_measurements||{};
 $('metrics').innerHTML=[['Cycle duration',number(c.cycle_duration_s,' s')],['Descent / ascent',number(c.descent_s,' s')+' / '+number(c.ascent_s,' s')],['Left projected knee range',number(m.left_projected_knee_min_deg,'°')+'–'+number(m.left_projected_knee_max_deg,'°')],['Right projected knee range',number(m.right_projected_knee_min_deg,'°')+'–'+number(m.right_projected_knee_max_deg,'°')]].map(([k,v])=>`<div class="metric">${k}<strong>${v}</strong></div>`).join('');
 $('poseProvenance').textContent=c.pose_provenance||'Sapiens2-5B predictions; automatic timing and angles are unverified candidates.';
 $('otherMovement').textContent=c.exercise==='walking'?'Not walking / setup or transition':'Not the expected exercise / setup or transition';
 document.querySelectorAll('input[name="conclusion"]').forEach(r=>r.checked=r.value===state.reviews[active]?.conclusion);
 $('notes').value=state.reviews[active]?.notes||'';message('');setVideo(autoplay,false,restoreTime);progress();
}
function setVideo(autoplay=false,preserveTime=false,restoreTime=0){
 const c=current();if(!c)return;
 const time=preserveTime?(metadataHandler&&pendingPlayback?pendingPlayback.time:video.currentTime):restoreTime;
 if(preserveTime&&metadataHandler&&pendingPlayback)autoplay=pendingPlayback.autoplay;
 pose=null;const request=++poseRequest;
 const showPose=$('view').value==='pose';
 const src=showPose&&c.pose_video?c.pose_video:c.original_video;
 const useOverlay=showPose&&!c.pose_video&&!!c.pose_data;
 canvas.classList.toggle('hidden',!useOverlay);draw();
 if(useOverlay){const script=document.createElement('script');script.src=c.pose_data;script.onload=()=>{
  if(request===poseRequest&&window.THEIA_POSE?.clip_id===c.clip_id){pose=window.THEIA_POSE;draw()}script.remove();};script.onerror=()=>{if(request===poseRequest)message('Pose overlay could not load. The original video is still available.');script.remove()};document.head.appendChild(script);}
 if(showPose&&!c.pose_video&&!c.pose_data)message('Pose is unavailable for this clip. Review the original; details are under Pose and automatic measurements.');
 // Canvas pose toggles use the same RGB video: leave its playback and pending load intact.
 if(preserveTime&&video.getAttribute('src')===src){draw();startFrames();return}
 const media=++mediaRequest;stopFrames();
 if(metadataHandler)video.removeEventListener('loadedmetadata',metadataHandler);
 pendingPlayback={time,autoplay};
 metadataHandler=()=>{
  if(media!==mediaRequest)return;
  metadataHandler=null;pendingPlayback=null;
  video.currentTime=Math.min(time,Math.max(0,video.duration-.01));
  video.playbackRate=Number($('speed').value);draw();startFrames();
  if(autoplay)video.play().catch(error=>{if(media===mediaRequest&&error.name!=='AbortError')message('Press Play to start this clip.')});
 };
 // Subscribe before load(), including when the file is already cached.
 video.addEventListener('loadedmetadata',metadataHandler,{once:true});
 video.src=src;video.load();video.playbackRate=Number($('speed').value);
}
function draw(mediaTime=video.currentTime){
 const w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}ctx.clearRect(0,0,w,h);
 const c=current();if(!c)return;
 let phase='Context';if(c.automatic_type==='gait_window')phase='Walking window';else if(c.candidate_bottom_s!=null&&mediaTime>=c.core_start_s&&mediaTime<=c.core_end_s)phase=mediaTime<c.candidate_bottom_s?'Predicted descent':'Predicted ascent';
 $('phase').textContent=`Clip ${mediaTime.toFixed(2)} s · ${phase} · source ${(c.source_start_s+mediaTime).toFixed(2)} s`;
 if(!pose||$('view').value!=='pose')return;
 const row=pose.frames[Math.floor(mediaTime*pose.fps+1e-4)];if(!row)return;
 const scale=Math.min(w/pose.width,h/pose.height),ox=(w-pose.width*scale)/2,oy=(h-pose.height*scale)/2;
 const point=i=>[ox+row[3*i]*scale,oy+row[3*i+1]*scale];
 const valid=i=>row[3*i+2]>=.3&&row[3*i]>=0&&row[3*i]<pose.width&&row[3*i+1]>=0&&row[3*i+1]<pose.height;
 ctx.lineWidth=2.5;
 for(const [a,b] of [[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,13],[13,15],[12,14],[14,16],[15,17],[15,19],[16,20],[16,22],[11,12],[5,6]])if(valid(a)&&valid(b)){ctx.strokeStyle=a%2?'#63fb98':'#ffc376';ctx.beginPath();ctx.moveTo(...point(a));ctx.lineTo(...point(b));ctx.stroke()}
 for(let i=0;i<23;i++)if(valid(i)){ctx.fillStyle='#fff36e';ctx.beginPath();ctx.arc(...point(i),2.8,0,Math.PI*2);ctx.fill()}
}
function stopFrames(){
 ++frameGeneration;
 if(frameHandle!==null&&video.cancelVideoFrameCallback)video.cancelVideoFrameCallback(frameHandle);
 frameHandle=null;
}
function startFrames(){
 if(!video.requestVideoFrameCallback||frameHandle!==null)return;
 const generation=frameGeneration;
 frameHandle=video.requestVideoFrameCallback((_,meta)=>{
  if(generation!==frameGeneration)return;
  frameHandle=null;draw(meta.mediaTime);startFrames();
 });
}
// A source replacement can discard pending frame callbacks. Re-arm after loads/seeks,
// and keep media events as a fallback so a missed callback cannot freeze the pose.
video.addEventListener('emptied',()=>{stopFrames();draw()});
for(const event of ['loadeddata','playing','seeked'])video.addEventListener(event,()=>{stopFrames();draw();startFrames()});
for(const event of ['timeupdate','seeking','pause','ended','ratechange'])video.addEventListener(event,()=>draw());
window.addEventListener('resize',()=>draw());
function advance(delta,requireConclusion=false){
 if(requireConclusion&&!radio()){message('Choose a conclusion, or use Next to leave this clip unreviewed.');return}
 const before=filtered(),index=before.findIndex(c=>c.uid===active),target=before[index+delta]?.uid;capture();
 if(target){load(target,delta>0);return}
 const remaining=filtered();if(!remaining.some(c=>c.uid===active)){load(remaining[0]?.uid,!!remaining.length);return}
 options();$('clipSelect').value=active;progress();message('End of this selection. Export your conclusions or choose another group.');
}
function mergeReview(payload,origin){
 if(!payload||typeof payload!=='object'||!payload.reviews||Array.isArray(payload.reviews))throw Error('A review JSON with a reviews object is required');
 if(payload.package_id!==KEY&&!catalog.packages.some(p=>p.package_id===payload.package_id))throw Error('Unrecognized package ID; no existing labels were changed');
 let added=0,conflicts=0,unknown=0;
 for(const [id,raw] of Object.entries(payload.reviews)){
  if(!raw||typeof raw!=='object'||Array.isArray(raw))throw Error('Invalid review entry');
  if(raw.conclusion&&!labels[raw.conclusion])throw Error('Unknown conclusion value');
  const r={...raw,conclusion:raw.conclusion==='not_a_squat'?'not_expected_exercise':raw.conclusion||null,notes:String(raw.notes||'')};
  if(!meaningful(r))continue;
  const c=payload.package_id===KEY?clips.find(c=>c.uid===id):clips.find(c=>c.package_id===payload.package_id&&c.clip_id===id);
  if(!c){state.unmapped_reviews.push({origin,package_id:payload.package_id,clip_id:id,review:r});unknown++;continue}
  const old=state.reviews[c.uid];
  if(!meaningful(old)){state.reviews[c.uid]=r;added++}
  else if((old.conclusion||null)!==r.conclusion||(old.notes||'')!==r.notes){
   if(!state.conflicts.some(x=>x.uid===c.uid&&JSON.stringify(x.incoming)===JSON.stringify(r))){state.conflicts.push({uid:c.uid,origin,incoming:r});conflicts++}}
 }
 if(Array.isArray(payload.conflicts))for(const c of payload.conflicts)if(!state.conflicts.some(x=>JSON.stringify(x)===JSON.stringify(c)))state.conflicts.push(c);
 if(Array.isArray(payload.unmapped_reviews))state.unmapped_reviews.push(...payload.unmapped_reviews);
 for(const key of ['reviewer_name_or_id','specialty_and_limits','overall_conclusions']){
  if(payload[key]&&!state[key])state[key]=String(payload[key]);
  else if(payload[key]&&state[key]!==payload[key])state.imported_reviewer_details.push({origin,field:key,value:String(payload[key])});
 }
 return {added,conflicts,unknown};
}
function showConflicts(){
 $('conflicts').innerHTML=state.conflicts.length?`<strong>${state.conflicts.length} imported label conflict(s)</strong><p>Both versions are preserved in exports. Resolve each below; nothing has been silently overwritten.</p>`+state.conflicts.map((c,i)=>`<p>${esc(c.uid)} · current: ${esc(labels[state.reviews[c.uid]?.conclusion]||'blank')} / ${esc(state.reviews[c.uid]?.notes||'')}<br>Imported: ${esc(labels[c.incoming?.conclusion]||'blank')} / ${esc(c.incoming?.notes||'')}<br><button data-conflict="${i}" data-choice="current">Keep current</button><button data-conflict="${i}" data-choice="incoming">Use imported</button></p>`).join(''):'';
}
function exportLabels(){capture();const payload={...state,exported_at:new Date().toISOString(),catalog_version:catalog.generated_at,
 review_mode:'expert visible-form review; model or publisher poses available; not a blinded pose benchmark',
 clips:clips.map(c=>({...c,doctor_conclusion:state.reviews[c.uid]?.conclusion||null,doctor_notes:state.reviews[c.uid]?.notes||''}))};
 const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='theia_all_video_conclusions_'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message('Exported all conclusions, including partial reviews and any unresolved conflicts.');}
for(const id of ['group','source','reviewStatus'])$(id).onchange=()=>{capture();if(id==='group')groups();load(active)};
$('search').oninput=()=>{capture();load(active)};
$('clipSelect').onchange=e=>{capture();load(e.target.value)};
$('view').onchange=()=>setVideo(!video.paused&&!video.ended,true);$('speed').onchange=()=>video.playbackRate=Number($('speed').value);
$('previous').onclick=()=>advance(-1);$('next').onclick=()=>advance(1);$('saveNext').onclick=()=>advance(1,true);
$('save').onclick=()=>{capture();progress();message('Saved. Export JSON to keep a portable backup.')};
$('clear').onclick=()=>{document.querySelectorAll('input[name="conclusion"]').forEach(r=>r.checked=false);capture();progress()};
document.querySelectorAll('input[name="conclusion"]').forEach(r=>r.onchange=()=>{capture();progress()});
$('notes').oninput=capture;for(const id of ['reviewer','specialty','overall'])$(id).oninput=capture;
$('nextUnreviewed').onclick=()=>{capture();const list=filtered(),i=list.findIndex(c=>c.uid===active);const target=[...list.slice(i+1),...list.slice(0,i+1)].find(c=>!state.reviews[c.uid]?.conclusion);if(target)load(target.uid,true);else message('All clips in this selection are reviewed.')};
$('summary').onclick=e=>{const b=e.target.closest('[data-uid]');if(!b)return;capture();$('group').value='all';$('reviewStatus').value='all';$('search').value='';groups();$('source').value='all';load(b.dataset.uid);$('title').scrollIntoView({behavior:'smooth'})};
$('export').onclick=exportLabels;$('import').onclick=()=>$('importFile').click();
$('importFile').onchange=async e=>{capture();const reports=[];for(const f of e.target.files){const backup=structuredClone(state);try{const result=mergeReview(JSON.parse(await f.text()),f.name);reports.push(`${f.name}: ${result.added} added, ${result.conflicts} conflicts, ${result.unknown} preserved unmatched`)}catch(error){state=backup;reports.push(`${f.name}: ${error.message}`)}}
 hydrate();persist();showConflicts();load(active);message(reports.join(' · '));e.target.value='';};
$('conflicts').onclick=e=>{const b=e.target.closest('[data-conflict]');if(!b)return;capture();const c=state.conflicts[Number(b.dataset.conflict)];if(b.dataset.choice==='incoming')state.reviews[c.uid]=c.incoming;state.conflicts.splice(Number(b.dataset.conflict),1);persist();showConflicts();load(active)};
video.onerror=()=>message('Video could not load. Extract the complete folder and keep media beside this page.');
function hydrate(){for(const [key,id] of [['reviewer_name_or_id','reviewer'],['specialty_and_limits','specialty'],['overall_conclusions','overall']])$(id).value=state[key]||''}
try{const saved=JSON.parse(localStorage.getItem(KEY)||'null');if(saved?.package_id===KEY&&saved.reviews)state={...state,...saved}}catch(e){$('storageWarning').textContent='Saved browser data could not be read. Import your last exported JSON.'}
for(const r of Object.values(state.reviews))if(r?.conclusion==='not_a_squat')r.conclusion='not_expected_exercise';
for(const p of catalog.packages){try{const old=JSON.parse(localStorage.getItem(p.package_id)||'null');if(old?.reviews)mergeReview(old,'Previous browser review: '+p.package_id)}catch(e){}}
hydrate();groups();showConflicts();load(clips[0]?.uid);
window.addEventListener('beforeunload',capture);
// Local catalog updates add newly completed clips without replacing review state.
setInterval(()=>{const s=document.createElement('script');s.src='catalog.js?refresh='+Date.now();s.onload=()=>{const next=window.THEIA_CATALOG;if(next.generated_at!==catalog.generated_at){capture();const was=active,time=video.currentTime,playing=!video.paused;catalog=next;clips=next.clips;groups();load(was,playing,time);}s.remove()};s.onerror=()=>s.remove();document.head.appendChild(s)},60000);
