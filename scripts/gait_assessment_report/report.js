'use strict';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=n=>n==null?'Unavailable':Number(n).toFixed(3);
const availability=e=>esc([...(e.reasons||[]),...(!e.quality_gate_passed?['No supported measurement/reference']:[]),...(e.predicted_evaluability_score!==null&&e.predicted_evaluability_score<.5?['Model evaluability below 0.5']:[]),...(e.predicted_value!==null&&e.predicted_value<0?['Negative predicted variation withheld']:[])].join('; ')||'Eligible');
let data,current=null,pose=null,windows=[],request=0;
const cache=new Map();
function listing(){
 const q=$('search').value.toLowerCase(),split=$('split').value;
 const items=data.inventory.filter(c=>(!split||c.split===split)&&`${c.clip_id} ${c.source} ${c.subject}`.toLowerCase().includes(q));
 $('clip').innerHTML=items.map(c=>`<option value="${esc(c.clip_id)}">${esc(c.clip_id)} · ${esc(c.split)} · ${c.windows} primary-method windows</option>`).join('');
 if(items.length)select(items.some(c=>c.clip_id===current)?current:items[0].clip_id);
 else {++request;current=null;windows=[];pose=null;$('video').removeAttribute('src');$('video').load();$('clipInfo').textContent='No matching clips';$('events').textContent='';draw();}
}
async function select(cid,preferred=null){
 const token=++request;current=cid;$('clip').value=cid;pose=null;windows=[];draw();
 const v=data.videos[cid],inventory=data.inventory.find(c=>c.clip_id===cid);
 $('events').textContent='Loading predictions…';$('clipInfo').textContent=cid;
 try{
  if(!cache.has(cid)){const r=await fetch('video_predictions/'+encodeURIComponent(cid)+'.json',{cache:'no-store'});cache.set(cid,r.ok?(await r.json()).windows:[]);}
  if(token!==request)return;
  const all=cache.get(cid),methods=[...new Set(all.map(w=>w.method))];
  if(!methods.length)methods.push(inventory.method||'sapiens');
  const method=methods.includes(preferred)?preferred:methods.includes(inventory.method)?inventory.method:methods[0];
  $('poseMethod').innerHTML=methods.map(m=>`<option value="${esc(m)}">${m==='sapiens'?'Sapiens2 1B':'Publisher ProGait'}</option>`).join('');$('poseMethod').value=method;
  windows=all.filter(w=>w.method===method);
  $('clipInfo').textContent=`${cid} · ${inventory.split} · ${method} · ${windows.length} prediction windows · ${inventory.subject||'identity unresolved'}`;
  const url='/gait_metrics/videos/'+v.video;if($('video').getAttribute('src')!==url)$('video').src=url;
  $('videoBox').style.width=`${Math.min(800,540*v.width/v.height)}px`;$('videoBox').style.aspectRatio=`${v.width}/${v.height}`;
  $('overlay').width=v.width;$('overlay').height=v.height;
  $('sourceLink').href='/gait_metrics/?clip='+encodeURIComponent(cid)+(method==='sapiens'?'&method=sapiens':'');
  const candidates=windows.filter(w=>w.events.some(e=>e.prediction_state==='positive'||e.disagreement));
  const displayed=candidates.length?candidates:windows.slice(0,3);
  $('events').innerHTML=displayed.map(w=>{
   const flagged=w.events.some(e=>e.prediction_state==='positive');
   return `<article class="${flagged?'flagged':''}"><strong>${w.start_time_s.toFixed(2)}–${w.end_time_s.toFixed(2)} s · original frames ${w.original_source_first_frame}–${w.original_source_end_frame_exclusive-1}</strong><p>${w.events.filter(e=>e.prediction_state==='positive').map(e=>`${esc(e.target_id)} · ${esc(e.region)} · ${esc(e.prediction_basis)} · metric direction: ${esc(e.metric_change_direction)}`).join('<br>')||'No positive model review flag; unavailable targets may remain.'}</p><details><summary>Teacher, model, reference and disagreements</summary><div class="tableWrap"><table><tr><th>Target</th><th>Teacher</th><th>Model</th><th>Measured</th><th>Predicted</th><th>Reference band</th><th>Disagreement</th><th>Availability / quality</th></tr>${w.events.map(e=>`<tr><td>${esc(e.target_id)}</td><td>${esc(e.teacher_state)}</td><td>${esc(e.prediction_state)}</td><td>${number(e.observed_value)} ${esc(e.unit)}</td><td>${number(e.predicted_value)} ${esc(e.unit)}</td><td>${e.reference?`${number(e.reference.lower)}–${number(e.reference.upper)} (${e.reference.participant_count} training people)`:'Unavailable'}</td><td>${e.disagreement===null?'Unavailable':e.disagreement?'Yes':'No'}</td><td>${availability(e)}</td></tr>`).join('')}</table></div></details><button data-time="${w.start_time_s}">Inspect interval</button></article>`;
  }).join('')||`<p>${esc(inventory.reason||'No supported windows for this clip.')}</p>`;
  const metricPath=windows[0]?.metric_path||inventory.metric_path;
  const prefix=method==='sapiens'&&metricPath?.includes('/harmonized/')?'harmonized/':'';
  const r=await fetch('/gait_metrics/'+prefix+'clips/'+encodeURIComponent(cid)+'/viewer.json');
  if(r.ok){const p=await r.json();if(token===request)pose=p;}
 }catch(e){if(token===request)$('clipInfo').textContent+=' · Unable to load: '+e.message;}
 if(token===request)draw();
}
function draw(){
 const ctx=$('overlay').getContext('2d'),v=data?.videos[current],t=$('video').currentTime;
 ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
 $('time').textContent=`Clip ${t.toFixed(2)} s${v?` · original frame ${v.source_first_frame+Math.floor(t*v.fps)}`:''}`;
 if(!pose||!v)return;
 let i=0;for(let j=1;j<pose.timestamp_s.length;j++)if(Math.abs(pose.timestamp_s[j]-t)<Math.abs(pose.timestamp_s[i]-t))i=j;
 if(Math.abs(pose.timestamp_s[i]-t)>.12)return;
 const point=name=>{const j=pose.joint_names.indexOf(name);return j>=0&&pose.quality_mask[i][j]?pose.xy_pixels[i][j]:null;};
 function path(names,color,width,closed=false){
  const p=names.map(point);if(p.some(x=>!x))return;ctx.beginPath();ctx.moveTo(...p[0]);p.slice(1).forEach(x=>ctx.lineTo(...x));if(closed)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineJoin='round';ctx.lineCap='round';ctx.stroke();
 }
 if($('pose').checked){for(const side of ['left','right']){path([side+'_shoulder',side+'_elbow',side+'_wrist'],'#87bce8',Math.max(2,v.height/300));path([side+'_shoulder',side+'_hip',side+'_knee',side+'_ankle'],'#87bce8',Math.max(2,v.height/300));}path(['left_shoulder','right_shoulder'],'#87bce8',3);path(['left_hip','right_hip'],'#87bce8',3);}
 if($('regions').checked){
  const regions=new Map();for(const w of windows.filter(w=>t>=w.start_time_s&&t<w.end_time_s))for(const e of w.events)if(e.prediction_state==='positive'||e.disagreement)regions.set(e.region,(regions.get(e.region)||false)||e.disagreement===true);
  ctx.globalAlpha=.65;
  for(const [region,disagreement]of regions){const color=disagreement?'#cf8cff':'#ffc451',width=Math.max(10,v.height/55);
   if(region==='trunk')path(['left_shoulder','right_shoulder','right_hip','left_hip'],color,width,true);
   if(region==='hips')path(['left_hip','right_hip'],color,width);if(region==='shoulders')path(['left_shoulder','right_shoulder'],color,width);
   if(region==='knees')for(const side of ['left','right'])path([side+'_hip',side+'_knee',side+'_ankle'],color,width);
   if(region==='arms')for(const side of ['left','right'])path([side+'_shoulder',side+'_elbow',side+'_wrist'],color,width);
  }ctx.globalAlpha=1;
 }
}
function load(d){
 data=d;$('modelTitle').textContent=d.model_name||'Movement assessment experiment';
 const test=d.evaluation.reports.test_sapiens;
 $('status').textContent=`Training and held-out evaluation complete · best epoch ${d.training.best_epoch} · ${d.training.epochs_entered} epochs · ${(d.latency.latency_ms.mean).toFixed(2)} ms model-only inference per window`;
 $('summary').innerHTML=[['Clips accounted for',d.inventory.length],['Pose windows',d.coverage.windows],['Supported classifiers',d.coverage.primary_labels_enabled],['Held-out average precision',number(test.macro_average_precision)],['Prevalence baseline',number(test.baseline_macro_average_precision)]].map(([name,value])=>`<div class="stat"><strong>${value}</strong>${name}</div>`).join('');
 $('artifacts').innerHTML=d.artifacts.map(x=>`<a href="${esc(x)}">${esc(x)}</a>`).join(' · ');
 $('figures').innerHTML=d.figures.map(x=>`<details><summary>${esc(x.replaceAll('_',' '))}</summary><a href="figures/${x}.png"><img loading="lazy" src="figures/${x}.png" alt="${esc(x.replaceAll('_',' '))}"></a></details>`).join('');
 const time=d.evaluation.review_intervals.test_sapiens;
 const coverage=d.review_coverage?.test_sapiens;
 $('metrics').innerHTML=`<p>Held-out participants: ${esc(test.participants.join(', '))}. ${test.clips} clips, ${test.physical_trials} physical trials, ${test.windows} windows. Average precision: ${number(test.macro_average_precision)}; train-prevalence baseline: ${number(test.baseline_macro_average_precision)}. Performance is withheld when both label signs are not represented.</p><p>Review duration ${number(time.review_duration_s)} of ${number(time.observed_duration_s)} seconds after merging overlaps. ${time.windows_with_no_available_target} windows have no available review target.${coverage?` Evaluable duration: ${number(coverage.evaluable_duration_s)} seconds; fully abstained windows: ${(100*coverage.fully_abstained_window_fraction).toFixed(1)}%.`:''}</p><div class="tableWrap"><table><tr><th>Target</th><th>Windows</th><th>People</th><th>Precision</th><th>Recall</th><th>F1</th><th>Average precision</th><th>Baseline AP</th></tr>${test.labels.map(r=>`<tr><td>${esc(r.id)}${r.enabled?'':' (disabled)'}</td><td>${r.n}</td><td>${r.support.participants.length}</td><td>${number(r.precision)}</td><td>${number(r.recall)}</td><td>${number(r.f1)}</td><td>${number(r.average_precision)}</td><td>${number(r.baseline_prevalence.average_precision)}</td></tr>`).join('')}</table></div><div class="tableWrap"><table><tr><th>Measurement</th><th>Windows</th><th>MAE</th><th>Train-median MAE</th><th>Quality-gated MAE</th><th>Unit</th></tr>${test.outcomes.map(o=>`<tr><td>${esc(o.id)}</td><td>${o.n}</td><td>${number(o.mae)}</td><td>${number(o.baseline_median.mae)}</td><td>${number(o.quality_gated.mae)}</td><td>${esc(o.unit)}</td></tr>`).join('')}</table></div>`;
 $('provenance').textContent=`Report built ${d.created_at}. Uncalibrated model scores. One-second changes are not full-stride ROM. Unknown identities and provisional cohorts are exploratory only.`;
 const cid=new URLSearchParams(location.search).get('clip');if(cid&&d.inventory.some(c=>c.clip_id===cid))current=cid;
 listing();
}
$('search').oninput=listing;$('split').onchange=listing;$('clip').onchange=e=>select(e.target.value);$('poseMethod').onchange=e=>select(current,e.target.value);$('pose').onchange=draw;$('regions').onchange=draw;
$('events').onclick=e=>{const b=e.target.closest('[data-time]');if(b){$('video').pause();$('video').currentTime=Number(b.dataset.time)+.1;}};
$('video').addEventListener('timeupdate',draw);$('video').addEventListener('seeked',draw);
if('requestVideoFrameCallback'in $('video')){const tick=()=>{draw();$('video').requestVideoFrameCallback(tick);};$('video').requestVideoFrameCallback(tick);}
fetch('report_data.json',{cache:'no-store'}).then(r=>r.json()).then(load).catch(e=>{$('status').textContent='Unable to load report: '+e.message;});
