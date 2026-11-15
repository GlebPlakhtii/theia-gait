'use strict';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data,groups=new Map(),current=null,pose=null,request=0;
const number=n=>n===null||n===undefined?'Unavailable':Number(n).toFixed(3);
function listing(){const q=$('search').value.toLowerCase(),split=$('split').value;
 const items=data.inventory.filter(c=>(!split||c.split===split)&&`${c.clip_id} ${c.source} ${c.subject}`.toLowerCase().includes(q));
 $('clip').innerHTML=items.map(c=>`<option value="${esc(c.clip_id)}">${esc(c.clip_id)} · ${esc(c.split)} · ${c.windows} windows</option>`).join('');
 if(items.length)select(items.some(c=>c.clip_id===current)?current:items[0].clip_id);else{$('clipInfo').textContent='No matching clips';$('events').textContent=''}
}
async function select(cid){const token=++request;current=cid;$('clip').value=cid;pose=null;
 const v=data.videos[cid],inventory=data.inventory.find(c=>c.clip_id===cid),windows=groups.get(cid)||[];
 $('clipInfo').textContent=`${cid} · ${inventory.split} · ${inventory.method||'pose unavailable'} · ${windows.length} prediction windows`;
 $('video').src='/gait_metrics/videos/'+v.video;$('videoBox').style.width=`${Math.min(800,540*v.width/v.height)}px`;$('videoBox').style.aspectRatio=`${v.width}/${v.height}`;
 $('overlay').width=v.width;$('overlay').height=v.height;
 $('sourceLink').href='/gait_metrics/?clip='+encodeURIComponent(cid)+(inventory.method==='sapiens'?'&method=sapiens':'');
 const candidates=windows.filter(w=>w.events.some(e=>e.prediction_state==='positive'));
 const displayed=candidates.length?candidates:windows.slice(0,3);
 $('events').innerHTML=displayed.map(w=>`<article class="${w.events.some(e=>e.prediction_state==='positive')?'flagged':''}"><strong>${w.start_time_s.toFixed(2)}–${w.end_time_s.toFixed(2)} s · ${esc(w.split)}</strong><p>${w.events.filter(e=>e.prediction_state==='positive').map(e=>`${esc(e.target_id)} · ${esc(e.region)} · ${esc(e.prediction_basis)}`).join('<br>')||'No positive review flag in this window; unavailable targets may remain.'}</p><details><summary>Measured evidence and model predictions</summary><div class="tableWrap"><table><tr><th>Target</th><th>Teacher</th><th>Model</th><th>Measured</th><th>Predicted</th></tr>${w.events.map(e=>`<tr><td>${esc(e.target_id)}</td><td>${esc(e.teacher_state)}</td><td>${esc(e.prediction_state)}</td><td>${number(e.observed_value)} ${esc(e.unit)}</td><td>${number(e.predicted_value)} ${esc(e.unit)}</td></tr>`).join('')}</table></div></details><button data-time="${w.start_time_s}">Inspect interval</button></article>`).join('')||`<p>${esc(inventory.reason||'Model predictions are not available yet.')}</p>`;
 try{const metricPath=windows[0]?.metric_path||inventory.metric_path;const prefix=metricPath?.includes('/harmonized/')?'harmonized/':'';
 const response=await fetch('/gait_metrics/'+prefix+'clips/'+encodeURIComponent(cid)+'/viewer.json');if(response.ok){const p=await response.json();if(token===request)pose=p;}}
 catch(e){$('clipInfo').textContent+=' · Pose overlay unavailable';}draw();
}
function draw(){const ctx=$('overlay').getContext('2d'),v=data?.videos[current],t=$('video').currentTime;
 ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);$('time').textContent=`Clip ${t.toFixed(2)} s`;
 if(!pose||!v)return;let i=0;for(let j=1;j<pose.timestamp_s.length;j++)if(Math.abs(pose.timestamp_s[j]-t)<Math.abs(pose.timestamp_s[i]-t))i=j;
 if(Math.abs(pose.timestamp_s[i]-t)>.12)return;
 const point=name=>{const j=pose.joint_names.indexOf(name);return j>=0&&pose.quality_mask[i][j]?pose.xy_pixels[i][j]:null};
 function path(names,color,width,closed=false){const p=names.map(point);if(p.some(x=>!x))return;ctx.beginPath();ctx.moveTo(...p[0]);p.slice(1).forEach(x=>ctx.lineTo(...x));if(closed)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineJoin='round';ctx.lineCap='round';ctx.stroke();}
 if($('pose').checked){for(const side of ['left','right']){path([side+'_shoulder',side+'_elbow',side+'_wrist'],'#87bce8',Math.max(2,v.height/300));path([side+'_shoulder',side+'_hip',side+'_knee',side+'_ankle'],'#87bce8',Math.max(2,v.height/300));}path(['left_shoulder','right_shoulder'],'#87bce8',3);path(['left_hip','right_hip'],'#87bce8',3);}
 if($('regions').checked){const regions=new Set((groups.get(current)||[]).filter(w=>t>=w.start_time_s&&t<w.end_time_s).flatMap(w=>w.events.filter(e=>e.prediction_state==='positive').map(e=>e.region)));
 ctx.globalAlpha=.6;for(const r of regions){const color='#ffc451',width=Math.max(10,v.height/55);
 if(r==='trunk')path(['left_shoulder','right_shoulder','right_hip','left_hip'],color,width,true);
 if(r==='hips')path(['left_hip','right_hip'],color,width);if(r==='shoulders')path(['left_shoulder','right_shoulder'],color,width);
 if(r==='knees')for(const side of ['left','right'])path([side+'_hip',side+'_knee',side+'_ankle'],color,width);
 if(r==='arms')for(const side of ['left','right'])path([side+'_shoulder',side+'_elbow',side+'_wrist'],color,width);}ctx.globalAlpha=1;}
}
function load(d){data=d;for(const w of d.predictions.windows){if(!groups.has(w.clip_id))groups.set(w.clip_id,[]);groups.get(w.clip_id).push(w);}
 const complete=d.evaluation?.status==='complete';$('status').textContent=complete?'Fine-tuned checkpoint and evaluation available':`Preparation status: ${d.training.status||'training not started'}. No completed training result is claimed.`;
 $('summary').innerHTML=[['Clips accounted for',d.inventory.length],['Pose windows',d.coverage.windows||0],['Supported classifier targets',d.coverage.primary_labels_enabled||0],['Training epochs',d.history.length]].map(([name,value])=>`<div class="stat"><strong>${value}</strong>${name}</div>`).join('');
 const artifactNames={'evaluation/results.json':complete?'Evaluation JSON':'CPU smoke check (partial)','label_protocol.json':'Label definitions','label_coverage.json':'Label coverage','backbone_update_audit.json':'Backbone update audit','checkpoints/best.pt':'Fine-tuned checkpoint'};
 $('artifacts').innerHTML=(d.artifacts||[]).map(x=>`<a href="${esc(x)}">${esc(artifactNames[x])}</a>`).join(' · ');
 $('figures').innerHTML=(d.figures||[]).map(x=>`<img src="figures/${x}.png" alt="${esc(x.replaceAll('_',' '))}">`).join('');
 const test=complete?d.evaluation.reports.test_sapiens:null;
 $('metrics').innerHTML=test?`<p>Held-out participants: ${esc(test.participants.join(', '))}. Automatic-label average precision: ${number(test.macro_average_precision)}. Labels with insufficient support are not scored.</p><div class="tableWrap"><table><tr><th>Target</th><th>Windows</th><th>Precision</th><th>Recall</th><th>F1</th><th>Average precision</th></tr>${test.labels.map(r=>`<tr><td>${esc(r.id)}</td><td>${r.n}</td><td>${number(r.precision)}</td><td>${number(r.recall)}</td><td>${number(r.f1)}</td><td>${number(r.average_precision)}</td></tr>`).join('')}</table></div>`:'Held-out results will appear after training and evaluation.';
 $('provenance').textContent=`Report built ${d.created_at}. Model scores are uncalibrated; short-window changes are not complete-stride measurements.`;listing();
}
$('search').oninput=listing;$('split').onchange=listing;$('clip').onchange=e=>select(e.target.value);$('pose').onchange=draw;$('regions').onchange=draw;
$('events').onclick=e=>{const button=e.target.closest('[data-time]');if(button){$('video').pause();$('video').currentTime=Number(button.dataset.time)+.1;}};
$('video').addEventListener('timeupdate',draw);$('video').addEventListener('seeked',draw);
if('requestVideoFrameCallback'in $('video')){const tick=()=>{draw();$('video').requestVideoFrameCallback(tick)};$('video').requestVideoFrameCallback(tick);}
fetch('report_data.json',{cache:'no-store'}).then(r=>r.json()).then(load).catch(e=>{$('status').textContent='Unable to load report: '+e.message;});
