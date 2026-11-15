'use strict';
const P=window.POINTS,$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
for(const [key,name] of Object.entries(P.names)){if(key==='baseline')continue;const o=document.createElement('option');o.value=key;o.textContent=name;$('candidate').append(o);}
$('candidate').value=P.selected;
P.classes.forEach((name,j)=>{if(!P.mask.some(row=>row[j]))return;const o=document.createElement('option');o.value=j;o.textContent=name.replaceAll('_',' ');$('criterion').append(o);});
function node(tag,attrs,text){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function draw(){
 const svg=$('scatter');svg.replaceChildren();const unit=$('unit').value,which=$('criterion').value,key=$('candidate').value,base=P.probabilities.baseline,pred=P.probabilities[key];
 const x=v=>80+v*800,y=v=>505-v*440;
 for(let i=0;i<=10;i++){const v=i/10;svg.append(node('line',{x1:x(v),x2:x(v),y1:65,y2:505,stroke:'#dce4ea'}),node('line',{x1:80,x2:880,y1:y(v),y2:y(v),stroke:'#dce4ea'}));if(i%2===0)svg.append(node('text',{x:x(v),y:530,'text-anchor':'middle',fill:'#476273','font-size':13},v.toFixed(1)),node('text',{x:65,y:y(v)+4,'text-anchor':'end',fill:'#476273','font-size':13},v.toFixed(1)));}
 svg.append(node('line',{x1:80,y1:505,x2:880,y2:65,stroke:'#91a3ae','stroke-dasharray':'5 5'}));
 let points=[],clips=new Map();
 P.rows.forEach((r,i)=>P.classes.forEach((label,j)=>{if(!P.mask[i][j]||(which!=='all'&&j!==+which))return;const target=P.targets[i][j];if(unit==='clip'){if(!clips.has(r.clip_id))clips.set(r.clip_id,{name:r.clip_id,person:r.subject,n:0,a:0,b:0});const c=clips.get(r.clip_id);c.n++;c.a+=Number((base[i][j]>=.5)!==Boolean(target));c.b+=Number((pred[i][j]>=.5)!==Boolean(target));}else points.push({a:base[i][j],b:pred[i][j],target,text:`${r.clip_id} · ${r.subject} · ${r.start_time_s.toFixed(2)} s · ${label.replaceAll('_',' ')} · target ${target} · existing ${base[i][j].toFixed(3)} → candidate ${pred[i][j].toFixed(3)}`});}));
 if(unit==='clip')points=[...clips.values()].map(c=>({a:c.a/c.n,b:c.b/c.n,target:2,text:`${c.name} · ${c.person} · ${c.n} decisions · error ${(100*c.a/c.n).toFixed(1)}% → ${(100*c.b/c.n).toFixed(1)}%`}));
 for(const p of points){const dot=node('circle',{cx:x(p.a),cy:y(p.b),r:unit==='clip'?6:3.5,fill:p.target===0?'#5087b6':p.target===1?'#db913e':'#278e7d',opacity:unit==='clip'?.8:.55,tabindex:0});dot.append(node('title',{},p.text));dot.onmouseenter=dot.onfocus=()=>{$('point-info').textContent=p.text;};svg.append(dot);}
 const label=unit==='clip'?'error rate':'uncalibrated model score';svg.append(node('text',{x:480,y:565,'text-anchor':'middle',fill:'#193044','font-size':16},'Existing demo · '+label),node('text',{x:22,y:285,transform:'rotate(-90 22 285)','text-anchor':'middle',fill:'#193044','font-size':16},P.names[key]+' · '+label),node('text',{x:80,y:35,fill:'#193044','font-size':17},unit==='clip'?'One point per video clip':'One point per available window criterion'));
 $('point-count').textContent=`${points.length} ${unit==='clip'?'clips':'window decisions'}`;$('point-info').textContent='Hover or focus a point to inspect its source.';
}
for(const id of ['candidate','criterion','unit'])$(id).onchange=draw;draw();
