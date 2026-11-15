const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const M=require('../scripts/progait_metrics.js');
const catalog=JSON.parse(fs.readFileSync('deliverables/Theia_ProGait_Metric_Review/catalog.json'));

test('only left/right knee min/max have definitions; catalog contains only ProGait',()=>{
 assert.equal(catalog.clips.length,1017);
 for(const c of catalog.clips){
  assert.equal(c.packet_key,'progait');
  for(const def of M.definitions)assert.ok(Object.hasOwn(c.automatic_measurements,def.id),def.id);
 }
 assert.equal(M.definitions.length,4);assert.ok(M.definitions.every(d=>/^((left)|(right))_projected_knee_(min|max)_deg$/.test(d.id)));
});
test('missing knee values stay unavailable, never coerced to zero',()=>{
 assert.equal(M.value({automatic_measurements:{}},M.definitions[0]),null);
 assert.equal(M.value({automatic_measurements:{left_projected_knee_min_deg:0}},M.definitions[0]),0);
});
test('zero corrections survive JSON round trips and blank annotations are omitted',()=>{
 const raw={left_projected_knee_min_deg:{corrected_value:0,notes:'Reviewed projection'},right_projected_knee_min_deg:{}};
 const a=M.normalize(raw);assert.equal(a.left_projected_knee_min_deg.corrected_value,0);assert.equal(Object.keys(a).length,1);
 assert.deepEqual(M.normalize(JSON.parse(JSON.stringify(a))),a);
});
test('invalid corrections remain recoverable drafts, never valid numeric measurements',()=>{
 for(const text of ['unknown','-1','181','Infinity']){
  const a=M.annotation({correction_input:text},M.definitions.find(d=>d.id==='left_projected_knee_min_deg'));
  assert.equal(a.corrected_value,null);assert.equal(a.correction_input,text);assert.ok(a.correction_error);
 }
});

function reviewHarness(){
 const source=fs.readFileSync('scripts/progait_review.js','utf8'),elements=new Map(),storage=new Map();
 const context2d={clearRect(){}};
 function el(id){if(!elements.has(id))elements.set(id,{value:'',textContent:'',classList:{toggle(){}},setAttribute(){},addEventListener(){},getContext:()=>context2d});return elements.get(id)}
 const sandbox=vm.createContext({window:{THEIA_CATALOG:catalog,ProGaitMetrics:M,addEventListener(){}},document:{getElementById:el,querySelector:()=>null},localStorage:{setItem:(k,v)=>storage.set(k,v)},console});
 vm.runInContext(source.slice(0,source.indexOf("for(const id of ['group','source','reviewStatus'])")),sandbox);
 const run=s=>vm.runInContext(s,sandbox);run('active=clips[0].uid');
 return {run,el,storage};
}
test('metric-only reviews autosave under an isolated key and retain form notes',()=>{
 const h=reviewHarness();h.el('notes').value='Visible gait note';h.el('assessment-left_projected_knee_min_deg').value='cannot_judge';h.el('metric-notes-left_projected_knee_min_deg').value='Oblique camera';
 h.run('capture()');const saved=JSON.parse(h.storage.get('theia-progait-metric-review-v1'));
 const r=saved.reviews[catalog.clips[0].uid];assert.equal(r.notes,'Visible gait note');assert.equal(r.metric_annotations.left_projected_knee_min_deg.notes,'Oblique camera');assert.equal(r.conclusion,null);
 assert.equal(h.storage.has('theia-unified-doctor-review-v1'),false);
});
test('metric-only import is retained; differences create a conflict instead of overwriting',()=>{
 const h=reviewHarness(),uid=catalog.clips[0].uid;
 const payload={package_id:'theia-progait-metric-review-v1',reviews:{[uid]:{metric_annotations:{right_projected_knee_max_deg:{assessment:'needs_review',notes:'Occlusion'}}}}};
 h.run(`mergeReview(${JSON.stringify(payload)},'test')`);assert.equal(h.run('Object.keys(state.reviews).length'),1);
 payload.reviews[uid].metric_annotations.right_projected_knee_max_deg.notes='Second reviewer';
 h.run(`mergeReview(${JSON.stringify(payload)},'test2')`);assert.equal(h.run('state.conflicts.length'),1);assert.equal(h.run('state.reviews[active].metric_annotations.right_projected_knee_max_deg.notes'),'Occlusion');
});
test('legacy unified import includes ProGait labels and excludes other datasets',()=>{
 const h=reviewHarness(),uid=catalog.clips[0].uid;
 h.run(`mergeReview(${JSON.stringify({package_id:'theia-unified-doctor-review-v1',reviews:{[uid]:{conclusion:'cannot_judge'},'original:ARED_001':{conclusion:'form_needs_review'}}})},'legacy')`);
 assert.equal(h.run('Object.keys(state.reviews).length'),1);assert.equal(h.run('state.unmapped_reviews.length'),0);
});
