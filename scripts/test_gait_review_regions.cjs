// Synthetic canvas checks: a head warning must not depend on blurred landmarks.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const elements=new Map();
const document={getElementById(id){if(!elements.has(id))elements.set(id,{checked:true,textContent:'',innerHTML:''});return elements.get(id)}};
const sandbox={window:{},document,fetch:()=>Promise.resolve({json:()=>Promise.resolve({clips:[]})})};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../results/gait_metrics_20261114/review_regions.js'),'utf8'),sandbox);
const renderer=sandbox.window.GaitRegionReview;
renderer.set({events:[{kind:'pose_quality',region:'head',title:'Head landmarks hidden',start_time_s:0,end_time_s:1,evidence_time_s:[0]}]},{});
function draw(xy,valid){const arcs=[],ctx={canvas:{width:1000,height:1000},save(){},restore(){},beginPath(){},arc(...args){arcs.push(args)},fill(){},stroke(){},setLineDash(){}};
 renderer.draw(ctx,{joint_names:['left_shoulder','right_shoulder','left_hip','right_hip','nose','left_ear','right_ear'],xy_pixels:[xy],quality_mask:[valid],clip_frame:[0]},0,.5);
 return arcs;
}
const body=[[90,200],[110,200],[90,400],[110,400]];
const first=draw([...body,[5000,-5000],[8000,9000],[NaN,NaN]],[true,true,true,true,true,true,false]);
assert.equal(first.length,1);assert.deepEqual(first[0].slice(0,2),[100,130]);assert.ok(Math.abs(first[0][2]-56)<1e-9);
const changed=draw([...body,[1,1],[2,2],[3,3]],[true,true,true,true,false,false,false]);
assert.deepEqual(changed,first,'Warning moved with unusable head landmarks');
assert.equal(draw([...body,[0,0],[0,0],[0,0]],[false,false,true,true,false,false,false]).length,0,'No torso anchor must not invent a head position');
assert.equal(draw(Array(7).fill([100,100]),Array(7).fill(true)).length,0,'Zero-length torso must not invent a head position');
console.log('Passed: head warning uses torso anchors, ignores invalid head estimates, and rejects missing/degenerate anchors.');
