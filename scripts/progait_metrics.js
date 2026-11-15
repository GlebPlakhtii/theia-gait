(function(root){
 'use strict';
 const definitions=['left','right'].flatMap(side=>[
  {id:`${side}_projected_knee_min_deg`,label:`${side==='left'?'Left':'Right'} knee · minimum`,side,bound:'Minimum',unit:'°',max:180,description:'Smallest valid 2D hip–knee–ankle angle in this clip. 180° is straight in projection; a smaller angle indicates more apparent bending. Left/right follow publisher keypoint labels, not screen position.'},
  {id:`${side}_projected_knee_max_deg`,label:`${side==='left'?'Left':'Right'} knee · maximum`,side,bound:'Maximum',unit:'°',max:180,description:'Largest valid 2D hip–knee–ankle angle in this clip. Camera perspective, occlusion and misplaced keypoints can affect this measurement. It is not calibrated anatomical extension.'}
 ]);
 const assessments={'':'Not annotated',plausible:'Plausible from this view',needs_review:'Measurement needs review',cannot_judge:'Cannot judge',not_applicable:'Not applicable'};
 const value=(clip,def)=>{const n=(def.location==='clip'?clip:clip.automatic_measurements||{})[def.id];return typeof n==='number'&&Number.isFinite(n)?n*(def.factor||1):null};
 function annotation(raw={},def){
  if(!raw||typeof raw!=='object'||Array.isArray(raw))throw Error('Invalid metric annotation');
  const assessment=raw.assessment||'';if(!Object.hasOwn(assessments,assessment))throw Error('Unknown metric assessment');
  const input=String(raw.correction_input??raw.corrected_value??'').trim();
  const numeric=input===''?null:Number(input);
  const valid=input===''||(Number.isFinite(numeric)&&numeric>=0&&(def.max===undefined||numeric<=def.max));
  return {assessment,correction_input:input,corrected_value:valid?numeric:null,unit:def.unit,
   correction_error:valid?null:`Enter a number from 0${def.max===undefined?' or greater':' to '+def.max} (${def.unit}).`,notes:String(raw.notes||'')};
 }
 const hasAnnotation=a=>!!(a&&(a.assessment||a.correction_input||a.corrected_value!=null||a.notes));
 function normalize(raw={}){
  if(!raw||typeof raw!=='object'||Array.isArray(raw))throw Error('Invalid metric annotations');
  const result={};for(const [id,a] of Object.entries(raw)){
   const def=definitions.find(d=>d.id===id);if(!def)throw Error('Unknown metric: '+id);
   const clean=annotation(a,def);if(hasAnnotation(clean))result[id]=clean;
  }return result;
 }
 const signature=review=>JSON.stringify({conclusion:review?.conclusion||null,notes:review?.notes||'',metrics:Object.fromEntries(Object.entries(normalize(review?.metric_annotations||{})).sort(([a],[b])=>a.localeCompare(b)))});
 const api={definitions,assessments,value,annotation,hasAnnotation,normalize,signature};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ProGaitMetrics=api;
})(typeof window==='undefined'?globalThis:window);
