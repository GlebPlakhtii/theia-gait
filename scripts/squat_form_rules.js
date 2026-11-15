/* Shared browser/Node decision engine. Research defaults, not clinical cutoffs. */
(function(root){
  'use strict';
  const finite=x=>typeof x==='number'&&Number.isFinite(x);
  function assess(clip,config){
    const f=clip.features,c=config,issues=[],rules=[];
    if(!f.completeRep) return {status:'context',label:'Context — no complete repetition',issues:['This segment is not a detected complete repetition.'],rules:[],flags:[],clinicalVerdict:null};
    if(!finite(f.coverage)||f.coverage<c.minCoverage)issues.push('Insufficient observed shoulder/hip/knee/ankle coverage.');
    if(!finite(f.flip)||f.flip>c.maxFlip)issues.push('3D reconstruction changes too much under horizontal-flip testing.');
    if(!finite(f.ikResidual)||f.ikResidual>c.maxIk)issues.push('The articulated OpenSim model does not fit the estimated markers closely enough.');
    if(!finite(f.reprojection)||f.reprojection>c.maxReprojection)issues.push('Estimated 3D geometry has weak image-plane reprojection consistency.');
    if(finite(f.modelAngleDifference)&&f.modelAngleDifference>15)issues.push('OpenSim and the unconstrained 3D estimate disagree on knee flexion.');
    function rule(id,title,value,limit,direction,deadband,unit,reason,fix,evidence,enabled=true){
      let state='unavailable';
      if(enabled&&finite(value)&&finite(limit)){
        const delta=direction==='above'?value-limit:limit-value;
        state=delta>deadband?'flag':delta>=-deadband?'borderline':'pass';
      }
      rules.push({id,title,value:finite(value)?value:null,limit,unit,state,reason,fix,evidence,
        thresholdOrigin:'Engineering default for this research prototype; not a validated diagnostic boundary.'});
    }
    rule('symmetry','Left/right knee-flexion difference',f.asymmetry,c.maxAsymmetry,'above',c.angleDeadband,'°',
      '90th-percentile simultaneous difference after 3D reconstruction and OpenSim fitting.',
      'Review foot setup and whether both legs descend and rise together. If equal bilateral motion is intended, rehearse matching depth; confirm the asymmetry in a second view before correcting it.',
      ['GWYNNE2014','GRABER2023']);
    rule('phase','Left/right bottom timing',f.phaseFraction,c.maxPhaseFraction,'above',.03,'fraction of rep',
      'Time between the left and right maximum-flexion frames, divided by repetition duration. Flat-bottom pauses can make this estimate ambiguous.',
      'Review the marked bottom frames. If one side consistently leads, practice a controlled synchronized descent and ascent at the prescribed load.',
      ['GRABER2023']);
    if(clip.profile==='ground'){
      rule('depth','Configured depth target',f.depth,c.groundDepth-c.depthTolerance,'below',c.angleDeadband,'°',
        'The default 90° target is an editable demonstration prescription, with a 10° tolerance. It is not a universal requirement for a good squat.',
        'Confirm the intended squat depth with the coach or clinician. If this target is appropriate, work toward that prescribed depth within the agreed comfortable range; otherwise change the target.',
        ['GRABER2023']);
      rule('excursion','Knee range of motion',f.meanRom,c.minExcursion,'below',c.angleDeadband,'°',
        'Mean left/right 5th-to-95th-percentile knee excursion; a configured task target rather than a medical cutoff.',
        'Check whether the task intentionally uses partial squats. If a full repetition is prescribed, review both the bottom and return positions instead of increasing depth automatically.',
        ['GRABER2023']);
    }
    rule('depth_change','Depth change from source reference',f.depthDrop,c.depthDrop,'above',c.angleDeadband,'°',
      'Compared with the first three quality-eligible repetitions in this same source. Those repetitions are a consistency reference, not an expert-approved normal.',
      'Review why the repetition became shallower: prescribed depth, equipment setup, comfort, or technique. Compare with the linked reference repetitions before changing the exercise or load.',
      ['NASA2014','GRABER2023'],finite(f.depthDropFraction)&&f.depthDropFraction>=c.minDepthDropFraction);
    rule('trunk','Trunk change at matched knee flexion',f.trunkChange,c.trunkChange,'above',c.angleDeadband,'°',
      'Additional trunk inclination at approximately 60° knee flexion, relative to the same-source reference. Forward lean alone is not a fault.',
      'Review trunk–tibia coordination against the reference repetition and the intended squat strategy. Maintain the prescribed strategy; this estimate cannot diagnose lumbar rounding.',
      ['GRABER2023']);
    rule('frontal','Estimated inward knee excursion',f.frontalChange,c.frontalChange,'above',c.angleDeadband,'°',
      'Change in signed frontal-plane projected knee angle from the top to the deepest quarter, evaluated only when the estimated camera yaw is within the configured frontal limit.',
      'Confirm the finding from a frontal view. If unintended inward movement is visible, review knee alignment with the foot and stance setup during a controlled repetition.',
      ['GWYNNE2014'],finite(f.yaw)&&f.yaw<=c.maxFrontalYaw);
    const flags=rules.filter(r=>r.state==='flag'),borderline=rules.filter(r=>r.state==='borderline');
    let status,label;
    if(issues.length){status='insufficient';label='Insufficient evidence';}
    else if(flags.length){status='bad';label='Bad under selected rules — review';}
    else if(borderline.length){status='borderline';label='Borderline — review';}
    else if(rules.some(r=>r.state==='pass')){status='good';label='Good under selected checks';}
    else{status='insufficient';label='Insufficient evidence';}
    return {status,label,issues,rules,flags:flags.map(r=>r.id),clinicalVerdict:null,
      explanation:'Prototype screening of visible movement. It does not measure pain, load distribution, muscle force, or injury risk.',
      assessed:rules.filter(r=>r.state!=='unavailable').length,totalChecks:rules.length};
  }
  const api={assess};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.SquatRules=api;
})(typeof window!=='undefined'?window:globalThis);

if(typeof require!=='undefined'&&require.main===module){
  const fs=require('fs'),assert=require('assert');
  if(process.argv[2]==='--test'){
    const c={minCoverage:.8,maxFlip:.1,maxIk:.06,maxReprojection:.12,maxAsymmetry:15,maxPhaseFraction:.15,
      groundDepth:90,depthTolerance:10,minExcursion:60,depthDrop:15,trunkChange:15,frontalChange:10,maxFrontalYaw:30,minDepthDropFraction:.2,angleDeadband:5};
    const base={profile:'ground',features:{completeRep:true,coverage:1,flip:.02,ikResidual:.02,reprojection:.03,
      modelAngleDifference:2,asymmetry:2,phaseFraction:0,depth:100,meanRom:80,yaw:60,frontalChange:40,
      depthDrop:null,depthDropFraction:null,trunkChange:null}};
    const variant=updates=>({...base,features:{...base.features,...updates}});
    assert.equal(module.exports.assess(base,c).status,'good');
    assert.equal(module.exports.assess(variant({coverage:.5,asymmetry:40}),c).status,'insufficient');
    assert.equal(module.exports.assess(variant({depth:65}),c).status,'bad');
    assert.equal(module.exports.assess(variant({asymmetry:15}),c).status,'borderline');
    assert.equal(module.exports.assess(variant({completeRep:false}),c).status,'context');
    assert.equal(module.exports.assess(variant({yaw:60,frontalChange:40}),c).rules.find(x=>x.id==='frontal').state,'unavailable');
    assert.equal(module.exports.assess({...variant({depth:30,meanRom:20}),profile:'ared'},c).status,'good');
    assert.equal(module.exports.assess(base,{...c,groundDepth:120}).status,'bad');
    console.log('Rule checks passed: quality overrides, depth target, deadband, camera gate, separate ARED profile.');
  }else{
    const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
    for(const row of data.clips)row.assessment=module.exports.assess(row,data.defaults);
    const counts={};for(const row of data.clips)counts[row.assessment.status]=(counts[row.assessment.status]||0)+1;
    data.summary={counts,scope:'Research rule outputs; not clinician-verified form verdicts'};
    fs.writeFileSync(process.argv[3],JSON.stringify(data,null,2)+'\n');console.log(JSON.stringify(counts));
  }
}
