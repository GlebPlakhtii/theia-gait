// Exercise the actual report status function with independent stage scenarios.
// No browser, production artifacts, workers, or network are changed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '../results/gait_metrics_20261114/comparison.js'), 'utf8');
const start = script.indexOf('async function progress(){');
const end = script.indexOf('\nfunction rerunExclusions', start);
assert(start >= 0 && end > start);
const source = script.slice(start, end);

async function run(overrides = {}) {
  const production = {phase:'pose_complete', status:'complete', completed_clips:219,
    total_jobs:219, completed_frames:39786, total_frames:39786, updated_at:new Date().toISOString()};
  const catalog = {counts:{observations_extracted:596}, clips:Array(596).fill({})};
  const post = {status:'derived_outputs_complete'};
  const report = {status:'final', updated_at:'fixed-comparison-generation'};
  const audit = {status:'passed_data_gate', comparison_updated_at:report.updated_at,
    native_jobs:219, sampled_pose_reviews:219, curated_clips:596};
  const artifacts = {'production_status.json':production, 'catalog.json':catalog,
    'postprocessing_status.json':post, 'data_handoff_audit.json':audit};
  for(const [name, changes] of Object.entries(overrides)) {
    if(name === 'report')Object.assign(report, changes);
    else artifacts[name] = changes === null ? null : {...artifacts[name], ...changes};
  }
  const elements = new Map();
  const queried = [];
  const scope = vm.createContext({Date, report,
    $:id=>{if(!elements.has(id))elements.set(id, {});return elements.get(id)},
    get:async name=>{queried.push(name);if(!artifacts[name])throw Error('not present');return artifacts[name]}});
  vm.runInContext(source, scope);
  await vm.runInContext('progress()', scope);
  return {text:elements.get('runStatus').textContent, queried};
}

(async()=>{
  let result = await run();
  assert.match(result.text, /final data audit are complete/);
  assert.match(result.text, /clinical form is not validated/);
  result = await run({'data_handoff_audit.json':null});
  assert.match(result.text, /final data\/QC audit is pending/);
  result = await run({'data_handoff_audit.json':{comparison_updated_at:'older-generation'}});
  assert.match(result.text, /final data\/QC audit is pending/);
  result = await run({'data_handoff_audit.json':{sampled_pose_reviews:218}});
  assert.match(result.text, /final data\/QC audit is pending/);
  result = await run({'postprocessing_status.json':{status:'running'}});
  assert.match(result.text, /metric export and comparison refresh are pending/);
  assert(!result.queried.includes('data_handoff_audit.json'));
  result = await run({report:{status:'interim'}});
  assert.match(result.text, /metric export and comparison refresh are pending/);
  result = await run({'production_status.json':{phase:'native_pose_inference', status:'running'}});
  assert.match(result.text, /^INTERIM RESULTS/);
  assert(!result.queried.includes('data_handoff_audit.json'));
  result = await run({'postprocessing_status.json':{status:'failed', message:'fixture failure'}});
  assert.match(result.text, /REPORT REFRESH STOPPED: fixture failure/);
  assert(!result.text.includes('final data audit are complete'));
  result = await run({'production_status.json':{status:'failed', message:'fixture failure'}});
  assert.match(result.text, /^Processing stopped: fixture failure/);
  console.log('Passed 9 report-stage scenarios using the actual progress function.');
})().catch(error=>{console.error(error);process.exitCode=1});
