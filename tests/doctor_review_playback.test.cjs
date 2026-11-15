// Playback lifecycle regression tests, without changing any saved doctor labels.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const source=fs.readFileSync(process.env.REVIEW_PLAYER_SOURCE||'scripts/doctor_review.js','utf8');

function setup({frameAPI=true}={}){
 const elements=new Map(),listeners=new Map(),callbacks=new Map(),scripts=[];
 let nextFrame=0,clears=0,fills=0;
 const context2d={clearRect(){clears++;fills=0},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){fills++}};
 function element(id){
  if(!elements.has(id))elements.set(id,{value:'',textContent:'',classList:{toggle(){}},clientWidth:640,clientHeight:480,getContext:()=>context2d});
  return elements.get(id);
 }
 const video=element('video');
 Object.assign(video,{src:'a.mp4',currentTime:2,duration:8,paused:false,ended:false,loads:0,
  getAttribute(name){return this[name]},
  addEventListener(name,fn,options={}){if(!listeners.has(name))listeners.set(name,[]);listeners.get(name).push({fn,once:options.once})},
  removeEventListener(name,fn){listeners.set(name,(listeners.get(name)||[]).filter(x=>x.fn!==fn))},
  load(){this.loads++;this.currentTime=0;this.paused=true;callbacks.clear();emit('emptied')},
  play(){this.paused=false;emit('playing');return Promise.resolve()},
 });
 if(frameAPI){video.requestVideoFrameCallback=fn=>{callbacks.set(++nextFrame,fn);return nextFrame};video.cancelVideoFrameCallback=id=>callbacks.delete(id)}
 function emit(name){for(const item of [...(listeners.get(name)||[])]){if(item.once)video.removeEventListener(name,item.fn);item.fn()}}
 element('view').value='pose';element('speed').value='1';
 const clips=[{uid:'a',clip_id:'A',original_video:'a.mp4',pose_data:'a.js',source_start_s:0},{uid:'b',clip_id:'B',original_video:'b.mp4',pose_data:'b.js',source_start_s:8},{uid:'c',clip_id:'C',original_video:'c.mp4',pose_video:'c-pose.mp4',source_start_s:0}];
 const window={THEIA_CATALOG:{clips},addEventListener(){}};
 const sandbox=vm.createContext({window,document:{getElementById:element,createElement:()=>({remove(){}}),head:{appendChild:s=>scripts.push(s)}},console});
 vm.runInContext(source.slice(0,source.indexOf('function advance(')),sandbox);
 const run=code=>vm.runInContext(code,sandbox);
 run("active='a'");
 return {video,element,callbacks,scripts,window,emit,run,clears:()=>clears,fills:()=>fills};
}

test('canvas toggles preserve the playing video, time, and speed without reloading',()=>{
 const h=setup();h.video.playbackRate=.5;
 h.run('setVideo(true,true)');assert.equal(h.video.loads,0);assert.equal(h.video.currentTime,2);assert.equal(h.video.paused,false);assert.equal(h.video.playbackRate,.5);
 h.element('view').value='original';h.run('setVideo(true,true)');assert.equal(h.video.loads,0);assert.equal(h.video.paused,false);
});

test('changing prerecorded pose view restores time and resumes after metadata',()=>{
 const h=setup();h.run("active='c';setVideo(true,true)");assert.equal(h.video.src,'c-pose.mp4');assert.equal(h.video.loads,1);
 h.emit('loadedmetadata');assert.equal(h.video.currentTime,2);assert.equal(h.video.paused,false);assert.equal(h.callbacks.size,1);
});

test('new clip restarts a discarded frame callback and ignores the previous callback',()=>{
 const h=setup();h.run('startFrames()');const old=[...h.callbacks.values()][0];
 h.run("active='b';setVideo(true)");h.emit('loadedmetadata');h.emit('loadeddata');
 assert.equal(h.callbacks.size,1);const before=h.clears();old(0,{mediaTime:7});assert.equal(h.clears(),before);assert.equal(h.callbacks.size,1);
 const [id,callback]=[...h.callbacks][0];h.callbacks.delete(id);callback(0,{mediaTime:1.5});
 assert.match(h.element('phase').textContent,/Clip 1.50 s/);assert.equal(h.callbacks.size,1);
});

test('seeking while paused redraws and rearms exactly one callback; media events recover dropped callbacks',()=>{
 const h=setup();h.run('startFrames()');h.callbacks.clear();h.video.paused=true;h.video.currentTime=3;
 h.emit('timeupdate');assert.match(h.element('phase').textContent,/Clip 3.00 s/);
 for(let i=0;i<5;i++)h.emit('seeked');assert.equal(h.callbacks.size,1);
 h.video.currentTime=.2;h.emit('timeupdate');assert.match(h.element('phase').textContent,/Clip 0.20 s/);
});

test('browsers without frame callbacks redraw on media events',()=>{
 const h=setup({frameAPI:false});h.video.currentTime=4;h.emit('timeupdate');assert.match(h.element('phase').textContent,/Clip 4.00 s/);
 h.video.currentTime=1;h.emit('seeked');assert.match(h.element('phase').textContent,/Clip 1.00 s/);
});

test('stale pose loads and errors cannot overwrite a newer clip',()=>{
 const h=setup();h.run('setVideo(true,true)');const old=h.scripts[0];
 h.run("active='b';setVideo(true)");h.window.THEIA_POSE={clip_id:'A'};old.onload();assert.equal(h.run('pose'),null);
 h.element('status').textContent='current';old.onerror();assert.equal(h.element('status').textContent,'current');
 h.window.THEIA_POSE={clip_id:'B',frames:[],fps:30};h.scripts[1].onload();assert.equal(h.run('pose.clip_id'),'B');
});

test('expired pose data is hidden instead of freezing its last frame indefinitely',()=>{
 const h=setup();h.run("pose={fps:1,width:640,height:480,frames:[Array(69).fill(1)]};draw(0)");assert.equal(h.fills(),23);
 h.run('draw(4)');assert.match(h.element('phase').textContent,/Clip 4.00 s/);assert.equal(h.fills(),0);
});

test('a canvas toggle during metadata loading retains Next autoplay',()=>{
 const h=setup();h.run("active='b';setVideo(true)");h.element('view').value='original';h.run('setVideo(false,true)');
 h.emit('loadedmetadata');assert.equal(h.video.loads,1);assert.equal(h.video.paused,false);
});

test('rapid prerecorded pose toggles retain the pending playback time and intent',()=>{
 const h=setup();h.run("active='c';setVideo(true,true)");
 h.element('view').value='original';h.run('setVideo(false,true)');
 h.emit('loadedmetadata');assert.equal(h.video.src,'c.mp4');assert.equal(h.video.currentTime,2);assert.equal(h.video.paused,false);
});
