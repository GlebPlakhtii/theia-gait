import * as THREE from './anatomy/vendor/three.module.js';
import {OrbitControls} from './anatomy/vendor/OrbitControls.js';
const $=id=>document.getElementById(id);
async function start(){
 const data=await fetch('anatomy/anatomy.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();});
 const container=$('anatomy-view'),renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));container.append(renderer.domElement);renderer.domElement.setAttribute('aria-label','Interactive OpenSim bones and 92 muscle paths. Drag to rotate, scroll to zoom.');renderer.domElement.setAttribute('role','img');
 const scene=new THREE.Scene();scene.background=new THREE.Color('#0b1520');const camera=new THREE.PerspectiveCamera(35,1,.01,30);camera.position.set(2.5,1.3,2);
 const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.92,0);controls.minDistance=1.1;controls.maxDistance=7;controls.enableDamping=true;
 scene.add(new THREE.HemisphereLight(0xddeeff,0x55525c,2.6));const light=new THREE.DirectionalLight(0xfff2df,3);light.position.set(3,5,4);scene.add(light);const fill=new THREE.DirectionalLight(0x7aaeff,2);fill.position.set(-2,1,-3);scene.add(fill);
 const grid=new THREE.GridHelper(2.8,14,0x355163,0x233747);grid.position.y=-.12;scene.add(grid);
 const bodies=new Map(),boneMat=new THREE.MeshStandardMaterial({color:0xe7ded0,roughness:.65,metalness:.05,side:THREE.DoubleSide});
 for(const m of data.meshes){let g=bodies.get(m.body);if(!g){g=new THREE.Group();g.matrixAutoUpdate=false;bodies.set(m.body,g);scene.add(g);}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(m.positions,3));geometry.setIndex(m.indices);geometry.computeVertexNormals();const mesh=new THREE.Mesh(geometry,boneMat);mesh.scale.fromArray(m.scale);g.add(mesh);}
 const pathGroup=new THREE.Group();scene.add(pathGroup);let pathMeshes=[],frame=0,playing=false,lastStep=0,direction=1;
 const choices=data.frames[0].muscles.map(m=>m.name);$('anatomy-muscle').innerHTML='<option value="all">All 92 muscle paths</option>'+choices.map(n=>`<option value="${n}">${n.replaceAll('_',' ')}</option>`).join('');
 const mats={normal:new THREE.MeshStandardMaterial({color:0xd34b58,roughness:.5}),selected:new THREE.MeshStandardMaterial({color:0xffd171,emissive:0x583600,roughness:.4}),muted:new THREE.MeshStandardMaterial({color:0x604251,transparent:true,opacity:.18,depthWrite:false})};
 function drawPose(){
  const f=data.frames[frame];for(const[name,g]of bodies){const a=f.transforms[name];g.matrix.set(a[0],a[1],a[2],a[9],a[3],a[4],a[5],a[10],a[6],a[7],a[8],a[11],0,0,0,1);g.visible=$('anatomy-bones').checked;}
  for(const m of pathMeshes){pathGroup.remove(m);m.geometry.dispose();}pathMeshes=[];
  const selected=$('anatomy-muscle').value;
  if($('anatomy-paths').checked)for(const m of f.muscles){const points=m.points.map(p=>new THREE.Vector3(...p));for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],delta=b.clone().sub(a),length=delta.length();if(length<1e-7)continue;const active=selected===m.name,mat=active?mats.selected:selected==='all'?mats.normal:mats.muted;const mesh=new THREE.Mesh(new THREE.CylinderGeometry(active?.007:.0045,active?.007:.0045,length,8),mat);mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());pathGroup.add(mesh);pathMeshes.push(mesh);}}
  $('anatomy-angle').value=f.knee_flexion_deg;$('anatomy-angle-value').textContent=`Knee ${f.knee_flexion_deg}° · hip ${f.hip_flexion_deg.toFixed(1)}° · ankle ${f.ankle_dorsiflexion_deg.toFixed(1)}°`;
  $('anatomy-selected').textContent=selected==='all'?'Red paths show model musculotendon routing; color does not indicate activation.':selected.replaceAll('_',' ')+' · model path length '+(f.muscles.find(m=>m.name===selected).length*100).toFixed(1)+' cm in this generic pose. Amber indicates selection only.';
 }
 $('anatomy-angle').oninput=e=>{playing=false;$('anatomy-play').textContent='Animate joint demo';frame=Math.round(Number(e.target.value)/2);drawPose();};
 for(const id of ['anatomy-bones','anatomy-paths','anatomy-muscle'])$(id).onchange=drawPose;
 $('anatomy-play').onclick=()=>{playing=!playing;$('anatomy-play').textContent=playing?'Pause joint demo':'Animate joint demo';};
 $('anatomy-reset').onclick=()=>{playing=false;frame=0;$('anatomy-camera').value='oblique';camera.position.set(2.5,1.3,2);controls.target.set(0,.92,0);$('anatomy-play').textContent='Animate joint demo';drawPose();};
 $('anatomy-camera').onchange=e=>{const poses={oblique:[2.5,1.3,2],front:[3,.95,0],side:[0,.95,3],back:[-3,.95,0]};camera.position.fromArray(poses[e.target.value]);controls.target.set(0,.92,0);controls.update();};
 new ResizeObserver(()=>{const w=container.clientWidth,h=container.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(container);
 let visible=false;new IntersectionObserver(entries=>visible=entries[0].isIntersecting).observe(container);
 function tick(t){requestAnimationFrame(tick);if(!visible)return;if(playing&&t-lastStep>90){frame+=direction;if(frame>=45){frame=45;direction=-1;}if(frame<=0){frame=0;direction=1;}lastStep=t;drawPose();}controls.update();renderer.render(scene,camera);}drawPose();requestAnimationFrame(tick);
 $('anatomy-status').textContent=`Official Gait2392 · ${data.body_count} bodies · ${data.coordinate_count} coordinates · ${data.muscle_count} musculotendon paths · ${data.meshes.length} bone meshes`;
}
start().catch(e=>{$('anatomy-status').textContent='3D view unavailable: '+e.message+'. The OpenSim model and exported geometry remain downloadable below.';console.error('Anatomy viewer',e);});
