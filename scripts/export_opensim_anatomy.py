"""Export official generic Gait2392 meshes and OpenSim-evaluated paths.
Illustrative prescribed joint poses only; no subject fit, force or activation solve.
"""
from pathlib import Path
import json,hashlib,math,xml.etree.ElementTree as E
import opensim as o
P=Path(__file__).resolve().parents[1]/'results/gait_experiments_20261115_0502/anatomy'
model=o.Model(str(P/'gait2392_thelen2003muscle.osim'));state=model.initSystem()
model.printToXML(str(P/'converted_model.osim'));root=E.parse(P/'converted_model.osim')
meshes=[]
for body in root.findall('.//BodySet/objects/Body'):
 for mesh in body.findall('./attached_geometry/Mesh'):
  filename=mesh.findtext('mesh_file');doc=E.parse(P/'Geometry'/filename)
  assert mesh.findtext('socket_frame')=='..'
  points=[float(v) for v in doc.find('.//Points/DataArray').text.split()]
  polys=doc.find('.//Polys');arrays={a.get('Name'):a for a in polys.findall('DataArray')}
  assert all(a.get('format')=='ascii' for a in [doc.find('.//Points/DataArray'),*arrays.values()])
  conn=[int(v) for v in arrays['connectivity'].text.split()];offsets=[int(v) for v in arrays['offsets'].text.split()];indices=[];start=0
  for end in offsets:
   poly=conn[start:end];start=end
   for i in range(1,len(poly)-1):indices.extend([poly[0],poly[i],poly[i+1]])
  meshes.append(dict(body=body.get('name'),file=filename,positions=points,indices=indices,scale=[float(v) for v in mesh.findtext('scale_factors').split()]))
vec=lambda v:[round(v[i],6) for i in range(3)]
frames=[];bodies=model.getBodySet();muscles=model.getMuscles()
for degree in range(0,91,2):
 coord=model.updCoordinateSet()
 for side in ['r','l']:
  coord.get('hip_flexion_'+side).setValue(state,math.radians(degree*.75),False)
  coord.get('knee_angle_'+side).setValue(state,-math.radians(degree),False)
  coord.get('ankle_angle_'+side).setValue(state,math.radians(degree*.22),False)
 model.realizePosition(state)
 transforms={}
 for i in range(bodies.getSize()):
  b=bodies.get(i);x=b.getTransformInGround(state);rot=x.R();t=x.p()
  transforms[b.getName()]=[round(rot.get(j,k),7) for j in range(3) for k in range(3)]+vec(t)
 paths=[]
 for i in range(muscles.getSize()):
  m=muscles.get(i);path=m.getGeometryPath().getCurrentPath(state)
  paths.append(dict(name=m.getName(),points=[vec(path.get(j).getLocationInGround(state)) for j in range(path.getSize())],length=round(m.getGeometryPath().getLength(state),6)))
 frames.append(dict(knee_flexion_deg=degree,hip_flexion_deg=degree*.75,ankle_dorsiflexion_deg=degree*.22,transforms=transforms,muscles=paths))
out=dict(model=model.getName(),coordinate_count=model.getCoordinateSet().getSize(),muscle_count=muscles.getSize(),body_count=bodies.getSize(),meshes=meshes,frames=frames,
 purpose='Generic anatomy and prescribed joint-angle demonstration. No participant/astronaut fit; no activation, force, injury or prosthetic adjustment estimation. Pelvis remains fixed; feet are not constrained to the floor.',
 credits=root.findtext('.//Model/credits'),source_repository='https://github.com/opensim-org/opensim-models',source_commit='d9b05d470b1a481c222372c85b75772faf8f7792')
(P/'anatomy.json').write_text(json.dumps(out,separators=(',',':')))
(P/'credits.txt').write_text(out['credits']+'\n\nChanges: exported bones to indexed meshes; evaluated muscle paths with OpenSim at prescribed symmetric joint angles; browser rendering uses schematic constant-radius tubes, not volumetric muscle anatomy.\n'+out['purpose'])
manifest=dict(model=out['model'],muscles=out['muscle_count'],meshes=len(meshes),poses=len(frames),source_commit=out['source_commit'],source_repository=out['source_repository'],source_model_path='Models/Gait2392_Simbody/gait2392_thelen2003muscle.osim',renderer=dict(package='three',version='0.180.0',distribution='https://cdn.jsdelivr.net/npm/three@0.180.0/'),clinical_validation=False,subject_fitted=False,activation_estimated=False,
 files={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and p.name!='manifest.json'})
(P/'manifest.json').write_text(json.dumps(manifest,indent=2));print({k:v for k,v in manifest.items() if k!='files'})
