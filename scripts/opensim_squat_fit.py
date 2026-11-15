"""Actual OpenSim inverse-kinematics fit to MotionBERT-estimated markers.

This is a nominal-scale articulated consistency model, not a subject-validated
musculoskeletal simulation. No masses, forces, torques or muscle activations
are interpreted. Estimated marker errors are not errors against motion capture.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import opensim as osim

from squat_form_pipeline import OUT, write, status, digest

MARKERS = {0:'pelvis',1:'right_hip',2:'right_knee',3:'right_ankle',4:'left_hip',5:'left_knee',6:'left_ankle',
           8:'thorax',11:'left_shoulder',14:'right_shoulder'}


def vec(a): return osim.Vec3(float(a[0]),float(a[1]),float(a[2]))
def body(name): return osim.Body(name,1.,osim.Vec3(0),osim.Inertia(1))
def marker(model,name,parent,location):model.addMarker(osim.Marker(name,parent,vec(location)))


def build_model(points,path):
    distance=lambda a,b:float(np.median(np.linalg.norm(points[:,a]-points[:,b],axis=1)))
    dimensions={'hip_width':distance(1,4),'shoulder_width':distance(11,14),'torso':distance(0,8)}
    model=osim.Model();model.setName('theia_nominal_kinematic_squat');model.setGravity(osim.Vec3(0))
    pelvis=body('pelvis_body');root=osim.FreeJoint('root',model.getGround(),osim.Vec3(0),osim.Vec3(0),pelvis,osim.Vec3(0),osim.Vec3(0))
    for i,name in enumerate(['root_rx','root_ry','root_rz','root_x','root_y','root_z']):root.upd_coordinates(i).setName(name)
    model.addBody(pelvis);model.addJoint(root);marker(model,'pelvis',pelvis,[0,0,0])
    for side,ids,sign in [('right',(1,2,3),-.5),('left',(4,5,6),.5)]:
        h,k,a=ids;hip=[0,0,sign*dimensions['hip_width']]
        thighlen,shanklen=distance(h,k),distance(k,a)
        dimensions[side+'_thigh']=thighlen;dimensions[side+'_shank']=shanklen
        thigh=body(side+'_thigh');shank=body(side+'_shank')
        hipjoint=osim.BallJoint(side+'_hip_joint',pelvis,vec(hip),osim.Vec3(0),thigh,osim.Vec3(0),osim.Vec3(0))
        for j in range(3):hipjoint.upd_coordinates(j).setName(side+'_hip_rotation_'+str(j))
        kneejoint=osim.PinJoint(side+'_knee_joint',thigh,vec([0,-thighlen,0]),osim.Vec3(0),shank,osim.Vec3(0),osim.Vec3(0))
        q=kneejoint.upd_coordinates(0);q.setName(side+'_knee_hinge');q.setDefaultValue(.1)
        q.setRangeMin(-math.pi+.05);q.setRangeMax(math.pi-.05);q.setDefaultClamped(True)
        model.addBody(thigh);model.addBody(shank);model.addJoint(hipjoint);model.addJoint(kneejoint)
        marker(model,side+'_hip',pelvis,hip);marker(model,side+'_knee',thigh,[0,-thighlen,0]);marker(model,side+'_ankle',shank,[0,-shanklen,0])
    trunk=body('trunk');back=osim.BallJoint('trunk_joint',pelvis,osim.Vec3(0),osim.Vec3(0),trunk,osim.Vec3(0),osim.Vec3(0))
    for j in range(3):back.upd_coordinates(j).setName('trunk_rotation_'+str(j))
    model.addBody(trunk);model.addJoint(back)
    marker(model,'thorax',trunk,[0,dimensions['torso'],0])
    marker(model,'left_shoulder',trunk,[0,dimensions['torso'],dimensions['shoulder_width']/2])
    marker(model,'right_shoulder',trunk,[0,dimensions['torso'],-dimensions['shoulder_width']/2])
    model.finalizeConnections();model.initSystem();model.printToXML(str(path))
    return model,dimensions


def fit_source(source):
    folder=OUT/'sources'/source['id'];lift=np.load(folder/'lifted.npz');inputs=np.load(folder/'inputs.npz')
    p=lift['world'].astype(float);t=inputs['time']-inputs['time'][0]
    output=folder/'opensim';output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return
    started=time.monotonic();model,dimensions=build_model(p,output/'model.osim')
    # TRC retains model-scale units explicitly. It is not a metric observation of this person.
    names=list(MARKERS.values());indices=list(MARKERS)
    table=osim.TimeSeriesTableVec3();labels=osim.StdVectorString()
    for name in names:labels.append(name)
    table.setColumnLabels(labels);table.addTableMetaDataString('Units','m')
    table.addTableMetaDataString('DataRate',str(source['fps']))
    for ti,row in zip(t,p):
        values=osim.RowVectorVec3(len(names))
        for j,index in enumerate(indices):values[j]=vec(row[index])
        table.appendRow(float(ti),values)
    trc=output/'estimated_markers.trc';osim.TRCFileAdapter.write(table,str(trc))
    tool=osim.InverseKinematicsTool();tool.setName(source['id']);tool.setModel(model)
    tool.set_marker_file(str(trc));tool.setStartTime(float(t[0]));tool.setEndTime(float(t[-1]))
    tool.set_accuracy(1e-5);tool.set_output_motion_file(str(output/'kinematics.mot'))
    tool.set_results_directory(str(output));tool.set_report_errors(True);tool.set_report_marker_locations(True)
    for name in names:
        task=osim.IKMarkerTask();task.setName(name);task.setWeight(.5 if 'shoulder' in name or name=='thorax' else 1.)
        tool.upd_IKTaskSet().cloneAndAppend(task)
    tool.printToXML(str(output/'ik_setup.xml'))
    if not tool.run():raise RuntimeError('OpenSim IK returned failure for '+source['id'])
    mot=osim.TimeSeriesTable(str(output/'kinematics.mot'));columns=list(mot.getColumnLabels());model=osim.Model(str(output/'model.osim'));state=model.initSystem()
    angles=np.array([[mot.getRowAtIndex(i)[j] for j in range(mot.getNumColumns())] for i in range(mot.getNumRows())])
    out_times=np.array(mot.getIndependentColumn());in_degrees=mot.hasTableMetaDataKey('inDegrees') and mot.getTableMetaDataAsString('inDegrees')=='yes'
    fitted=np.zeros((len(out_times),len(indices),3));coordinates=model.updCoordinateSet()
    for i,ti in enumerate(out_times):
        state.setTime(float(ti))
        for j,name in enumerate(columns):
            q=coordinates.get(name);v=angles[i,j]
            if in_degrees and q.getMotionType()==osim.Coordinate.Rotational:v=np.radians(v)
            q.setValue(state,float(v),False)
        model.realizePosition(state)
        for j,name in enumerate(names):
            v=model.getMarkerSet().get(name).getLocationInGround(state)
            fitted[i,j]=[v[k] for k in range(3)]
    assert len(out_times)==len(t) and np.allclose(out_times,t,atol=1e-5)
    residual=np.sqrt(((fitted-p[:,indices])**2).sum(-1))
    full=np.full_like(p,np.nan);full[:,indices]=fitted
    np.savez_compressed(output/'fit.npz',world=full,residual=residual,marker_indices=indices,time=t,
                        coordinates=angles,coordinate_names=np.array(columns))
    write(output/'complete.json',dict(at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        engine=osim.GetVersionAndDate(),frames=len(t),elapsed_s=time.monotonic()-started,
        marker_rms_fraction_leg=float(np.sqrt(np.mean(residual**2))),marker_median_fraction_leg=float(np.median(residual)),
        dimensions_nominal_metres=dimensions,raw_lift_sha256=digest(folder/'lifted.npz'),
        meaning='Actual OpenSim IK against estimated markers; errors measure fit consistency, not anatomical accuracy.',
        scale='One model metre equals the source median estimated hip-to-knee plus knee-to-ankle length; not measured height.',
        excluded_outputs=['joint loads','ground reaction forces','muscle activation','pain','injury risk'],
        model='Custom pelvis, bilateral ball hips and hinge knees, ball trunk; no ankle/foot or lumbar segment model.',
        file_sha256={q.name:digest(q) for q in output.iterdir() if q.is_file()}))
    status('opensim',source=source['id'],frames=len(t),marker_rms_fraction_leg=float(np.sqrt(np.mean(residual**2))),elapsed_s=round(time.monotonic()-started,2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source');args=parser.parse_args()
    osim.Logger.setLevelString('error')
    for source in json.loads((OUT/'sources.json').read_text()):
        if args.source and source['id']!=args.source:continue
        fit_source(source)
    status('opensim_complete')
