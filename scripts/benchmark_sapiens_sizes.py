"""Matched 1B/5B pose agreement and local latency, with cached 5B predictions.

Publisher RTMW-assisted, manually corrected annotations are a conditional
reference, not independent clinical ground truth. Same crops, frames, decoder,
BF16 and flip averaging are used for both sizes.
"""
import gc
import gzip
import hashlib
import json
import time
import argparse
from pathlib import Path

import av
import numpy as np
import torch

from run_sapiens2_pose import ROOT, MODEL_WEIGHTS, REVISION, frame_roi, infer, load_model, read_csv
from gait_geometry import bend

OUT = ROOT / 'results/gait_metrics_20261114/model_benchmark'
PREP = ROOT / 'data/prepared/progait_comparison_20261114'
BASE = ROOT / 'data/poses/sapiens2_5b_progait_comparison'
NAMES = ['nose','left_eye','right_eye','left_ear','right_ear',
         'left_shoulder','right_shoulder','left_elbow','right_elbow',
         'left_wrist','right_wrist','left_hip','right_hip','left_knee',
         'right_knee','left_ankle','right_ankle','left_big_toe',
         'left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']


def write(name, obj):
    path = OUT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def frames(row, indices, crops):
    wanted = set(map(int, indices))
    with av.open(row['local_path']) as con:
        for fi, frame in enumerate(con.decode(video=0)):
            if fi > max(wanted):
                break
            if fi in wanted:
                im = frame.to_ndarray(format='bgr24')
                h, w = im.shape[:2]
                box = np.array(frame_roi(crops, fi)) * [w,h,w,h]
                yield fi, im, box


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['1b','0.4b'],default='1b');args=parser.parse_args()
    variant_name=args.variant
    if variant_name!='1b':OUT=OUT/variant_name
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rows = read_csv(PREP/'queue.csv')
    rois = json.loads((PREP/'rois.json').read_text())['clips']
    ckpt = ROOT/f'models/sapiens2-{variant_name}/sapiens2_{variant_name}_pose.safetensors'
    assert hashlib.file_digest(ckpt.open('rb'),'sha256').hexdigest() == MODEL_WEIGHTS[variant_name]
    bundle = load_model(ckpt, variant_name)
    names = {v:int(k) for k,v in bundle[3]['keypoint_id2name'].items()}
    ix = [names[n] for n in NAMES]
    times = []; timed_examples = []; comparisons = []; diffs = {variant_name:[], '5b':[]}
    knee_errors={variant_name:[],'5b':[]};pixel_errors={variant_name:[],'5b':[]}
    side_knee_errors={variant_name:[],'5b':[]}
    write('protocol.json', dict(frames=180, sources=15, reference='Publisher RTMW-assisted and manually corrected annotations',
        crop_source='Publisher annotations; conditional comparison', vendor_revision=REVISION,
        settings='BF16, 1024x768, flip average, official UDP decoder, identical source frames and ROIs',
        limitations='Not independent manual accuracy, clinical validation, or an unseen-subject test.',
        gpu=torch.cuda.get_device_name(0), weights=MODEL_WEIGHTS))
    warmed = False
    for row in rows:
        cid = row['clip_id']; cached = np.load(BASE/cid/'raw_308.npz')
        points=[]; scores=[]; indices=[]
        for fi, im, box in frames(row, cached['source_frame_index'], rois[cid]):
            if not warmed:
                for _ in range(3): infer(im,box,bundle)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();warmed=True
            start=time.perf_counter(); p,s=infer(im,box,bundle);torch.cuda.synchronize()
            times.append(time.perf_counter()-start);points.append(p);scores.append(s);indices.append(fi)
            if len(indices) <= 2: timed_examples.append((row,fi,box))
        np.savez_compressed(OUT/(cid+'_'+variant_name+'.npz'), source_frame_index=indices,keypoints=points,scores=scores,
                            timestamp_s=cached['timestamp_s'])
        subset=cid.split('_')[1];ident=cid[len('PG_'+subset+'_'):]
        with gzip.open(ROOT/'data/raw/ProGait/annotations'/subset/(ident+'_keypoints.npy.gz'),'rb') as f:
            ref=np.load(f)[indices]
        with av.open(row['local_path']) as con:w,h=con.streams.video[0].width,con.streams.video[0].height
        valid=(ref[:,:,2]>=.3)&(ref[:,:,0]>=0)&(ref[:,:,0]<w)&(ref[:,:,1]>=0)&(ref[:,:,1]<h)
        diag=np.array([np.linalg.norm(x[g,:2].max(0)-x[g,:2].min(0)) for x,g in zip(ref,valid)])
        entry={'clip_id':cid,'frames':len(indices)}
        for variant,pred in [(variant_name,np.array(points)),('5b',cached['keypoints'])]:
            px=np.linalg.norm(pred[:,ix,:]-ref[:,:,:2],axis=-1)
            px[~valid]=np.nan;pixel_errors[variant].append(px)
            norm=px/diag[:,None]
            norm[~valid]=np.nan;diffs[variant].append(norm)
            for hip,knee,ankle in [(11,13,15),(12,14,16)]:
                mask=valid[:,[hip,knee,ankle]].all(1)
                p23=pred[:,ix,:]
                expected=bend(ref[:,hip,:2],ref[:,knee,:2],ref[:,ankle,:2])
                actual=bend(p23[:,hip],p23[:,knee],p23[:,ankle])
                errors=np.abs(actual-expected)[mask].tolist()
                knee_errors[variant].extend(errors)
                if '_s' in ident:side_knee_errors[variant].extend(errors)
            entry[variant]={'pck_005':float(np.nanmean((norm<=.05)[valid])),
                            'median_error_body_diagonals':float(np.nanmedian(norm))}
        comparisons.append(entry)
        print(f'{variant_name} {len(comparisons)}/15 clips; {len(times)} frames; mean {np.mean(times):.3f}s/frame',flush=True)
    peak1=torch.cuda.max_memory_allocated()
    del bundle;gc.collect();torch.cuda.empty_cache()
    # Fresh local 5B timings on the first two sampled frames of each source.
    bundle=load_model(ROOT/'models/sapiens2-5b/sapiens2_5b_pose.safetensors','5b')
    times5=[];warmed=False
    for row,fi,box in timed_examples:
        for _,im,_ in frames(row,[fi],rois[row['clip_id']]):
            if not warmed:
                for _ in range(3):infer(im,box,bundle)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();warmed=True
            start=time.perf_counter();infer(im,box,bundle);torch.cuda.synchronize()
            times5.append(time.perf_counter()-start)
    groups={'all_23':list(range(23)), 'knees':[13,14], 'ankles':[15,16], 'feet':[17,18,19,20,21,22]}
    result={'status':'complete','frames':len(times),'sources':len(rows),'comparisons':comparisons,'models':{}}
    for variant,tt,peak in [(variant_name,times,peak1),('5b',times5,torch.cuda.max_memory_allocated())]:
        err=np.concatenate(diffs[variant]);px=np.concatenate(pixel_errors[variant]);metrics={}
        for name,joints in groups.items():
            e=err[:,joints];e=e[np.isfinite(e)]
            metrics[name]={'points':len(e),'pck_005':float(np.mean(e<=.05)),
                'median_error_body_diagonals':float(np.median(e)),'p95_error_body_diagonals':float(np.percentile(e,95)),
                'median_pixel_distance':float(np.nanmedian(px[:,joints]))}
        result['models'][variant]={'timed_frames':len(tt),'mean_seconds_per_frame':float(np.mean(tt)),
            'median_seconds_per_frame':float(np.median(tt)), 'fps':float(1/np.mean(tt)),
            'peak_cuda_allocated_GB':peak/1e9,'agreement':metrics,
            'unsigned_projected_knee_angle_disagreement':dict(mean_absolute_degrees=float(np.mean(knee_errors[variant])),
                median_absolute_degrees=float(np.median(knee_errors[variant])),p95_absolute_degrees=float(np.percentile(knee_errors[variant],95)),
                scope='All sampled views; image-plane geometry only, not anatomical knee-angle accuracy'),
            'side_view_projected_knee_angle_disagreement':dict(mean_absolute_degrees=float(np.mean(side_knee_errors[variant])),
                median_absolute_degrees=float(np.median(side_knee_errors[variant])),p95_absolute_degrees=float(np.percentile(side_knee_errors[variant],95)),
                scope='Nominal side views only; publisher assisted reference, not anatomical ground truth'),
            'estimated_38232_frames_hours_inference_only':float(np.mean(tt)*38232/3600)}
    result['speedup_mean']=float(np.mean(times5)/np.mean(times))
    result['interpretation']='Conditional agreement with publisher annotations; no clinical correctness or prosthetic mechanical-joint validation.'
    write('results.json',result); print(json.dumps(result['models'],indent=2),flush=True)


if __name__=='__main__':main()
