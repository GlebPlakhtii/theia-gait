"""Native-clock, single-GPU Sapiens2 inference; local development footage only.
Adapted from Meta's pinned vis_pose.py preprocessing, flip test and UDP decoding.
The vendor source and model retain their Sapiens2 license.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '7e5bae88456ac418ff0e58e74106c9fe192055d4'
WEIGHT_SHA = 'b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52'
MODEL_WEIGHTS = {
    '5b': WEIGHT_SHA,
    '1b': '2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f',
    '0.4b': 'f1b7b0e5423d667634a355dd86a14c5b03d243f2ac77943301c7db9f93329106',
}
FIELDS = ('clip_id source_frame_index timestamp_s person_track_id joint_schema joint_id '
          'x y optional_z confidence coordinate_system model_name model_version preprocessing').split()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def save_progress(path, identity, indices, timestamps, points, scores):
    """Commit the entire raw prefix atomically; an interrupted save keeps the old one."""
    import numpy as np
    temporary = path.with_suffix('.npz.part')
    with temporary.open('wb') as f:
        np.savez_compressed(f, identity=json.dumps(identity, sort_keys=True),
                            source_frame_index=indices, timestamp_s=timestamps,
                            keypoints=points, scores=scores)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def load_progress(path, identity, eligible):
    import numpy as np
    if not path.exists():
        return [], [], [], []
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved['identity'])) != identity:
            raise ValueError('Checkpoint inputs/configuration changed; use a new output directory')
        indices = saved['source_frame_index'].copy()
        timestamps = saved['timestamp_s'].copy()
        points, scores = saved['keypoints'].copy(), saved['scores'].copy()
    n = len(indices)
    if n > len(eligible) or indices.tolist() != [int(r['source_frame_index']) for r in eligible[:n]]:
        raise ValueError('Checkpoint is not the expected native-frame prefix')
    if timestamps.shape != (n,) or not np.array_equal(timestamps, [float(r['timestamp_s']) for r in eligible[:n]]):
        raise ValueError('Checkpoint native timestamp mismatch')
    if points.shape != (n, 308, 2) or scores.shape != (n, 308):
        raise ValueError('Checkpoint tensor shape mismatch')
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError('Checkpoint contains nonfinite predictions')
    return indices.tolist(), timestamps.tolist(), list(points), list(scores)


def frame_roi(spec, source_frame_index):
    """Require explicit coverage when a reviewed per-frame crop map is supplied."""
    import math
    if 'bbox_by_source_frame' in spec:
        box = spec['bbox_by_source_frame'][str(source_frame_index)]
    else:
        box = spec['bbox_normalized_xyxy']
    if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError('Invalid normalized ROI')
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError('Empty or reversed ROI')
    return box


def write_pose_rows(writer, cid, fi, timestamp, kp, sc, selected, width, height, model_name='Sapiens2-5B'):
    import numpy as np
    for ji, name in selected.items():
        x, y = map(float, kp[ji]); valid = 0 <= x < width and 0 <= y < height
        writer.writerow(dict(clip_id=cid, source_frame_index=fi, timestamp_s=timestamp,
            person_track_id='foreground_1', joint_schema='sapiens2_body_feet23_named', joint_id=name,
            x=f'{x:.6f}' if valid else '', y=f'{y:.6f}' if valid else '', optional_z='',
            confidence=float(np.clip(sc[ji], 0, 1)) if valid else 0, coordinate_system='pixel',
            model_name=model_name, model_version=REVISION,
            preprocessing='foreground_ROI;official_UDP_affine_768x1024;RGB_ImageNet;BF16;flip_average;native_PTS'))


def load_model(checkpoint, variant='5b'):
    import torch
    import sapiens.pose
    from sapiens.registry import MODELS
    from sapiens.engine.config import Config
    from sapiens.engine.datasets import Compose
    from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
    from safetensors.torch import load_file
    base = ROOT / 'vendor/sapiens2/sapiens/pose'
    if variant not in MODEL_WEIGHTS:
        raise ValueError(f'Unsupported model variant: {variant}')
    cfg = Config.fromfile(str(base / f'configs/keypoints308/shutterstock_goliath_3po/sapiens2_{variant}_keypoints308_shutterstock_goliath_3po-1024x768.py'))
    cfg.model['backbone'].pop('init_cfg', None)
    with torch.device('meta'):
        model = MODELS.build(cfg.model)
    state = load_file(str(checkpoint), device='cpu')
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Convert on CPU before moving: FP32 weights alone exceed laptop VRAM.
    model = model.to(dtype=torch.bfloat16).to('cuda').eval()
    prep = MODELS.build(cfg.data_preprocessor)  # Keep normalization FP32 on CPU.
    pipeline = Compose(cfg.test_pipeline)
    meta = parse_pose_metainfo(dict(from_file=str(base / 'configs/_base_/keypoints308.py')))
    codec_args = dict(cfg.codec)
    assert codec_args.pop('type') == 'UDPHeatmap'
    codec = UDPHeatmap(**codec_args)
    return model, prep, pipeline, meta, codec


def infer(image, bbox, bundle):
    import numpy as np
    import torch
    model, prep, pipeline, meta, codec = bundle
    data = prep(pipeline(dict(img=image, bbox=np.array([bbox], dtype=np.float32),
                              bbox_score=np.ones(1, dtype=np.float32))))
    inputs = data['inputs'].to('cuda', dtype=torch.bfloat16)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(inputs)
        flipped = model(inputs.flip(-1)).flip(-1)[:, meta['flip_indices']]
        heatmaps = ((pred.float() + flipped.float()) * 0.5).cpu().numpy()[0]
    kpts, scores = codec.decode(heatmaps)
    geom = data['data_samples']['meta']
    kpts = kpts / geom['input_size'] * geom['bbox_scale'] + geom['bbox_center'] - 0.5 * geom['bbox_scale']
    return kpts[0], scores[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', default=str(ROOT / 'data/manifests/pose_queue_ready.csv'))
    p.add_argument('--rois', default=str(ROOT / 'data/manifests/pose_rois.json'))
    p.add_argument('--model-variant', choices=sorted(MODEL_WEIGHTS), default='5b')
    p.add_argument('--checkpoint', help='Defaults to the verified checkpoint for --model-variant')
    p.add_argument('--output', required=True)
    p.add_argument('--pilot', type=int, default=0, help='Seeded stratified frames per clip, 0 = every native frame')
    p.add_argument('--clip', action='append', help='Optional exact clip IDs')
    p.add_argument('--checkpoint-every', type=int, default=25, help='Persist raw predictions every N frames')
    a = p.parse_args()
    if a.checkpoint_every < 1:
        raise ValueError('Checkpoint interval must be positive')
    import av
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(8)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    # One writer per output tree, including when another chat reuses this queue.
    import fcntl
    output_lock = (out / '.inference.lock').open('a+')
    try:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another inference process is writing {out}; reuse its live run instead of starting a duplicate')
    output_lock.seek(0);output_lock.truncate();output_lock.write(str(os.getpid()));output_lock.flush()
    queue = [q for q in read_csv(a.queue) if not a.clip or q['clip_id'] in a.clip]
    rois = json.loads(Path(a.rois).read_text())
    weight_sha = MODEL_WEIGHTS[a.model_variant]
    model_name = 'Sapiens2-' + a.model_variant.upper()
    config_id = hashlib.sha256((Path(a.rois).read_text()+str(a.pilot)+REVISION+weight_sha).encode()).hexdigest()
    ckpt = Path(a.checkpoint) if a.checkpoint else ROOT / f'models/sapiens2-{a.model_variant}/sapiens2_{a.model_variant}_pose.safetensors'
    assert hashlib.file_digest(ckpt.open('rb'), 'sha256').hexdigest() == weight_sha
    start = time.monotonic()
    bundle = load_model(ckpt, a.model_variant)
    meta = bundle[3]
    names = {int(k): v for k, v in meta['keypoint_id2name'].items()}
    # Export named body/feet landmarks only; preserve all raw predictions in NPZ.
    selected = {i: n for i, n in names.items() if n in {
        'nose','left_eye','right_eye','left_ear','right_ear',
        *[f'{side}_{joint}' for side in ('left','right') for joint in
          ('shoulder','elbow','wrist','hip','knee','ankle','big_toe','small_toe','heel')]}}
    assert len(selected) == 23, selected
    (out / 'joint_schema.json').write_text(json.dumps({'all_308': names, 'body_feet_23': selected}, indent=2))
    summary = {'model': model_name, 'vendor_revision': REVISION, 'weight_sha256': weight_sha,
               'precision': 'BF16', 'resolution_hw': [1024,768], 'flip_test': True,
               'pilot_frames_per_clip': a.pilot, 'coordinate_system': 'pixel',
               'track_method': rois.get('provenance') or ('manual per-frame foreground ROIs; no identity recognition' if any('bbox_by_source_frame' in rois['clips'][q['clip_id']] for q in queue) else 'manual fixed foreground ROI; no identity recognition'),
               'roi_sha256': hashlib.sha256(Path(a.rois).read_bytes()).hexdigest(),
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'checkpoint_every_frames': a.checkpoint_every, 'clips': []}
    for q in queue:
        cid = q['clip_id']
        dest = out / cid
        if (dest / 'complete.json').exists():
            assert json.loads((dest / 'complete.json').read_text())['config_id'] == config_id, 'Output configuration changed; use a new directory'
            print('Already complete:', cid, flush=True)
            summary['clips'].append(json.loads((dest / 'complete.json').read_text()))
            continue
        dest.mkdir(exist_ok=True)
        assert hashlib.file_digest(open(q['local_path'],'rb'),'sha256').hexdigest() == q['sha256']
        fmap = read_csv(q['frame_map'])
        eligible = [r for r in fmap if float(q['start_s']) <= float(r['timestamp_s']) < float(q['end_s'])]
        if a.pilot:
            bins = np.array_split(np.arange(len(eligible)), min(a.pilot,len(eligible)))
            rng = np.random.default_rng(20261113)
            picks = [int(rng.choice(b)) for b in bins]
            picks[0], picks[-1] = 0, len(eligible)-1
            eligible = [eligible[i] for i in picks]
        wanted = {int(r['source_frame_index']): r for r in eligible}
        if not wanted:
            raise ValueError(f'Empty selected interval: {cid}')
        crop_spec = rois['clips'][cid]
        frame_rois = {fi: frame_roi(crop_spec, fi) for fi in wanted}
        identity = dict(version=1, config_id=config_id, queue_row=q,
                        frame_map_sha256=hashlib.sha256(Path(q['frame_map']).read_bytes()).hexdigest(),
                        inference_implementation='pinned_sapiens2_UDP_BF16_flip_v1')
        progress_path = dest / 'checkpoint_308.npz'
        indices, timestamps, points, scores_out = load_progress(progress_path, identity, eligible)
        resumed_frames = len(indices)
        if resumed_frames:
            print(f'Resuming {cid}: {resumed_frames} raw native frames restored', flush=True)
        clipstart = time.monotonic()
        preview_indices = set(np.linspace(0,len(eligible)-1,min(6,len(eligible)),dtype=int))
        with open(dest / 'poses.csv.part','w',newline='') as f, av.open(q['local_path']) as container:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            stream = container.streams.video[0]
            for fi, ts, kp, sc in zip(indices, timestamps, points, scores_out):
                write_pose_rows(writer, cid, int(fi), wanted[int(fi)]['timestamp_s'], kp, sc, selected, stream.width, stream.height, model_name)
            for fi, frame in enumerate(container.decode(video=0)):
                if fi > max(wanted):
                    break
                if fi not in wanted:
                    continue
                if resumed_frames and fi <= indices[resumed_frames - 1]:
                    continue
                r = wanted[fi]
                if abs(float(frame.pts * frame.time_base) - float(r['source_pts_s'])) > 1e-5:
                    raise ValueError(f'Native PTS mismatch: {cid} frame {fi}')
                image = frame.to_ndarray(format='bgr24')
                h,w = image.shape[:2]
                bbox = np.array(frame_rois[fi]) * np.array([w,h,w,h])
                t0=time.monotonic()
                kp, sc = infer(image, bbox, bundle)
                if not np.isfinite(kp).all() or not np.isfinite(sc).all():
                    raise ValueError('Nonfinite model output')
                indices.append(fi); timestamps.append(float(r['timestamp_s'])); points.append(kp); scores_out.append(sc)
                write_pose_rows(writer, cid, fi, r['timestamp_s'], kp, sc, selected, w, h, model_name)
                if len(indices) % a.checkpoint_every == 0:
                    save_progress(progress_path, identity, indices, timestamps, points, scores_out)
                    f.flush()
                if len(indices)-1 in preview_indices:
                    vis=image.copy()
                    for ji,name in selected.items():
                        if sc[ji]>=0.3:
                            xy=tuple(np.rint(kp[ji]).astype(int)); cv2.circle(vis,xy,4,(0,255,255),-1)
                    for side,color in [('left',(0,255,0)),('right',(0,128,255))]:
                        for j1,j2 in [('shoulder','elbow'),('elbow','wrist'),('shoulder','hip'),('hip','knee'),('knee','ankle'),('ankle','big_toe')]:
                            i1=next(i for i,n in selected.items() if n==f'{side}_{j1}')
                            i2=next(i for i,n in selected.items() if n==f'{side}_{j2}')
                            if min(sc[i1],sc[i2])>=.3:
                                cv2.line(vis,tuple(np.rint(kp[i1]).astype(int)),tuple(np.rint(kp[i2]).astype(int)),color,2)
                    cv2.imwrite(str(dest/f'preview_{fi:06d}.jpg'),vis)
                if a.pilot or len(indices)%25==0:
                    print(f'{cid} {len(indices)}/{len(eligible)} last={time.monotonic()-t0:.2f}s GPU_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB',flush=True)
            assert len(indices)==len(eligible),(cid,len(indices),len(eligible))
        save_progress(progress_path, identity, indices, timestamps, points, scores_out)
        with (dest/'raw_308.npz.part').open('wb') as f:
            np.savez_compressed(f,source_frame_index=indices,timestamp_s=timestamps,keypoints=points,scores=scores_out)
        (dest/'raw_308.npz.part').replace(dest/'raw_308.npz')
        (dest/'poses.csv.part').rename(dest/'poses.csv')
        done=dict(clip_id=cid,config_id=config_id,frames=len(indices),elapsed_s=time.monotonic()-clipstart,
                  source_sha256=q['sha256'],source_frame_first=indices[0],source_frame_last=indices[-1],
                  pose_csv_sha256=hashlib.sha256((dest/'poses.csv').read_bytes()).hexdigest(),
                  visibility_review='pending; heatmap score is not calibrated visibility',
                  resumed_raw_frames=resumed_frames,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  raw_npz_sha256=hashlib.sha256((dest/'raw_308.npz').read_bytes()).hexdigest())
        (dest/'complete.json').write_text(json.dumps(done,indent=2)); summary['clips'].append(done)
        (out/'run.json').write_text(json.dumps(summary,indent=2))
    summary.update(elapsed_s=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (out/'run.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
