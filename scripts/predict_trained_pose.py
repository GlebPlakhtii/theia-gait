"""Run the saved three-seed Theia ensemble on an existing pose track."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from train_existing_data import PoseTCN, load_csv_pose, normalize_pose, windows, NAMES, FPS, LENGTH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Named-joint pose CSV or ProGait _keypoints.npy.gz')
    parser.add_argument('--task', choices=['gait', 'squat'], required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True, help='Repeat for all three seed checkpoints')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publisher-fps', type=float, default=30., help='Only used for publisher NPY input')
    parser.add_argument('--width', type=int, help='Original RGB width; required for publisher NPY')
    parser.add_argument('--height', type=int, help='Original RGB height; required for publisher NPY')
    args = parser.parse_args()
    if args.input.suffix == '.csv':
        t, pose = load_csv_pose(args.input)
        bounds = None
    else:
        if not args.width or not args.height:
            parser.error('Publisher NPY input requires --width and --height to apply the training image-bound mask')
        with gzip.open(args.input, 'rb') as f:
            pose = np.load(f, allow_pickle=False)
        if pose.ndim != 3 or pose.shape[1:] != (23, 3):
            raise ValueError('Expected a publisher pose array with shape [frames,23,3]')
        t = np.arange(len(pose)) / args.publisher_fps
        bounds = (args.width, args.height)
    a, quality = normalize_pose(t, pose, bounds)
    w, starts = windows(a)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.from_numpy(w).to(device)
    seed_probs = []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = PoseTCN().to(device).eval()
        model.load_state_dict(checkpoint['state_dict'])
        pieces = []
        with torch.no_grad():
            for offset in range(0, len(x), 128):
                pieces.append(model(x[offset:offset + 128], args.task).softmax(-1).cpu().numpy())
        seed_probs.append(np.concatenate(pieces))
    p = np.mean(seed_probs, axis=0)
    mean = p.mean(0)
    result = dict(task=args.task, source=str(args.input.resolve()), device=device, quality=quality,
        class_map=checkpoint['label_maps'][args.task], predicted_index=int(mean.argmax()), probabilities=mean.tolist(),
        checkpoints=[str(q.resolve()) for q in args.checkpoint],
        interpretation='Uncalibrated research scores for the selected source task; no ARED form validation or autonomous exercise instruction.',
        operational_decision='abstain',
        windows=[dict(start_s=float(t[0] + s / FPS),
                      end_s_exclusive=float(min(t[-1] + 1 / FPS, t[0] + (s + LENGTH) / FPS)),
                      probabilities=row.tolist(), predicted_index=int(row.argmax())) for s, row in zip(starts, p)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Saved {len(p)} window predictions to {args.output}; operational decision: abstain')


if __name__ == '__main__':
    main()
