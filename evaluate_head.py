"""Evaluate a trained TAP-style prediction head on extracted SVD features.

This evaluator consumes the .pt schema written by extract_features.py:
  feat, query_points, target_points, occluded, video, meta

Usage:
python evaluate_head.py \
    --feat_dir /content/output/davis \
    --output_dir /content/result/head4 \
    --head_ckpt checkpoints/head_train/head_step_001200.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils import compute_tapvid_metrics
from train_head import (
    HEAD_KWARGS,
    TAPStylePredictionHead,
    load_head_checkpoint,
    predict_head,
)


def load_from_pt(pt_path) -> dict:
    """Load extracted features and metadata from a .pt file."""
    return torch.load(pt_path, map_location='cpu')


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def squeeze_batch(value: np.ndarray) -> np.ndarray:
    if value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def get_image_size(data: dict, meta: dict) -> tuple[int, int]:
    if 'video' in data:
        video = data['video']
        return int(video.shape[1]), int(video.shape[2])
    if 'eval_size' not in meta:
        raise ValueError('Feature file has no video and no meta["eval_size"].')
    eval_size = int(meta['eval_size'])
    return eval_size, eval_size


def evaluate_video(
    pt_path,
    head: TAPStylePredictionHead,
    device: torch.device,
    query_chunk_size: int,
) -> tuple[dict, dict]:
    data = load_from_pt(pt_path)
    meta = data.get('meta', {})
    vid_name = meta.get('vid_name', Path(pt_path).stem)

    feat = data['feat'].to(device=device, dtype=torch.float32)
    query_points_arr = squeeze_batch(as_numpy(data['query_points'])).astype(np.float32)
    gt_tracks_query = squeeze_batch(as_numpy(data['target_points'])).astype(np.float32)
    occluded_query = squeeze_batch(as_numpy(data['occluded'])).astype(np.bool_)
    img_H, img_W = get_image_size(data, meta)
    if 'query_mode' not in meta:
        raise ValueError(f'{vid_name}: feature metadata has no query_mode.')
    query_mode = meta['query_mode']

    if feat.shape[0] != gt_tracks_query.shape[1]:
        raise ValueError(
            f'{vid_name}: feature frames {feat.shape[0]} != target frames '
            f'{gt_tracks_query.shape[1]}'
        )
    if feat.shape[0] != occluded_query.shape[1]:
        raise ValueError(
            f'{vid_name}: feature frames {feat.shape[0]} != occlusion frames '
            f'{occluded_query.shape[1]}'
        )
    if query_points_arr.shape[0] != gt_tracks_query.shape[0]:
        raise ValueError(
            f'{vid_name}: queries {query_points_arr.shape[0]} != target tracks '
            f'{gt_tracks_query.shape[0]}'
        )
    if query_points_arr.shape[0] != occluded_query.shape[0]:
        raise ValueError(
            f'{vid_name}: queries {query_points_arr.shape[0]} != occlusion tracks '
            f'{occluded_query.shape[0]}'
        )

    pred_tracks_arr, pred_occ_arr = predict_head(
        head=head,
        spatial_feat=feat,
        query_points_arr=query_points_arr,
        img_H=img_H,
        img_W=img_W,
        query_chunk_size=query_chunk_size,
    )

    metrics = compute_tapvid_metrics(
        query_points=query_points_arr[np.newaxis],
        gt_occluded=occluded_query[np.newaxis],
        gt_tracks=gt_tracks_query[np.newaxis],
        pred_occluded=pred_occ_arr[np.newaxis],
        pred_tracks=pred_tracks_arr[np.newaxis],
        query_mode=query_mode,
    )
    scalar_metrics = {key: float(value[0]) for key, value in metrics.items()}
    return scalar_metrics, meta


def main(args) -> None:
    feat_dir = Path(args.feat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(feat_dir.glob('*.pt'))
    if args.max_videos > 0:
        pt_files = pt_files[:args.max_videos]
    if not pt_files:
        raise FileNotFoundError(f'No .pt files found in {feat_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    head = TAPStylePredictionHead(**HEAD_KWARGS).to(device)
    load_head_checkpoint(head, args.head_ckpt, device)
    head.eval()

    print(f'Using device: {device}')
    print(f'Loaded prediction head from {args.head_ckpt}')
    print(f'Found {len(pt_files)} videos in {feat_dir}')

    per_video_metrics = {}
    failed_videos = {}
    feat_meta = {}

    for pt_path in tqdm(pt_files, desc='Videos'):
        vid_name = pt_path.stem
        try:
            metrics, meta = evaluate_video(
                pt_path=pt_path,
                head=head,
                device=device,
                query_chunk_size=args.query_chunk_size,
            )
            per_video_metrics[vid_name] = metrics
            feat_meta = meta
            tqdm.write(
                f"  {vid_name}: AJ={metrics['average_jaccard']:.4f}  "
                f"<delta={metrics['average_pts_within_thresh']:.4f}  "
                f"OA={metrics['occlusion_accuracy']:.4f}"
            )
        except Exception as exc:
            tqdm.write(f'  ERROR on {vid_name}: {exc}')
            failed_videos[vid_name] = str(exc)

    num_samples = len(per_video_metrics)
    if num_samples == 0:
        raise RuntimeError('No videos were evaluated successfully.')

    all_keys = list(next(iter(per_video_metrics.values())).keys())
    mean_metrics = {
        key: float(np.mean([per_video_metrics[name][key] for name in per_video_metrics]))
        for key in all_keys
    }

    results = {
        'per_video': per_video_metrics,
        'mean': mean_metrics,
        'config': {
            'feat_dir': str(feat_dir),
            'head_ckpt': args.head_ckpt,
            'query_chunk_size': args.query_chunk_size,
            'head_kwargs': HEAD_KWARGS,
            'num_samples': num_samples,
            'num_pt_files': len(pt_files),
            'num_failed': len(failed_videos),
            'failed_videos': failed_videos,
            'eval_size': feat_meta.get('eval_size'),
            'query_mode': feat_meta.get('query_mode'),
        },
    }

    out_json = output_dir / 'metrics.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\n===== Mean metrics across all videos =====')
    for key, value in mean_metrics.items():
        print(f'  {key}: {value:.4f}')
    print(f'\nSaved to: {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a TAP-style prediction head on extracted features.'
    )
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Directory containing extracted .pt feature files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for metrics.json.')
    parser.add_argument('--head_ckpt', type=str, required=True,
                        help='Path to a trained TAP-style prediction head checkpoint.')
    parser.add_argument('--query_chunk_size', type=int, default=32,
                        help='Number of queries processed per head prediction chunk.')
    parser.add_argument('--max_videos', type=int, default=-1,
                        help='Limit evaluated .pt files; <=0 uses all.')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
