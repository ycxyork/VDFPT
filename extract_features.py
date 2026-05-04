"""
Extract SVD features and Get queries for TAP-Vid datasets.

Usage:
python extract_features.py \
    --data_path data/tapvid_davis/tapvid_davis.pkl \
    --output_dir /content/output/davis
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import logging

from data_utils import (
    create_kubric_eval_train_dataset,
    create_kubric_eval_dataset,
    create_davis_dataset,
    create_kinetics_dataset,
    create_rgb_stacking_dataset,
    resize_video,
)
import feature_svd
feature_svd.VERBOSE = False


# ============================================================
# Dataset / preprocessing helpers
# ============================================================

def create_feature_dataset(
    data_path: str,
    query_mode: str,
    resolution: int,
    max_videos: int,
    kubric_shuffle_buffer: int,
):
    """Create a local TAP-Vid iterator for feature extraction.

    return dict:
      video:         (1, T, H, W, 3), float32 in [-1, 1]
      query_points:  (1, N, 3), where each point is [t, y, x] pixel coordinates
      target_points: (1, N, T, 2), pixel coordinates in [x, y]
      occluded:      (1, N, T), bool
    """
    data_path_obj = Path(data_path)
    data_str = str(data_path_obj).lower()
    output_size = (resolution, resolution)

    if 'kubric' in data_str:
        dataset_name = 'kubric'
        dataset_iter = create_kubric_eval_dataset(
            mode='',
            train_size=output_size,
        )
        return dataset_name, (sample[dataset_name] for sample in dataset_iter)
    if 'kinetics' in data_str:
        dataset_name = 'kinetics'
        dataset_iter = create_kinetics_dataset(
            str(data_path_obj), query_mode=query_mode, resolution=output_size,
        )
        return dataset_name, (sample[dataset_name] for sample in dataset_iter)
    if 'rgb' in data_str:
        dataset_name = 'robotics'
        dataset_iter = create_rgb_stacking_dataset(
            str(data_path_obj), query_mode=query_mode, resolution=output_size,
        )
        return dataset_name, (sample[dataset_name] for sample in dataset_iter)
    if 'davis' in data_str:
        dataset_name = 'davis'
        dataset_iter = create_davis_dataset(
            str(data_path_obj), query_mode=query_mode, resolution=output_size,
        )
        return dataset_name, (sample[dataset_name] for sample in dataset_iter)
    raise ValueError(f'Cannot infer dataset from data_path: {data_path}')


def prepare_video_tensor(video: np.ndarray) -> torch.Tensor:
    """Convert (T, H, W, 3) float video in [-1, 1] to (T, 3, H, W)."""
    if video.ndim != 4:
        raise ValueError(f'Expected video shape (T, H, W, 3), got {video.shape}')
    return torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2).contiguous()


def normalize_video_for_svd(video: np.ndarray) -> np.ndarray:
    """Return float32 video in [-1, 1] for SVD feature extraction."""
    video = video.astype(np.float32, copy=False)
    video_min = float(np.min(video))
    video_max = float(np.max(video))
    if video_min >= -1.0 and video_max <= 1.0 and video_min < 0.0:
        return video
    if video_min >= 0.0 and video_max <= 1.0:
        return video * 2.0 - 1.0
    return video / 255.0 * 2.0 - 1.0


def delete_query(
    query_points: np.ndarray,
    target_points: np.ndarray,
    occluded: np.ndarray,
):
    """Remove queries whose query frame is outside the clipped sequence."""
    num_frames = target_points.shape[1]
    valid = query_points[:, 0] < num_frames
    return query_points[valid], target_points[valid], occluded[valid]


def resize_for_eval(
    video: np.ndarray,
    query_points: np.ndarray,
    target_points: np.ndarray,
    eval_size: int,
):
    """Resize saved video and point coordinates to eval_size."""
    height, width = video.shape[1:3]
    if (height, width) == (eval_size, eval_size):
        return video, query_points, target_points

    scale_x = eval_size / width
    scale_y = eval_size / height

    video = resize_video(video, (eval_size, eval_size))
    query_points = query_points.copy()
    query_points[:, 1] *= scale_y
    query_points[:, 2] *= scale_x

    target_points = target_points.copy()
    target_points[..., 0] *= scale_x
    target_points[..., 1] *= scale_y
    return video, query_points, target_points


# ============================================================
# Main extraction
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load dataset ----
    dataset_name, dataset_iter = create_feature_dataset(
        args.data_path,
        query_mode=args.query_mode,
        resolution=args.resolution,
        max_videos=args.max_videos,
        kubric_shuffle_buffer=args.kubric_shuffle_buffer,
    )
    print(
        f"Using {dataset_name} dataset from {args.data_path}\n"
        f"Query mode: {args.query_mode}. Input resolution={args.resolution}. Eval size={args.eval_size} "
    )

    # ---- init featurizer (once for all videos) ----
    print("Initializing SVDFeaturizer ...")
    featurizer = feature_svd.SVDFeaturizer(
        svd_id=args.svd_id,
    )

    # ---- iterate videos ----
    for video_idx, sample in enumerate(tqdm(dataset_iter, desc="Extracting features")):
        vid_name = f"{dataset_name}_{video_idx:04d}"
        out_pt = output_dir / f"{vid_name}.pt"

        if out_pt.exists() and not args.overwrite:
            print(f"  Skip {vid_name} already exists")
            continue

        video_np = normalize_video_for_svd(sample['video'][0])  # (T, H, W, 3), [-1, 1]
        query_points = sample['query_points'][0]  # (N, 3)
        target_points = sample['target_points'][0]  # (N, T, 2)
        occluded = sample['occluded'][0]  # (N, T)

        T_total = video_np.shape[0]
        if not args.track_all:
            segments = [(0, min(T_total, args.max_frames))]
        else:
            segments = []
            start = 0
            while start < T_total:
                end = min(start + args.max_frames, T_total)
                segments.append((start, end))
                start = end
        T_used = segments[-1][1]
        tqdm.write(f"{vid_name}: {T_total} frames || "
              f"{'sliding' if args.track_all else 'truncate'} mode: "
              f"{len(segments)} segments {T_used} frames total")

        # ---- extract features for each segment----
        feat_segments = []
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_len = seg_end - seg_start
            video_segment = video_np[seg_start:seg_end]  # (seg_len, H, W, 3)
            video_tensor = prepare_video_tensor(video_segment)

            with torch.no_grad():
                seg_feat = featurizer.forward(
                    video_tensor=video_tensor.cuda(),
                    sigma=args.sigma,
                    up_ft_index=args.up_ft_block,
                    ensemble_size=args.ensemble_size,
                    num_frames=seg_len,
                )  # (seg_len, C, H', W')

            feat_segments.append(seg_feat.cpu())
            tqdm.write(f"    segment [{seg_start}:{seg_end}]  feat={tuple(seg_feat.shape)}")

            del seg_feat, video_tensor
            torch.cuda.empty_cache()
            gc.collect()

        # ---- concatenate all segments ----
        spatial_feat = torch.cat(feat_segments, dim=0)  # (T_used, C, H', W')
        assert spatial_feat.shape[0] == T_used, \
            f"Expected {T_used} frames, got {spatial_feat.shape[0]}"

        # ---- delete queries that exceed the max_frames ----
        query_points_save, target_points_save, occluded_save = delete_query(
            query_points,
            target_points[:, :T_used],
            occluded[:, :T_used],
        )
        video_save, query_points_save, target_points_save = resize_for_eval(
            video_np[:T_used],
            query_points_save,
            target_points_save,
            args.eval_size,
        )

        # ---- save everything needed for matching and evaluation----
        torch.save({
            'feat':          spatial_feat.cpu(),             # (F, C, H', W')
            'query_points':  torch.from_numpy(query_points_save), # (N, 3)
            'target_points': torch.from_numpy(target_points_save), # (N, F, 2)
            'occluded':      torch.from_numpy(occluded_save),
            'video':         torch.from_numpy(video_save),
            'meta': {
                'dataset_name':  dataset_name,
                'vid_name':      vid_name,
                'num_frames':    T_used,
                'feat_shape':    tuple(spatial_feat.shape),
                'input_size':    args.resolution,
                'eval_size':     args.eval_size,
                'query_mode':    args.query_mode,
                'up_ft_block':   args.up_ft_block,
                'sigma':         args.sigma,
                'cin':           args.cin,
                'ensemble_size': args.ensemble_size,
            },
        }, str(out_pt))

        print(f"  Saved: {out_pt}  feat={tuple(spatial_feat.shape)}")

        del spatial_feat
        gc.collect()

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract SVD features for TAP-Vid datasets.'
    )
    parser.add_argument('--data_path',     type=str,   required=True,
                        help='Path to a TAP-Vid pkl, or Kinetics shard directory')
    parser.add_argument('--output_dir',    type=str,   default='output/')
    
    parser.add_argument('--svd_id',        type=str,
                        default='stabilityai/stable-video-diffusion-img2vid-xt')
    parser.add_argument('--up_ft_block',   type=int,   default=2,
                        help='Up-block index to extract features from (0-3)')
    parser.add_argument('--sigma',         type=float, default=0.003,
                        help='EDM noise level')
    parser.add_argument('--cin',           type=float,   default=2.5,)
    parser.add_argument('--ensemble_size', type=int,   default=4,
                        help='Number of noisy copies to average over')
    parser.add_argument('--resolution',    type=int,   default=512,
                        help='Square video resolution to feed into SVD')
    
    parser.add_argument('--max_frames',    type=int,   default=100,
                        help='Max frames per segment to process')
    parser.add_argument('--eval_size',     type=int,   default=256,
                        help='Square video and coordinate size saved for evaluation')
    parser.add_argument('--track_all',     action='store_true',
                        help='Split the entire video into consecutive segments and extract features for all. '
                             'If not set, only the first max_frames are used.')
    parser.add_argument('--query_mode', type=str, default='strided',
                        choices=['first', 'strided'],
                        help='Query sampling mode. strided samples visible points every 5 frames.')
    parser.add_argument('--max_videos', type=int, default=2000,
                        help='Maximum number of dataset videos to extract. Used by Kubric.')
    parser.add_argument('--kubric_shuffle_buffer', type=int, default=128,
                        help='Shuffle buffer for Kubric training samples. Set <=0 to disable.')
    
    parser.add_argument('--overwrite',     action='store_true',
                        help='Re-extract even if .pt already exists')
    parser.set_defaults(track_all=True)
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()
    if args.kubric_shuffle_buffer <= 0:
        args.kubric_shuffle_buffer = None
    main(args)
