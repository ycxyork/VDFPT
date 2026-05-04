"""
Extract SVD features for TAP-Vid dataset.

Usage:
python extract_features_old.py \
    --data_path data/tapvid_davis/tapvid_davis.pkl \
    --output_dir /content/output \
    --track_all 
"""

import gc
import math
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import cv2

from feature_svd import SVDFeaturizer


# ============================================================
# Utils for loading and preprocessing videos
# ============================================================

def load_tapvid_davis(data_path: str) -> dict:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} videos from davis: {list(data.keys())[:5]} ...")
    # ---test print structure of one sample---
    sample = list(data.values())[0]
    print(type(sample['video']), sample['video'].shape)
    return data

def decode_video_bytes(video_bytes):
    frames = []
    for frame_bytes in video_bytes:
        # bytes → numpy
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Bad frame decode")
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.stack(frames)  # (T, H, W, 3)

def load_tapvid_rgb(data_path: str) -> dict:
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)  # list
    print(f"Loaded {len(data_list)} videos from rgb (list format)")
    data = {}
    for i, sample in enumerate(data_list):
        vid_name = f"video_{i:04d}" 
        video = sample['video']        # (T, H, W, 3)
        points = sample['points']      # (N, T, 2)
        occluded = sample['occluded']  # (N, T)
        if isinstance(video[0], (bytes, bytearray)):
            video = decode_video_bytes(video)
        data[vid_name] = {
            'video': video,
            'points': points.astype('float32'),
            'occluded': occluded.astype('bool')
        }
    print(f"Converted to dict with keys: {list(data.keys())[:5]} ...")
    # ---test print structure of one sample---
    sample = list(data.values())[0]
    print(type(sample['video']), sample['video'].shape)
    return data


def load_tapvid_kinetics(data_path: str) -> dict:
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} videos (list format)")
    data = {}
    for i, sample in enumerate(data_list):
        vid_name = f"video_{i:04d}"
        video_bytes = sample['video']  # bytes
        points = sample['points']
        occluded = sample['occluded']
        video = decode_video_bytes(video_bytes)
        data[vid_name] = {
            'video': video.astype(np.uint8),   # (T, H, W, 3)
            'points': points.astype('float32'),
            'occluded': occluded.astype('bool')
        }
    print(f"Converted to dict with keys: {list(data.keys())[:5]} ...")
    # ---test print structure of one sample---
    sample = list(data.values())[0]
    print(type(sample['video']), sample['video'].shape)
    return data

def preprocess_video(video: np.ndarray) -> torch.Tensor:
    """
    Resize frames to 512x512 and normalize to [-1, 1]
    Args:
        video:  (T, H, W, 3) uint8
    Returns:
        tensor: (F, 3, 512, 512) float32 in [-1, 1]
    """
    T = video.shape[0]
    frames = []
    for frame in video:
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 127.5 - 1.0   # [0,255] → [-1,1]
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # (3, 512, 512)
        frames.append(frame)
    return torch.stack(frames, dim=0)  # (F, 3, 512, 512)


# def save_video_mp4(video: np.ndarray, out_path: str, fps: int = 8):
#     """Save (T, H, W, 3) uint8 numpy array as mp4."""
#     write_video(out_path, torch.from_numpy(video), fps=fps, video_codec='libx264')


# ============================================================
# Main extraction
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load dataset ----
    data_str = str(args.data_path).lower()
    data = None
    # data = load_tapvid_davis(args.data_path)
    if 'kinetics' in data_str:
        data = load_tapvid_kinetics(args.data_path)
    elif 'rgb' in data_str:
        data = load_tapvid_rgb(args.data_path)
    else:
        data = load_tapvid_davis(args.data_path)

    # ---- init featurizer (once for all videos) ----
    print("Initializing SVDFeaturizer ...")
    featurizer = SVDFeaturizer(
        svd_id=args.svd_id,
    )

    # ---- iterate videos ----
    for vid_name in tqdm(sorted(data.keys()), desc="Extracting features"):
        out_pt  = output_dir / f"{vid_name}.pt"
        out_mp4 = output_dir / f"{vid_name}.mp4"

        if out_pt.exists() and not args.overwrite:
            print(f"  Skip {vid_name} already exists")
            continue

        sample    = data[vid_name]
        video_np  = sample['video']    # (T, H, W, 3) uint8
        points    = sample['points']   # (N, T, 2)    float32, normalized (x,y)
        occluded  = sample['occluded'] # (N, T)       bool

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
        print(f"{vid_name}: {T_total} frames || "
              f"{'sliding' if args.track_all else 'truncate'} mode: "
              f"{len(segments)} segments {T_used} frames total")
        
        # save original-resolution mp4 once
        # if not out_mp4.exists():
        #     save_video_mp4(video_np, str(out_mp4), fps=args.fps)

        # ---- extract features for each segment----
        feat_segments = []
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_len = seg_end - seg_start
            video_segment = video_np[seg_start:seg_end]  # (seg_len, H, W, 3)
 
            # preprocess: resize to 512x512, normalize to [-1, 1]
            video_tensor = preprocess_video(video_segment)
 
            with torch.no_grad():
                seg_feat = featurizer.forward(
                    video_tensor=video_tensor.cuda(),
                    sigma=args.sigma,
                    up_ft_index=args.up_ft_block,
                    ensemble_size=args.ensemble_size,
                    num_frames=seg_len,
                )  # (seg_len, C, H', W')
 
            feat_segments.append(seg_feat.cpu())
            print(f"    segment [{seg_start}:{seg_end}]  feat={tuple(seg_feat.shape)}")
 
            del seg_feat, video_tensor
            torch.cuda.empty_cache()
            gc.collect()

        # ---- concatenate all segments----
        spatial_feat = torch.cat(feat_segments, dim=0)  # (T_used, C, H', W')
        assert spatial_feat.shape[0] == T_used, \
            f"Expected {T_used} frames, got {spatial_feat.shape[0]}"

        # ---- save everything needed for matching and evaluation----
        torch.save({
            'feat':     spatial_feat.cpu(),               # (F, C, H', W') fp16
            'points':   torch.from_numpy(points[:, :T_used]),  # (N, F, 2) normalized (x,y)
            'occluded': torch.from_numpy(occluded[:, :T_used]),# (N, F) bool
            'video':    torch.from_numpy(video_np[:T_used]),   # (F, H, W, 3) uint8
            'meta': {
                'vid_name':      vid_name,
                'num_frames':    T_used,
                'orig_shape':    tuple(video_np.shape),
                'feat_shape':    tuple(spatial_feat.shape),
                'up_ft_block':   args.up_ft_block,
                'sigma':         args.sigma,
                'ensemble_size': args.ensemble_size,
            },
        }, str(out_pt))

        print(f"  Saved: {out_pt}  feat={tuple(spatial_feat.shape)}")

        del spatial_feat
        gc.collect()

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract SVD features for TAP-Vid DAVIS dataset.'
    )
    parser.add_argument('--data_path',     type=str,   required=True,
                        help='Path to tapvid_dataname.pkl')
    parser.add_argument('--output_dir',    type=str,   required=True,
                        default='output/tapvid_davis')
    parser.add_argument('--svd_id',        type=str,
                        default='stabilityai/stable-video-diffusion-img2vid-xt')
    parser.add_argument('--up_ft_block',   type=int,   default=2,
                        help='Up-block index to extract features from (0-3)')
    parser.add_argument('--sigma',         type=float, default=0.003,
                        help='EDM noise level')
    parser.add_argument('--ensemble_size', type=int,   default=4,
                        help='Number of noisy copies to average over')
    parser.add_argument('--max_frames',    type=int,   default=100,
                        help='Max frames per segment to process')
    parser.add_argument('--track_all',     action='store_true',
                        help='Split the entire video into consecutive segments and extract features for all'
                             'If not set (default), only the first max_frames are used')
    parser.add_argument('--fps',           type=int,   default=8,
                        help='FPS for saved mp4')
    parser.add_argument('--overwrite',     action='store_true',
                        help='Re-extract even if .pt already exists')
    args = parser.parse_args()
    main(args)
'''
python extract_features.py \
    --data_path data/project/0000_of_0010.pkl \
    --output_dir /content/output \
    --track_all 
'''