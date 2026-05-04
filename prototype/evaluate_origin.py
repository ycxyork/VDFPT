"""
Evaluate SVD features on TAP-Vid DAVIS.

Usage:
    python evaluate.py \
        --feat_dir /content/output \
        --output_dir /content/result
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import write_video
from tqdm import tqdm


# ============================================================
# 1. Load .pt
# ============================================================

def load_from_pt(pt_path: str) -> dict:
    """
    Load extracted features and metadata from a .pt file.

    Returns dict with:
        feat:     (F, C, H', W') float16 tensor
        points:   (N, F, 2)      float32 tensor, normalized (x, y) in [0, 1]
        occluded: (N, F)         bool tensor
        video:    (F, H, W, 3)   uint8 tensor
        meta:     dict
    """
    data = torch.load(pt_path, map_location='cpu')
    return data


# ============================================================
# 2. Track a single point
# ============================================================

def track_point(
    spatial_feat: torch.Tensor,   # (F, C, feat_H, feat_W)
    query_frame: int,
    query_y: float,               # in 256x256 space
    query_x: float,               # in 256x256 space
    img_H: int = 256,
    img_W: int = 256,
    local_window: int = 32,
    use_soft_argmax: bool = False,
    temperature: float = 0.05,
    top_k: int = 5,
):
    """
    Track a single query point across all frames using cosine similarity.

    Returns:
        trajectories: list of (pred_y, pred_x) in img_H x img_W space, length F
    """
    F_frames, C, feat_H, feat_W = spatial_feat.shape
    dtype = spatial_feat.dtype
    device = spatial_feat.device

    trajectories = [(0.0, 0.0)] * F_frames

    # upsample query frame feature to img_H x img_W
    query_feat_low = spatial_feat[query_frame:query_frame + 1]   # (1, C, feat_H, feat_W)
    query_feat_up  = F.interpolate(query_feat_low, size=(img_H, img_W),
                                   mode='bilinear', align_corners=False)  # (1, C, H, W)

    # sample query vector at (query_y, query_x)
    norm_x = (query_x + 0.5) / img_W * 2 - 1.0
    norm_y = (query_y + 0.5) / img_H * 2 - 1.0
    query_coord = torch.tensor([[[[norm_x, norm_y]]]], dtype=dtype, device=device)

    query_vec = F.grid_sample(query_feat_up, query_coord,
                              mode='bilinear', align_corners=False)
    query_vec = query_vec.view(1, C)
    query_vec = F.normalize(query_vec, p=2, dim=1)  # (1, C)

    def process_frame(f_idx, ref_y, ref_x):
        target_feat_low = spatial_feat[f_idx:f_idx + 1]
        target_feat_up  = F.interpolate(target_feat_low, size=(img_H, img_W),
                                        mode='bilinear', align_corners=False)

        target_feat_flat = target_feat_up.view(C, -1).transpose(0, 1)   # (H*W, C)
        target_feat_flat = F.normalize(target_feat_flat, p=2, dim=1)

        sim_map = torch.mm(target_feat_flat, query_vec.t()).view(img_H, img_W)

        if local_window is not None:
            y_min = max(0, int(ref_y - local_window))
            y_max = min(img_H, int(ref_y + local_window + 1))
            x_min = max(0, int(ref_x - local_window))
            x_max = min(img_W, int(ref_x + local_window + 1))
            mask = torch.full_like(sim_map, -1e4)
            mask[y_min:y_max, x_min:x_max] = 0
            sim_map = sim_map + mask

        if not use_soft_argmax:
            max_idx = torch.argmax(sim_map)
            pred_y = float((max_idx // img_W).item())
            pred_x = float((max_idx % img_W).item())
        else:
            sim_flat = sim_map.view(-1)
            tk_vals, tk_idx = torch.topk(sim_flat, k=top_k, dim=0)
            weights = F.softmax(tk_vals / temperature, dim=0)
            pred_y = torch.sum(weights * (tk_idx // img_W).float()).item()
            pred_x = torch.sum(weights * (tk_idx %  img_W).float()).item()

        return pred_y, pred_x

    # query frame itself
    pred_y, pred_x = query_y, query_x
    trajectories[query_frame] = (pred_y, pred_x)

    # forward
    curr_y, curr_x = trajectories[query_frame]
    for f_idx in range(query_frame + 1, F_frames):
        curr_y, curr_x = process_frame(f_idx, curr_y, curr_x)
        trajectories[f_idx] = (curr_y, curr_x)

    # backward
    curr_y, curr_x = trajectories[query_frame]
    for f_idx in range(query_frame - 1, -1, -1):
        curr_y, curr_x = process_frame(f_idx, curr_y, curr_x)
        trajectories[f_idx] = (curr_y, curr_x)

    return trajectories


# ============================================================
# 3. Visualize tracking → mp4
# ============================================================

def visualize_tracking(
    trajectories: list,          # list of (pred_y, pred_x), length F, in 256x256 space
    gt_track: np.ndarray,        # (F, 2) float32, pixel (x, y) in 256x256 space
    video_tensor: torch.Tensor,  # (F, H, W, 3) uint8
    query_frame: int,
    query_y: float,
    query_x: float,
    save_path: str = 'tracking.mp4',
    fps: int = 8,
    dot_radius: int = 4,
):
    """
    Draw three sets of points on each frame and save as mp4:
      - Red:   fixed query point (same position every frame)
      - Green: predicted position at this frame
      - Blue:  ground truth position at this frame

    All coordinates are in the video's native resolution space,
    so we scale from 256x256 to (H, W) before drawing.
    """
    F_num = video_tensor.shape[0]
    H, W = video_tensor.shape[1], video_tensor.shape[2]
    assert len(trajectories) == F_num

    scale_x = W / 256.0
    scale_y = H / 256.0

    frames_out = []

    for t in range(F_num):
        frame = video_tensor[t].cpu().numpy().copy()  # (H, W, 3) uint8

        # predicted point (green)
        pred_y, pred_x = trajectories[t]
        px = int(round(pred_x * scale_x))
        py = int(round(pred_y * scale_y))
        cv2.circle(frame, (px, py), dot_radius, (0, 255, 0), -1)

        # gt point (blue)
        gx = int(round(float(gt_track[t, 0]) * scale_x))
        gy = int(round(float(gt_track[t, 1]) * scale_y))
        cv2.circle(frame, (gx, gy), dot_radius, (0, 0, 255), -1)

        # query point (red, fixed every frame)
        qx = int(round(query_x * scale_x))
        qy = int(round(query_y * scale_y))
        cv2.circle(frame, (qx, qy), dot_radius, (255, 0, 0), -1)

        # white ring on query frame to mark the start
        if t == query_frame:
            cv2.circle(frame, (px, py), dot_radius + 2, (255, 255, 255), 1)

        frames_out.append(torch.from_numpy(frame))

    video_out = torch.stack(frames_out, dim=0)  # (F, H, W, 3) uint8
    write_video(save_path, video_out, fps=fps, video_codec='libx264',
                options={'crf': '18'})
    print(f"  Saved: {save_path}")


# ============================================================
# 4. compute_tapvid_metrics (from official TAP-Vid code)
# ============================================================

def compute_tapvid_metrics(
    query_points: np.ndarray,   # (1, N, 3)  [t, y, x] pixel coords
    gt_occluded: np.ndarray,    # (1, N, T)  bool
    gt_tracks: np.ndarray,      # (1, N, T, 2) [x, y] pixel coords
    pred_occluded: np.ndarray,  # (1, N, T)  bool
    pred_tracks: np.ndarray,    # (1, N, T, 2) [x, y] pixel coords
    query_mode: str = 'first',
) -> dict:
    summing_axis = (1, 2)
    metrics = {}

    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == 'first':
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == 'strided':
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError('Unknown query mode ' + query_mode)

    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0  # (1, N, T)

    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=summing_axis,
    ) / np.sum(evaluation_points, axis=summing_axis)
    metrics['occlusion_accuracy'] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks), axis=-1
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        count_correct = np.sum(is_correct & evaluation_points, axis=summing_axis)
        count_visible = np.sum(visible & evaluation_points, axis=summing_axis)
        frac_correct = count_correct / count_visible
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives  = np.sum(is_correct & pred_visible & evaluation_points, axis=summing_axis)
        gt_positives    = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=summing_axis)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    return metrics


# ============================================================
# 5. Main evaluation loop
# ============================================================

def evaluate_video(pt_path: str, output_dir: Path, eval_size: int = 256):
    """
    Evaluate one video: track all query points, compute metrics, save videos.

    Returns:
        metrics_dict: per-point metrics aggregated to per-video scalar dict
    """
    data      = load_from_pt(pt_path)
    feat      = data['feat'].float().cuda()    # (F, C, H', W')
    points    = data['points'].numpy()         # (N, F, 2) normalized (x,y)
    occluded  = data['occluded'].numpy()       # (N, F) bool
    video     = data['video']                  # (F, H, W, 3) uint8 tensor
    vid_name  = data['meta']['vid_name']

    F_frames = feat.shape[0]
    N_points = points.shape[0]

    # convert normalized points → eval_size pixel coords
    # points: (x, y) normalized → (x, y) in eval_size space
    gt_tracks_px = points * eval_size          # (N, F, 2) pixel (x, y) in 256x256

    # build query_points in [t, y, x] format using 'first' mode:
    # for each track, find the first non-occluded frame as query
    query_points_list = []   # each: [t, y, x]
    valid_track_mask  = []

    for n in range(N_points):
        visible_frames = np.where(~occluded[n])[0]
        if len(visible_frames) == 0:
            valid_track_mask.append(False)
            continue
        valid_track_mask.append(True)
        t_q = int(visible_frames[0])  # fist non-occluded frame
        x_q = float(gt_tracks_px[n, t_q, 0])
        y_q = float(gt_tracks_px[n, t_q, 1])
        query_points_list.append([t_q, y_q, x_q])   # [t, y, x]

    valid_track_mask = np.array(valid_track_mask)
    gt_tracks_px   = gt_tracks_px[valid_track_mask]   # (N_valid, F, 2)
    occluded_valid = occluded[valid_track_mask]       # (N_valid, F)
    N_valid        = gt_tracks_px.shape[0]

    query_points_arr = np.array(query_points_list, dtype=np.float32)  # (N_valid, 3)

    # track each point
    all_pred_tracks = []

    for n in tqdm(range(N_valid), desc=f'  Tracking {vid_name}', leave=False):
        t_q = int(query_points_arr[n, 0])
        y_q = float(query_points_arr[n, 1])   # already in 256x256 space
        x_q = float(query_points_arr[n, 2])

        trajectories = track_point(
            spatial_feat=feat,
            query_frame=t_q,
            query_y=y_q,
            query_x=x_q,
            img_H=eval_size,
            img_W=eval_size,
        )
        # trajectories: list of (pred_y, pred_x) in 256x256 space
        # convert to (F, 2) array in (x, y) order to match gt_tracks format
        pred_xy = np.array([[px, py] for py, px in trajectories], dtype=np.float32)  # (F, 2)
        all_pred_tracks.append(pred_xy)

        # visualize this track
        if n < 0:   # save at most 5 videos per video for quick inspection, can be adjusted
            vis_path = output_dir / f"{vid_name}_track{n:02d}.mp4"
            visualize_tracking(
                trajectories=trajectories,
                gt_track=gt_tracks_px[n],          # (F, 2) pixel (x, y)
                video_tensor=video,
                query_frame=t_q,
                query_y=y_q,
                query_x=x_q,
                save_path=str(vis_path),
            )

    pred_tracks_arr = np.stack(all_pred_tracks, axis=0)   # (N_valid, F, 2)

    # compute metrics — add batch dim=1
    query_points_b = query_points_arr[np.newaxis]          # (1, N_valid, 3)
    gt_tracks_b    = gt_tracks_px[np.newaxis]              # (1, N_valid, F, 2)
    gt_occluded_b  = occluded_valid[np.newaxis]            # (1, N_valid, F)
    pred_tracks_b  = pred_tracks_arr[np.newaxis]           # (1, N_valid, F, 2)
    pred_occ_b     = np.zeros_like(gt_occluded_b)          # all False

    metrics = compute_tapvid_metrics(
        query_points=query_points_b,
        gt_occluded=gt_occluded_b,
        gt_tracks=gt_tracks_b,
        pred_occluded=pred_occ_b,
        pred_tracks=pred_tracks_b,
        query_mode='first',
    )

    # squeeze batch dim → scalar per video
    scalar_metrics = {k: float(v[0]) for k, v in metrics.items()}
    return scalar_metrics


def main(args):
    feat_dir   = Path(args.feat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(feat_dir.glob('*.pt'))
    assert len(pt_files) > 0, f"No .pt files found in {feat_dir}"
    print(f"Found {len(pt_files)} videos in {feat_dir}")

    per_video_metrics = {}

    for pt_path in tqdm(pt_files, desc='Videos'):
        vid_name = pt_path.stem
        try:
            metrics = evaluate_video(pt_path, output_dir, eval_size=args.eval_size)
            per_video_metrics[vid_name] = metrics
            print(f"  {vid_name}: AJ={metrics['average_jaccard']:.4f}  "
                  f"<delta={metrics['average_pts_within_thresh']:.4f}  "
                  f"OA={metrics['occlusion_accuracy']:.4f}")
        except Exception as e:
            print(f"  ERROR on {vid_name}: {e}")
            continue

    # aggregate mean across all videos
    all_keys = list(next(iter(per_video_metrics.values())).keys())
    mean_metrics = {
        k: float(np.mean([per_video_metrics[v][k] for v in per_video_metrics]))
        for k in all_keys
    }

    results = {
        'per_video': per_video_metrics,
        'mean':      mean_metrics,
        'config': {
            'feat_dir':   str(feat_dir),
            'eval_size':  args.eval_size,
            'query_mode': 'first',
        }
    }

    out_json = output_dir / 'metrics.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n===== Mean metrics across all videos =====")
    for k, v in mean_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved to: {out_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir',   type=str, required=True,
                        help='Directory containing .pt files, e.g. block2_t261/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save metrics.json and tracking videos')
    parser.add_argument('--eval_size',  type=int, default=256,
                        help='Evaluation resolution (default: 256)')
    parser.add_argument('--save_video', action='store_true',
                        help='Save tracking visualization mp4 (disabled by default for grid search)')
    args = parser.parse_args()
    main(args)