"""
Evaluate SVD features on TAP-Vid dataset.

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
# Load .pt
# ============================================================

def load_from_pt(pt_path: str) -> dict:
    """
    Load extracted features and metadata from a .pt file.

    Returns dict with:
        feat:     (F, C, H', W') float16 tensor
        points:   (N, F, 2)      float32 tensor, normalized
        occluded: (N, F)         bool tensor
        video:    (F, H, W, 3)   uint8 tensor
        meta:     dict
    """
    data = torch.load(pt_path, map_location='cpu')
    return data


# ============================================================
# Track a single point
# ============================================================
def track_point(
    spatial_feat: torch.Tensor,   # (F, C, feat_H, feat_W)
    query_frame: int,
    query_y: float,               # in 256x256 space
    query_x: float,               # in 256x256 space
    img_H: int = 256,
    img_W: int = 256,
    local_window: int = 64,
    pixel_window: int = 3,
    temperature: float = 0.02,
    occlusion_threshold: float = 15
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
    occlusions = [False] * F_frames


    def get_feature(f_idx, y, x):
        feat_low = spatial_feat[f_idx:f_idx + 1]
        feat_up = F.interpolate(feat_low, size=(img_H, img_W),
                                mode='bilinear', align_corners=False)
        nx = (x + 0.5) / img_W * 2 - 1.0
        ny = (y + 0.5) / img_H * 2 - 1.0
        coord = torch.tensor([[[[nx, ny]]]], dtype=dtype, device=device)
        vec = F.grid_sample(feat_up, coord, mode='bilinear', align_corners=False)
        return F.normalize(vec.view(1, C), p=2, dim=1)
    
    def process_frame(q_vec, target_f_idx, ref_y, ref_x, local_window=None, pixel_window=3):
        target_feat_low = spatial_feat[target_f_idx:target_f_idx + 1]
        target_feat_up  = F.interpolate(target_feat_low, size=(img_H, img_W),
                                        mode='bilinear', align_corners=False)

        target_feat_flat = target_feat_up.view(C, -1).transpose(0, 1)   # (H*W, C)
        target_feat_flat = F.normalize(target_feat_flat, p=2, dim=1)

        sim_map = torch.mm(target_feat_flat, q_vec.t()).view(img_H, img_W)

        if local_window is not None:
            y_min = max(0, int(ref_y - local_window))
            y_max = min(img_H, int(ref_y + local_window + 1))
            x_min = max(0, int(ref_x - local_window))
            x_max = min(img_W, int(ref_x + local_window + 1))
            mask = torch.full_like(sim_map, -1e4)
            mask[y_min:y_max, x_min:x_max] = 0
            sim_map = sim_map + mask

        max_idx = torch.argmax(sim_map)
        pred_y_int = int((max_idx // img_W).item())
        pred_x_int = int((max_idx % img_W).item())

        if pixel_window is not None:
            y_min_sub = max(0, pred_y_int - pixel_window)
            y_max_sub = min(img_H, pred_y_int + pixel_window + 1)
            x_min_sub = max(0, pred_x_int - pixel_window)
            x_max_sub = min(img_W, pred_x_int + pixel_window + 1)

            local_sim = sim_map[y_min_sub:y_max_sub, x_min_sub:x_max_sub]

            local_weights = F.softmax(local_sim.flatten() / temperature, dim=0).view_as(local_sim)

            y_coords = torch.arange(y_min_sub, y_max_sub, dtype=dtype, device=device)
            x_coords = torch.arange(x_min_sub, x_max_sub, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

            pred_y_sub = torch.sum(local_weights * grid_y).item()
            pred_x_sub = torch.sum(local_weights * grid_x).item()
            return pred_y_sub, pred_x_sub
        else:
            return pred_y_int, pred_x_int

    # query frame itself
    pred_y, pred_x = query_y, query_x
    trajectories[query_frame] = (query_y, query_x)
    occlusions[query_frame] = False

    query_vec = get_feature(query_frame, query_y, query_x)
    # forward
    curr_y, curr_x = trajectories[query_frame]
    last_vis = query_frame
    for f_idx in range(query_frame + 1, F_frames):
        pred_y, pred_x = process_frame(query_vec, f_idx, curr_y, curr_x, local_window, pixel_window)
        curr_vec = get_feature(f_idx, pred_y, pred_x)
        rev_y, rev_x = process_frame(curr_vec, last_vis, curr_y, curr_x)

        error = ((rev_y - curr_y)**2 + (rev_x - curr_x)**2)**0.5
        occlusions[f_idx] = bool(error > occlusion_threshold)

        if occlusions[f_idx]:
            trajectories[f_idx] = (curr_y, curr_x)
        else:
            last_vis = f_idx
            curr_y, curr_x = pred_y, pred_x
            trajectories[f_idx] = (curr_y, curr_x)

    # backward
    curr_y, curr_x = trajectories[query_frame]
    for f_idx in range(query_frame - 1, -1, -1):
        pred_y, pred_x = process_frame(query_vec, f_idx, curr_y, curr_x, local_window, pixel_window)
        curr_vec = get_feature(f_idx, pred_y, pred_x)
        rev_y, rev_x = process_frame(curr_vec, f_idx + 1, curr_y, curr_x)

        error = ((rev_y - curr_y)**2 + (rev_x - curr_x)**2)**0.5
        occlusions[f_idx] = bool(error > occlusion_threshold)
        
        if occlusions[f_idx]:
            trajectories[f_idx] = (curr_y, curr_x)
        else:
            curr_y, curr_x = pred_y, pred_x
            trajectories[f_idx] = (curr_y, curr_x)

    return trajectories, occlusions


@torch.no_grad()
def track_point_dp_smooth(
    spatial_feat: torch.Tensor,   # (F, C, feat_H, feat_W)
    query_frame: int,
    query_y: float,               # in img_H x img_W space
    query_x: float,               # in img_H x img_W space
    img_H: int = 256,
    img_W: int = 256,
    local_window: int = 64,
    occlusion_threshold: float = 15,
    prior_sigma: float = 12.0,
    prior_weight: float = 0.08,
    smooth_sigma: float = 1.5,
):
    """
    Experimental lightweight smoothing tracker.

    Compared with track_point(), this keeps the same greedy local matching and
    bidirectional occlusion check, but adds:
      1. a soft Gaussian motion prior on sim_map around the previous prediction;
      2. a final Gaussian smoothing pass over the predicted trajectory.
    """
    F_frames, C, feat_H, feat_W = spatial_feat.shape
    dtype = spatial_feat.dtype
    device = spatial_feat.device

    trajectories = [(0.0, 0.0)] * F_frames
    occlusions = [False] * F_frames

    yy, xx = torch.meshgrid(
        torch.arange(img_H, dtype=dtype, device=device),
        torch.arange(img_W, dtype=dtype, device=device),
        indexing='ij',
    )

    def get_feature(f_idx, y, x):
        feat_low = spatial_feat[f_idx:f_idx + 1]
        feat_up = F.interpolate(feat_low, size=(img_H, img_W),
                                mode='bilinear', align_corners=False)
        nx = (x + 0.5) / img_W * 2 - 1.0
        ny = (y + 0.5) / img_H * 2 - 1.0
        coord = torch.tensor([[[[nx, ny]]]], dtype=dtype, device=device)
        vec = F.grid_sample(feat_up, coord, mode='bilinear', align_corners=False)
        return F.normalize(vec.view(1, C), p=2, dim=1)

    def process_frame(q_vec, target_f_idx, ref_y, ref_x, apply_prior=True):
        target_feat_low = spatial_feat[target_f_idx:target_f_idx + 1]
        target_feat_up = F.interpolate(target_feat_low, size=(img_H, img_W),
                                       mode='bilinear', align_corners=False)
        target_feat_flat = target_feat_up.view(C, -1).transpose(0, 1)
        target_feat_flat = F.normalize(target_feat_flat, p=2, dim=1)
        sim_map = torch.mm(target_feat_flat, q_vec.t()).view(img_H, img_W)

        if apply_prior and prior_weight > 0:
            dist2 = (yy - float(ref_y)) ** 2 + (xx - float(ref_x)) ** 2
            prior = torch.exp(-dist2 / (2.0 * prior_sigma ** 2))
            sim_map = sim_map + prior_weight * prior

        if local_window is not None:
            y_min = max(0, int(ref_y - local_window))
            y_max = min(img_H, int(ref_y + local_window + 1))
            x_min = max(0, int(ref_x - local_window))
            x_max = min(img_W, int(ref_x + local_window + 1))
            mask = torch.full_like(sim_map, -1e4)
            mask[y_min:y_max, x_min:x_max] = 0
            sim_map = sim_map + mask

        max_idx = torch.argmax(sim_map)
        pred_y_int = int((max_idx // img_W).item())
        pred_x_int = int((max_idx % img_W).item())
        return pred_y_int, pred_x_int

    def smooth_trajectory(points):
        if smooth_sigma is None or smooth_sigma <= 0:
            return points

        radius = max(1, int(round(3 * smooth_sigma)))
        kernel_x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-(kernel_x ** 2) / (2.0 * smooth_sigma ** 2))
        kernel = (kernel / kernel.sum()).view(1, 1, -1)

        traj = torch.tensor(points, dtype=torch.float32, device=device).transpose(0, 1).unsqueeze(1)
        traj = F.pad(traj, (radius, radius), mode='replicate')
        traj = F.conv1d(traj, kernel).squeeze(1).transpose(0, 1)
        traj[:, 0].clamp_(0, img_H - 1)
        traj[:, 1].clamp_(0, img_W - 1)
        smoothed = [(float(y), float(x)) for y, x in traj.detach().cpu().tolist()]
        smoothed[query_frame] = (query_y, query_x)
        return smoothed

    trajectories[query_frame] = (query_y, query_x)
    occlusions[query_frame] = False

    query_vec = get_feature(query_frame, query_y, query_x)

    curr_y, curr_x = trajectories[query_frame]
    last_vis = query_frame
    for f_idx in range(query_frame + 1, F_frames):
        pred_y, pred_x = process_frame(query_vec, f_idx, curr_y, curr_x, apply_prior=True)
        curr_vec = get_feature(f_idx, pred_y, pred_x)
        rev_y, rev_x = process_frame(curr_vec, last_vis, curr_y, curr_x, apply_prior=False)

        error = ((rev_y - curr_y) ** 2 + (rev_x - curr_x) ** 2) ** 0.5
        occlusions[f_idx] = bool(error > occlusion_threshold)

        if occlusions[f_idx]:
            trajectories[f_idx] = (curr_y, curr_x)
        else:
            last_vis = f_idx
            curr_y, curr_x = pred_y, pred_x
            trajectories[f_idx] = (curr_y, curr_x)

    curr_y, curr_x = trajectories[query_frame]
    for f_idx in range(query_frame - 1, -1, -1):
        pred_y, pred_x = process_frame(query_vec, f_idx, curr_y, curr_x, apply_prior=True)
        curr_vec = get_feature(f_idx, pred_y, pred_x)
        rev_y, rev_x = process_frame(curr_vec, f_idx + 1, curr_y, curr_x, apply_prior=False)

        error = ((rev_y - curr_y) ** 2 + (rev_x - curr_x) ** 2) ** 0.5
        occlusions[f_idx] = bool(error > occlusion_threshold)

        if occlusions[f_idx]:
            trajectories[f_idx] = (curr_y, curr_x)
        else:
            curr_y, curr_x = pred_y, pred_x
            trajectories[f_idx] = (curr_y, curr_x)

    trajectories = smooth_trajectory(trajectories)

    return trajectories, occlusions


# ============================================================
# Visualize tracking to mp4
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
        if t == query_frame:
            qx = int(round(query_x * scale_x))
            qy = int(round(query_y * scale_y))
            cv2.circle(frame, (qx, qy), dot_radius, (255, 0, 0), -1)

        # white ring on query frame to mark the start
        if t == query_frame:
            cv2.circle(frame, (px, py), dot_radius + 5, (255, 255, 255), 1)

        frames_out.append(torch.from_numpy(frame))

    video_out = torch.stack(frames_out, dim=0)  # (F, H, W, 3) uint8
    write_video(save_path, video_out, fps=fps, video_codec='libx264',
                options={'crf': '18'})
    print(f"  Saved: {save_path}")


# ============================================================
# Compute_tapvid_metrics (from official TAP-Vid code)
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
# Main evaluation
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
    gt_tracks_px = points * eval_size          # (N, F, 2) pixel (x, y) in eval_size space

    # build query_points in [t, y, x] format using 'first' mode:
    # for each track, find the first non-occluded frame as query
    query_points_list = []
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
    all_pred_occlusions = []
    
    for n in tqdm(range(N_valid), desc=f'  Tracking {vid_name}', leave=False):
        t_q = int(query_points_arr[n, 0])
        y_q = float(query_points_arr[n, 1])   # already in 256x256 space
        x_q = float(query_points_arr[n, 2])

        trajectories, occlusions = track_point(
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
        pred_occ = np.array(occlusions, dtype=np.bool_)
        all_pred_occlusions.append(pred_occ)

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
        # print(gt_tracks_px[n])
        # print(occluded_valid[n])
        # print(pred_xy)
        # print(pred_occ)
        # break

    pred_tracks_arr = np.stack(all_pred_tracks, axis=0)   # (N_valid, F, 2)
    pred_occ_arr = np.stack(all_pred_occlusions, axis=0)      # (N_valid, F)

    # compute metrics — add batch dim=1
    query_points_b = query_points_arr[np.newaxis]          # (1, N_valid, 3)
    gt_tracks_b    = gt_tracks_px[np.newaxis]              # (1, N_valid, F, 2)
    gt_occluded_b  = occluded_valid[np.newaxis]            # (1, N_valid, F)
    pred_tracks_b  = pred_tracks_arr[np.newaxis]           # (1, N_valid, F, 2)
    pred_occ_b     = pred_occ_arr[np.newaxis]              # (1, N_valid, F)

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
    failed_videos = {}

    for pt_path in tqdm(pt_files, desc='Videos'):
        vid_name = pt_path.stem
        # if vid_name != "bike-packing":  # for quick testing, only evaluate one video, can be adjusted
        #     continue
        try:
            metrics = evaluate_video(pt_path, output_dir, eval_size=args.eval_size)
            per_video_metrics[vid_name] = metrics
            print(f"  {vid_name}: AJ={metrics['average_jaccard']:.4f}  "
                  f"<delta={metrics['average_pts_within_thresh']:.4f}  "
                  f"OA={metrics['occlusion_accuracy']:.4f}")
        except Exception as e:
            print(f"  ERROR on {vid_name}: {e}")
            failed_videos[vid_name] = str(e)
            continue

    num_samples = len(per_video_metrics)
    if num_samples == 0:
        raise RuntimeError("No videos were evaluated successfully.")

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
            'num_samples': num_samples,
            'num_pt_files': len(pt_files),
            'num_failed': len(failed_videos),
            'failed_videos': failed_videos,
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
