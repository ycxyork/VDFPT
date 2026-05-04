"""
Evaluate extracted SVD features on TAP-Vid datasets.

This evaluator consumes the .pt schema written by extract_features.py:
  feat, query_points, target_points, occluded, video, meta

Usage:
python evaluate.py \
    --feat_dir /content/output/davis \
    --output_dir /content/result/vis
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

from data_utils import compute_tapvid_metrics


def load_from_pt(pt_path: str) -> dict:
    """Load extracted features and metadata from a .pt file."""
    return torch.load(pt_path, map_location='cpu')


def can_preupsample(feat: torch.Tensor, img_H: int, img_W: int, max_free_ratio: float = 0.9) -> bool:
    """Return True if pre-upsampling the whole video is likely to fit in GPU memory."""
    F_frames, C, _, _ = feat.shape
    upsampled_numel = F_frames * C * img_H * img_W
    upsampled_bytes = upsampled_numel * feat.element_size()

    if not feat.is_cuda:
        return True

    try:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(feat.device)
    except RuntimeError:
        return False

    return upsampled_bytes < free_bytes * max_free_ratio

# Track a single point
def track_point(
    spatial_feat: torch.Tensor,   # (F, C, feat_H, feat_W)
    query_frame: int,
    query_y: float, # in img_H x img_W space
    query_x: float, # in img_H x img_W space
    img_H: int,
    img_W: int,
    local_window: int = 64,
    pixel_window: int = 3,
    temperature: float = 0.02,
    occlusion_threshold: float = 15,
    already_upsample: bool = False,
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

    if already_upsample:
        assert feat_H == img_H and feat_W == img_W, (
            f"already_upsample=True expects feature size {(feat_H, feat_W)} "
            f"to match image size {(img_H, img_W)}"
        )

    def maybe_upsample(feat_frame):
        if already_upsample:
            return feat_frame
        return F.interpolate(
            feat_frame, size=(img_H, img_W), mode='bilinear', align_corners=False
        )

    def get_feature(f_idx, y, x):
        feat_low = spatial_feat[f_idx:f_idx + 1]
        feat_up = maybe_upsample(feat_low)
        nx = (x + 0.5) / img_W * 2 - 1.0
        ny = (y + 0.5) / img_H * 2 - 1.0
        coord = torch.tensor([[[[nx, ny]]]], dtype=dtype, device=device)
        vec = F.grid_sample(feat_up, coord, mode='bilinear', align_corners=False)
        return F.normalize(vec.view(1, C), p=2, dim=1)

    def process_frame(q_vec, target_f_idx, ref_y, ref_x, local_window=None, pixel_window=3):
        target_feat_low = spatial_feat[target_f_idx:target_f_idx + 1]
        target_feat_up = maybe_upsample(target_feat_low)

        target_feat_flat = target_feat_up.view(C, -1).transpose(0, 1)
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

        if pixel_window is None:
            return pred_y_int, pred_x_int

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

    # query frame itself
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

        error = ((rev_y - curr_y) ** 2 + (rev_x - curr_x) ** 2) ** 0.5
        occlusions[f_idx] = bool(error > occlusion_threshold)
        if occlusions[f_idx]:
            trajectories[f_idx] = (curr_y, curr_x)
        else:
            last_vis = f_idx
            curr_y, curr_x = pred_y, pred_x
            trajectories[f_idx] = (curr_y, curr_x)

    # backward
    curr_y, curr_x = trajectories[query_frame]
    last_vis = query_frame
    for f_idx in range(query_frame - 1, -1, -1):
        pred_y, pred_x = process_frame(query_vec, f_idx, curr_y, curr_x, local_window, pixel_window)
        curr_vec = get_feature(f_idx, pred_y, pred_x)
        rev_y, rev_x = process_frame(curr_vec, last_vis, curr_y, curr_x)

        error = ((rev_y - curr_y) ** 2 + (rev_x - curr_x) ** 2) ** 0.5
        occlusions[f_idx] = bool(error > occlusion_threshold)
        if occlusions[f_idx]:
            trajectories[f_idx] = (curr_y, curr_x)
        else:
            last_vis = f_idx
            curr_y, curr_x = pred_y, pred_x
            trajectories[f_idx] = (curr_y, curr_x)

    return trajectories, occlusions


# Visualize tracking to mp4
def visualize_tracking(
    trajectories: list,
    gt_track: np.ndarray,
    video_tensor: torch.Tensor,
    query_frame: int,
    query_y: float,
    query_x: float,
    save_path: str = 'tracking.mp4',
    fps: int = 16,
    dot_radius: int = 4,
    vis_size: int = 512,
    strip_gap: int = 8,
):
    """Draw predicted and ground-truth tracks and save an mp4 plus a horizontal image strip."""
    F_num = video_tensor.shape[0]
    H, W = video_tensor.shape[1], video_tensor.shape[2]
    assert len(trajectories) == F_num
    scale_x = vis_size / W
    scale_y = vis_size / H
    vis_radius = max(1, int(round(dot_radius * (scale_x + scale_y) * 0.5)))

    def frame_to_uint8(t):
        frame = video_tensor[t].cpu().numpy()
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip((frame + 1.0) * 127.5, 0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8, copy=True)
        return cv2.resize(frame, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

    def scale_point(y, x):
        return int(round(x * scale_x)), int(round(y * scale_y))

    def draw_points(frame, t):
        pred_y, pred_x = trajectories[t]
        cv2.circle(frame, scale_point(pred_y, pred_x), vis_radius, (0, 255, 0), -1)

        gx = float(gt_track[t, 0])
        gy = float(gt_track[t, 1])
        cv2.circle(frame, scale_point(gy, gx), vis_radius, (0, 0, 255), -1)

        if t == query_frame:
            cv2.circle(frame, scale_point(query_y, query_x), vis_radius, (255, 0, 0), -1)
            cv2.circle(frame, scale_point(pred_y, pred_x), vis_radius + 4, (255, 255, 255), 1)
        return frame

    def choose_strip_frames(stride=5, count=5):
        candidates = []
        max_steps = max(F_num, count * stride)
        for offset in range(0, max_steps + stride, stride):
            if offset == 0:
                frame_ids = [query_frame]
            else:
                frame_ids = [query_frame - offset, query_frame + offset]
            for frame_id in frame_ids:
                if 0 <= frame_id < F_num and frame_id not in candidates:
                    candidates.append(frame_id)
                if len(candidates) >= count:
                    return sorted(candidates[:count])
        return sorted(candidates)

    frames_out = []
    for t in range(F_num):
        frame = draw_points(frame_to_uint8(t), t)
        frames_out.append(torch.from_numpy(frame))

    video_out = torch.stack(frames_out, dim=0)
    write_video(save_path, video_out, fps=fps, video_codec='libx264', options={'crf': '18'})
    print(f"  Saved: {save_path}")

    strip_frames = [draw_points(frame_to_uint8(t), t) for t in choose_strip_frames()]
    if strip_frames:
        gap = np.full((vis_size, strip_gap, 3), 255, dtype=np.uint8)
        strip_parts = []
        for idx, frame in enumerate(strip_frames):
            if idx > 0:
                strip_parts.append(gap)
            strip_parts.append(frame)
        strip = np.concatenate(strip_parts, axis=1)
        strip_path = str(Path(save_path).with_suffix('.jpg'))
        cv2.imwrite(strip_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {strip_path}")

# Main evaluation
def evaluate_video(pt_path: str, output_dir: Path):
    """
    Evaluate one video: track all query points, compute metrics, save videos.

    Returns:
        metrics_dict: per-point metrics aggregated to per-video scalar dict
    """
    
    data = load_from_pt(pt_path)
    feat = data['feat'].float().cuda()              # (F, C, H', W')
    query_points_arr = data['query_points'].numpy() # (N, 3) pixel [t, y, x]
    gt_tracks_query = data['target_points'].numpy() # (N, F, 2) pixel [x, y]
    occluded_query = data['occluded'].numpy()       # (N, F) bool
    video = data['video']
    meta = data['meta']
    vid_name = meta['vid_name']
    query_mode = meta['query_mode']

    img_H, img_W = int(meta['eval_size']), int(meta['eval_size'])

    assert feat.shape[0] == gt_tracks_query.shape[1], (
        f"Feature frames {feat.shape[0]} != target frames {gt_tracks_query.shape[1]}"
    )
    assert feat.shape[0] == occluded_query.shape[1], (
        f"Feature frames {feat.shape[0]} != occlusion frames {occluded_query.shape[1]}"
    )
    assert query_points_arr.shape[0] == gt_tracks_query.shape[0], (
        f"Queries {query_points_arr.shape[0]} != tracks {gt_tracks_query.shape[0]}"
    )
    assert query_points_arr.shape[0] == occluded_query.shape[0], (
        f"Queries {query_points_arr.shape[0]} != occlusion tracks {occluded_query.shape[0]}"
    )

    all_pred_tracks = []
    all_pred_occlusions = []

    use_preupsampled_feat = can_preupsample(feat, img_H, img_W)
    if use_preupsampled_feat:
        try:
            with torch.no_grad():
                feat_for_tracking = F.interpolate(
                    feat, size=(img_H, img_W), mode='bilinear', align_corners=False
                )
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            use_preupsampled_feat = False
            feat_for_tracking = feat
            tqdm.write(
                f"  {vid_name}: whole-video pre-upsample ran out of memory; "
                "falling back to per-frame upsample."
            )
    else:
        feat_for_tracking = feat
        tqdm.write(
            f"  {vid_name}: skip whole-video pre-upsample to avoid high memory use; "
            "falling back to per-frame upsample."
        )

    N_queries = query_points_arr.shape[0]
    with torch.no_grad():
        for n in tqdm(range(N_queries), desc=f'  Tracking {vid_name}', leave=False):
            t_q = int(query_points_arr[n, 0])
            y_q = float(query_points_arr[n, 1])
            x_q = float(query_points_arr[n, 2])

            trajectories, occlusions = track_point(
                spatial_feat=feat_for_tracking,
                query_frame=t_q,
                query_y=y_q,
                query_x=x_q,
                img_H=img_H,
                img_W=img_W,
                already_upsample=use_preupsampled_feat,
            )
            pred_xy = np.array([[px, py] for py, px in trajectories], dtype=np.float32)
            all_pred_tracks.append(pred_xy)
            pred_occ = np.array(occlusions, dtype=np.bool_)
            all_pred_occlusions.append(pred_occ)

    pred_tracks_arr = np.stack(all_pred_tracks, axis=0)
    pred_occ_arr = np.stack(all_pred_occlusions, axis=0)

    # compute metrics — add batch dim=1
    metrics = compute_tapvid_metrics(
        query_points=query_points_arr[np.newaxis],
        gt_occluded=occluded_query[np.newaxis],
        gt_tracks=gt_tracks_query[np.newaxis],
        pred_occluded=pred_occ_arr[np.newaxis],
        pred_tracks=pred_tracks_arr[np.newaxis],
        query_mode=query_mode,
    )

    trackwise_metrics = compute_tapvid_metrics(
        query_points=query_points_arr[np.newaxis],
        gt_occluded=occluded_query[np.newaxis],
        gt_tracks=gt_tracks_query[np.newaxis],
        pred_occluded=pred_occ_arr[np.newaxis],
        pred_tracks=pred_tracks_arr[np.newaxis],
        query_mode=query_mode,
        get_trackwise_metrics=True,
    )
    track_scores = np.asarray(trackwise_metrics['average_jaccard'][0])
    score_for_sort = np.nan_to_num(track_scores, nan=-np.inf)
    best_indices = list(np.argsort(score_for_sort)[::-1][:4])
    worst_indices = [idx for idx in np.argsort(score_for_sort) if idx not in best_indices][:2]

    for rank, n in enumerate(best_indices, start=1):
        t_q = int(query_points_arr[n, 0])
        y_q = float(query_points_arr[n, 1])
        x_q = float(query_points_arr[n, 2])
        trajectories = [(float(py), float(px)) for px, py in pred_tracks_arr[n]]
        score = float(track_scores[n])
        vis_path = output_dir / f"{vid_name}_best{rank}_track{n:04d}_aj{score:.4f}.mp4"
        visualize_tracking(
            trajectories=trajectories,
            gt_track=gt_tracks_query[n],
            video_tensor=video,
            query_frame=t_q,
            query_y=y_q,
            query_x=x_q,
            save_path=str(vis_path),
        )

    for rank, n in enumerate(worst_indices, start=50):
        t_q = int(query_points_arr[n, 0])
        y_q = float(query_points_arr[n, 1])
        x_q = float(query_points_arr[n, 2])
        trajectories = [(float(py), float(px)) for px, py in pred_tracks_arr[n]]
        score = float(track_scores[n])
        vis_path = output_dir / f"{vid_name}_worst{rank}_track{n:04d}_aj{score:.4f}.mp4"
        visualize_tracking(
            trajectories=trajectories,
            gt_track=gt_tracks_query[n],
            video_tensor=video,
            query_frame=t_q,
            query_y=y_q,
            query_x=x_q,
            save_path=str(vis_path),
        )

    del feat, feat_for_tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k: float(v[0]) for k, v in metrics.items()}, meta


def main(args):
    feat_dir = Path(args.feat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(feat_dir.glob('*.pt'))
    assert len(pt_files) > 0, f"No .pt files found in {feat_dir}"
    print(f"Found {len(pt_files)} videos in {feat_dir}")

    per_video_metrics = {}
    failed_videos = {}
    feat_meta = {}

    for pt_path in tqdm(pt_files, desc='Videos'):
        vid_name = pt_path.stem
        try:
            metrics, meta = evaluate_video(pt_path, output_dir)
            per_video_metrics[vid_name] = metrics
            feat_meta = meta
            tqdm.write(
                f"  {vid_name}: AJ={metrics['average_jaccard']:.4f}  "
                f"<delta={metrics['average_pts_within_thresh']:.4f}  "
                f"OA={metrics['occlusion_accuracy']:.4f}"
            )
        except Exception as exc:
            tqdm.write(f"  ERROR on {vid_name}: {exc}")
            failed_videos[vid_name] = str(exc)

    num_samples = len(per_video_metrics)
    if num_samples == 0:
        raise RuntimeError("No videos were evaluated successfully.")

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
            'num_samples': num_samples,
            'num_pt_files': len(pt_files),
            'num_failed': len(failed_videos),
            'failed_videos': failed_videos,
            'eval_size': feat_meta.get('eval_size'),
            'query_mode': feat_meta.get('query_mode'),
        },
    }

    out_json = output_dir / 'metrics.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n===== Mean metrics across all videos =====")
    for key, value in mean_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nSaved to: {out_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir',   type=str, required=True,
                        help='Directory containing .pt files, e.g. block2_t261/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save metrics.json and tracking videos')
    args = parser.parse_args()
    main(args)
