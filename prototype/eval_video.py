import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
import json
from PIL import Image
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import queue

def load_video_frames(video_path, num_frames=25, height=512, width=512, t=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if t is None:
        t = total_frames - 1
    assert t < total_frames, f"t={t} exceeds total frames {total_frames}"
    assert t >= num_frames - 1, f"Range [0, {t}] has only {t+1} frames, less than num_frames={num_frames}"

    indices = set(np.linspace(0, t, num_frames, dtype=int))
    print(f"Total frames: {total_frames}, sampling {num_frames} frames from [0, {t}]")
    print(f"Sampled indices: {sorted(indices)}")

    frames = []
    for i in range(t + 1):
        ret, frame = cap.read()
        assert ret, f"Frame {i} could not be read"
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        if len(frames) == num_frames:
            break
    cap.release()

    assert len(frames) == num_frames, f"Expected {num_frames} frames, got {len(frames)}"

    base, ext = os.path.splitext(video_path)
    save_path = f"{base}_read.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 25, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved sampled frames to: {save_path}")

    pixel_values = (torch.tensor(np.array(frames))
                        .permute(0, 3, 1, 2)
                        .float() / 127.5 - 1.0)
    return pixel_values

def track_point(
    spatial_feat: torch.Tensor,       # (F, C, feat_H, feat_W)
    query_frame: int,                 # (0 ~ F-1)
    query_y: float,                   # (0 ~ img_H-1)
    query_x: float,                   # (0 ~ img_W-1)
    img_H: int = 512,
    img_W: int = 512,
    local_window: int = None,
    use_soft_argmax: bool = False,
    temperature: float = 0.05,
    top_k: int = 5
):
    """
    Returns:
        trajectories: list of (y, x)
        vis_maps: list of similarity maps
    """
    F_frames, C, feat_H, feat_W = spatial_feat.shape
    dtype = spatial_feat.dtype
    device = spatial_feat.device
    
    trajectories = [(0, 0)] * F_frames
    vis_maps = [None] * F_frames

    # Extract query feature vector
    query_feat_low = spatial_feat[query_frame:query_frame+1] # (1, C, feat_H, feat_W)
    query_feat_up = F.interpolate(query_feat_low, size=(img_H, img_W), mode='bilinear', align_corners=False) # (1, C, H, W)
    
    norm_x = (query_x + 0.5) / img_W * 2 - 1.0
    norm_y = (query_y + 0.5) / img_H * 2 - 1.0
    query_coord = torch.tensor([[[[norm_x, norm_y]]]], dtype=dtype, device=device)
    
    query_vec = F.grid_sample(query_feat_up, query_coord, mode='bilinear', align_corners=False)
    query_vec = query_vec.view(1, C)
    query_vec = F.normalize(query_vec, p=2, dim=1) # (1, C)

    def process_frame(f_idx, ref_y, ref_x):
        """
        ref_y, ref_x: center of local window
        """
        target_feat_low = spatial_feat[f_idx:f_idx+1] 
        target_feat_up = F.interpolate(target_feat_low, size=(img_H, img_W), mode='bilinear', align_corners=False)
        
        target_feat_flat = target_feat_up.view(C, -1).transpose(0, 1) 
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
            
            tk_y = (tk_idx // img_W).float()
            tk_x = (tk_idx % img_W).float()
            
            pred_y = torch.sum(weights * tk_y).item()
            pred_x = torch.sum(weights * tk_x).item()
        
        return pred_y, pred_x, sim_map
    
    pred_y, pred_x, sim_map = process_frame(query_frame, query_y, query_x)
    trajectories[query_frame] = (pred_y, pred_x)
    vis_maps[query_frame] = sim_map.detach().cpu().numpy()
    
    curr_y, curr_x = trajectories[query_frame]
    for f_idx in range(query_frame + 1, F_frames):
        curr_y, curr_x, sim_map = process_frame(f_idx, curr_y, curr_x)
        trajectories[f_idx] = (curr_y, curr_x)
        vis_maps[f_idx] = sim_map.detach().cpu().numpy()
        
    curr_y, curr_x = trajectories[query_frame]
    for f_idx in range(query_frame - 1, -1, -1):
        curr_y, curr_x, sim_map = process_frame(f_idx, curr_y, curr_x)
        trajectories[f_idx] = (curr_y, curr_x)
        vis_maps[f_idx] = sim_map.detach().cpu().numpy()
    
    return trajectories, vis_maps

def visualize_tracking(
    trajectories, 
    vis_maps, 
    video_tensor,       # (F, 3, H, W)
    query_frame: int, 
    query_y: float, 
    query_x: float,
    save_path: str = 'tracking_video.png', 
    img_H: int = 512, 
    img_W: int = 512,
):
    F_num = len(trajectories)
    cols = 5
    rows = (F_num + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3+3, rows * 6))
    
    if rows * 2 == 1 or cols == 1:
        axes = np.atleast_2d(axes)

    for t in range(F_num):
        row = (t // cols) * 2
        col = t % cols

        pred_y, pred_x = trajectories[t]

        frame = ((video_tensor[t] + 1.0) / 2.0 * 255.0).clamp(0, 255)
        frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Top: RGB Frame + Predicted Point
        ax_frame = axes[row, col]
        ax_frame.imshow(frame)
        
        ax_frame.scatter(pred_x, pred_y, c='lime', s=10, zorder=5)
        if t == query_frame:
            ax_frame.scatter(query_x, query_y, c='red', s=10, zorder=5)
            
        ax_frame.set_title(f"Frame {t}", fontsize=10)
        ax_frame.axis('off')

        # Bottom: Soft Label (Similarity) Map
        ax_sim = axes[row + 1, col]
        sim_map = vis_maps[t]  # (img_H, img_W)
        
        if torch.is_tensor(sim_map):
            sim_display = sim_map.cpu().numpy()
        else:
            sim_display = np.array(sim_map)

        vmax = max(sim_display.max(), 1e-6)
        im = ax_sim.imshow(sim_display, cmap='hot', vmin=0, vmax=vmax)
        
        ax_sim.scatter(pred_x, pred_y, c='lime', s=20, zorder=5, marker='+', linewidths=1.5)
        if t == query_frame:
            ax_sim.scatter(query_x, query_y, c='cyan', s=20, zorder=5, marker='+', linewidths=1.5)
            
        ax_sim.set_title(f"Sim Map max={sim_display.max():.4f}", fontsize=8)
        ax_sim.axis('off')

        divider = make_axes_locatable(ax_sim)
        cax = divider.append_axes('right', size='5%', pad=0.03)
        plt.colorbar(im, cax=cax)

    for t in range(F_num, cols * rows):
        row = (t // cols) * 2
        col = t % cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')

    plt.suptitle(
        f'Point Tracking | query: frame={query_frame} y={query_y} x={query_x}',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tracking visualization to: {save_path}")


# for name in ['1', '2', '3', '4', '5']:
#     video_path = f'eval_videos/{name}.mp4'
#     feat_path = f'extracted_features/{name}_svd.pt'
#     print(f"Processing video: {video_path} with features: {feat_path}")
# ---- config ----
video_path   = 'eval_videos/2.mp4'
feat_path    = 'extracted_features/2_svd.pt'
NUM_FRAMES   = 25
IMG_H        = 512
IMG_W        = 512
QUERY_FRAME  = 0
QUERY_Y      = 370
QUERY_X      = 260
video_name = os.path.splitext(os.path.basename(video_path))[0]
# ---- load video ----
video_tensor = load_video_frames(video_path, num_frames=NUM_FRAMES, height=IMG_H, width=IMG_W, t=25)
# (F, 3, H, W), [-1, 1]

# ---- load features ----
spatial_feat = torch.load(feat_path)['spatial']  # (F, C, feat_H, feat_W)
print(f"Loaded features: {spatial_feat.shape}")

# ---- track ----
traj, vis_maps = track_point(
    spatial_feat=spatial_feat,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    img_H=IMG_H,
    img_W=IMG_W,
    #local_window=50,
    # n_last_frames=7,
    # temperature=1.0,
    # topk=5,
    # size_mask_neighborhood=3,  # on 32×32 feat map, radius=3 covers ~96px in image
)
print("Trajectory:", traj)
visualize_tracking(
    trajectories=traj,
    vis_maps=vis_maps,
    video_tensor=video_tensor,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    save_path=f'tracking_{video_name}.png',
    img_H=IMG_H,
    img_W=IMG_W,
)
traj, vis_maps = track_point(
    spatial_feat=spatial_feat,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    img_H=IMG_H,
    img_W=IMG_W,
    local_window=50
)
visualize_tracking(
    trajectories=traj,
    vis_maps=vis_maps,
    video_tensor=video_tensor,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    save_path=f'tracking_{video_name}_local50.png',
    img_H=IMG_H,
    img_W=IMG_W,
)
traj, vis_maps = track_point(
    spatial_feat=spatial_feat,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    img_H=IMG_H,
    img_W=IMG_W,
    local_window=50,
    use_soft_argmax=True
)
visualize_tracking(
    trajectories=traj,
    vis_maps=vis_maps,
    video_tensor=video_tensor,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    save_path=f'tracking_{video_name}_local50_topk.png',
    img_H=IMG_H,
    img_W=IMG_W,
)
