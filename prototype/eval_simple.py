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

def track_point_spair_style(
    spatial_feat,        # (F, C, feat_H, feat_W)
    query_frame: int,    # 源帧索引
    query_y: int,        # 源帧关键点 y（原图像素坐标）
    query_x: int,        # 源帧关键点 x
    img_H: int = 512,
    img_W: int = 512,
):
    """
    对齐 eval_spair 的全局余弦相似度匹配
    在所有帧上追踪 query_frame 中的 (query_y, query_x) 点
    """
    F_num, C, feat_H, feat_W = spatial_feat.shape

    # 上采样到原图分辨率（对齐 eval_spair 的 nn.Upsample）
    feat_up = F.interpolate(
        spatial_feat, size=(img_H, img_W),
        mode='bilinear', align_corners=False
    )  # (F, C, img_H, img_W)

    # 源帧特征向量（对齐 eval_spair 的 src_vec）
    src_vec = feat_up[query_frame, :, query_y, query_x].view(1, C)  # (1, C)
    src_vec = F.normalize(src_vec, dim=1).transpose(0, 1)            # (C, 1)

    results = []
    for t in range(F_num):
        # 目标帧（对齐 eval_spair 的 trg_vec）
        trg_vec = feat_up[t].view(C, -1).transpose(0, 1)  # (HW, C)
        trg_vec = F.normalize(trg_vec, dim=1)               # (HW, C)

        # 全局余弦相似度（完全对齐 eval_spair）
        cos_map = torch.mm(trg_vec, src_vec).view(img_H, img_W).cpu().numpy()

        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)
        pred_y, pred_x = max_yx[0], max_yx[1]
        confidence = cos_map.max()

        results.append({
            'frame':      t,
            'pred_y':     pred_y,
            'pred_x':     pred_x,
            'confidence': float(confidence),
            'cos_map':    cos_map,   # 保留完整相似度图，便于可视化
        })

    return results

def visualize_tracking(results, video_tensor, query_frame, query_y, query_x,
                        save_path='tracking.png', img_H=512, img_W=512):
    """
    Visualize tracking results with cos_map overlay.
    Each frame shows: original frame + predicted point + cos_map heatmap side by side.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    F_num = len(results)
    cols = 5
    rows = (F_num + cols - 1) // cols

    # Two subplots per frame: left=frame+point, right=cos_map
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))

    for t, res in enumerate(results):
        row = (t // cols) * 2
        col = t % cols

        frame = ((video_tensor[t] + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # ---- top row: frame + predicted point ----
        ax_frame = axes[row, col]
        ax_frame.imshow(frame)
        ax_frame.scatter(res['pred_x'], res['pred_y'], c='lime', s=60, zorder=5, label='pred')
        if t == query_frame:
            ax_frame.scatter(query_x, query_y, c='red', s=60, zorder=5, label='query')
        ax_frame.set_title(f"Frame {t}  conf={res['confidence']:.3f}", fontsize=7)
        ax_frame.axis('off')

        # ---- bottom row: cos_map heatmap ----
        ax_cos = axes[row + 1, col]
        cos_map = res['cos_map']
        im = ax_cos.imshow(cos_map, cmap='hot', vmin=cos_map.min(), vmax=cos_map.max())
        # mark predicted position on cos_map
        ax_cos.scatter(res['pred_x'], res['pred_y'], c='lime', s=40, zorder=5,
                       marker='+', linewidths=1.5)
        if t == query_frame:
            ax_cos.scatter(query_x, query_y, c='cyan', s=40, zorder=5,
                           marker='+', linewidths=1.5)
        ax_cos.set_title(f"cos_map [{cos_map.min():.2f}, {cos_map.max():.2f}]", fontsize=6)
        ax_cos.axis('off')

        divider = make_axes_locatable(ax_cos)
        cax = divider.append_axes('right', size='5%', pad=0.03)
        plt.colorbar(im, cax=cax)

    # Hide unused subplots
    for t in range(F_num, cols * rows):
        row = (t // cols) * 2
        col = t % cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')

    plt.suptitle(f'Point Tracking (spair-style) | query: frame={query_frame} y={query_y} x={query_x}',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ---- config ----
VIDEO_PATH   = '1.mp4'
FEAT_PATH    = 'extracted_features/1_svd.pt'
NUM_FRAMES   = 25
IMG_H        = 512
IMG_W        = 512
QUERY_FRAME  = 0
QUERY_Y      = 230
QUERY_X      = 325

# ---- load video ----
video_tensor = load_video_frames(VIDEO_PATH, num_frames=NUM_FRAMES, height=IMG_H, width=IMG_W, t=25)
# (F, 3, H, W), [-1, 1]

# ---- load features ----
spatial_feat = torch.load(FEAT_PATH)['spatial']  # (F, C, feat_H, feat_W)
print(f"Loaded features: {spatial_feat.shape}")

# ---- track ----
results = track_point_spair_style(
    spatial_feat=spatial_feat,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    img_H=IMG_H,
    img_W=IMG_W,
)

# ---- visualize ----
visualize_tracking(
    results=results,
    video_tensor=video_tensor,
    query_frame=QUERY_FRAME,
    query_y=QUERY_Y,
    query_x=QUERY_X,
    save_path='tracking_result.png',
    img_H=IMG_H,
    img_W=IMG_W,
)