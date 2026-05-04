import torch
import torch.nn.functional as F
import cv2

def load_features(save_path):
    data = torch.load(save_path, map_location="cpu")
    return data["spatial"], data["temporal"], data

def pixel_to_feat_coord(px, py, orig_h, orig_w, feat_h, feat_w):
    """原始像素坐标 -> 特征图坐标（连续值）"""
    fx = px / orig_w * feat_w
    fy = py / orig_h * feat_h
    return fx, fy

def feat_to_pixel_coord(fx, fy, orig_h, orig_w, feat_h, feat_w):
    """特征图坐标 -> 原始像素坐标"""
    px = fx / feat_w * orig_w
    py = fy / feat_h * orig_h
    return px, py

def sample_feat_at_point(spatial_feat, px, py, orig_h, orig_w):
    """
    在给定像素坐标处双线性插值采样特征向量
    spatial_feat: (Frames, C, feat_h, feat_w)
    px, py: 原始像素坐标
    返回: (Frames, C) 每帧在该位置的特征向量
    """
    frames, c, feat_h, feat_w = spatial_feat.shape
    
    # 转为特征图坐标，归一化到 [-1, 1]（grid_sample 要求）
    fx = px / orig_w * feat_w
    fy = py / orig_h * feat_h
    
    # 归一化
    gx = (fx / (feat_w - 1)) * 2 - 1
    gy = (fy / (feat_h - 1)) * 2 - 1
    
    # grid_sample: (Frames, C, feat_h, feat_w) x grid(Frames, 1, 1, 2)
    grid = torch.tensor([gx, gy], dtype=torch.float32)
    grid = grid.view(1, 1, 1, 2).expand(frames, 1, 1, 2)
    
    sampled = F.grid_sample(
        spatial_feat.float(), grid, 
        mode='bilinear', align_corners=True
    )  # (Frames, C, 1, 1)
    
    return sampled.squeeze(-1).squeeze(-1)  # (Frames, C)

def track_point(spatial_feat, query_px, query_py, query_frame_idx, orig_h, orig_w):
    """
    给定第 query_frame_idx 帧的像素点 (query_px, query_py)
    在所有帧中找到最相似的位置
    
    spatial_feat: (1, Frames, C, feat_h, feat_w)
    返回: 每帧的预测像素坐标 (Frames, 2)
    """
    # (Frames, C, feat_h, feat_w)
    feat = spatial_feat.squeeze(0).float()
    frames, c, feat_h, feat_w = feat.shape
    
    # 1. 采样 query 点的特征向量
    query_feat = sample_feat_at_point(
        feat, query_px, query_py, orig_h, orig_w
    )  # (Frames, C)
    query_vec = query_feat[query_frame_idx]  # (C,) 只取 query 帧的特征
    
    # 2. 在每帧特征图上做全局相似度搜索
    # feat: (Frames, C, feat_h, feat_w)
    # 展平空间维度: (Frames, C, feat_h*feat_w)
    feat_flat = feat.view(frames, c, -1)
    
    # 归一化（余弦相似度）
    query_norm = F.normalize(query_vec.unsqueeze(0), dim=-1)        # (1, C)
    feat_norm = F.normalize(feat_flat, dim=1)                        # (Frames, C, H*W)
    
    # 相似度: (Frames, H*W)
    similarity = torch.einsum('nc,nch->nh', 
                               query_norm.expand(frames, -1), 
                               feat_norm)
    
    # 3. 取最大相似度位置
    best_idx = similarity.argmax(dim=-1)  # (Frames,)
    
    # 4. 还原为像素坐标
    best_fy = (best_idx // feat_w).float()
    best_fx = (best_idx  % feat_w).float()
    
    best_px, best_py = feat_to_pixel_coord(best_fx, best_fy, orig_h, orig_w, feat_h, feat_w)
    
    # (Frames, 2) -> 每帧的 (x, y)
    tracks = torch.stack([best_px, best_py], dim=-1)
    return tracks

def visualize_tracking_on_video(
    video_path,
    tracks,                    # (Frames, 2) 每帧的 (x, y) 像素坐标
    query_frame_idx=0,
    query_point=None,          # (px, py) 查询点原始坐标
    output_path=None,          # 不指定则自动命名
    dot_radius=6,
    trail_length=10,           # 拖尾长度（显示过去几帧的轨迹）
    color_map=True,            # 是否按时间着色
):
    """
    将 point tracking 结果可视化叠加到原始视频上
    
    tracks: (Frames, 2) tensor，每帧预测的像素坐标 (x, y)
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 输出路径
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"{video_name}_tracked.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    
    # tracks 转为 numpy
    tracks_np = tracks.cpu().numpy()  # (Frames, 2)
    num_track_frames = len(tracks_np)
    
    # 预生成每帧的颜色（按时间从蓝->绿->红渐变）
    def get_color(frame_idx, total):
        t = frame_idx / max(total - 1, 1)
        if t < 0.5:
            # 蓝 -> 绿
            r = 0
            g = int(255 * (t * 2))
            b = int(255 * (1 - t * 2))
        else:
            # 绿 -> 红
            r = int(255 * ((t - 0.5) * 2))
            g = int(255 * (1 - (t - 0.5) * 2))
            b = 0
        return (b, g, r)  # OpenCV BGR
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= num_track_frames:
            break
        
        cx, cy = int(round(tracks_np[frame_idx, 0])), int(round(tracks_np[frame_idx, 1]))
        
        # 1. 绘制拖尾轨迹
        trail_start = max(0, frame_idx - trail_length)
        for t in range(trail_start, frame_idx):
            pt1 = (int(round(tracks_np[t, 0])),     int(round(tracks_np[t, 1])))
            pt2 = (int(round(tracks_np[t + 1, 0])), int(round(tracks_np[t + 1, 1])))
            
            color = get_color(t, num_track_frames) if color_map else (200, 200, 200)
            
            # 拖尾越老越细越透明
            alpha = (t - trail_start) / max(frame_idx - trail_start, 1)
            thickness = max(1, int(3 * alpha))
            
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        
        # 2. 绘制当前帧的点
        current_color = get_color(frame_idx, num_track_frames) if color_map else (0, 255, 0)
        cv2.circle(frame, (cx, cy), dot_radius, (255, 255, 255), -1)           # 白色外圈
        cv2.circle(frame, (cx, cy), dot_radius - 2, current_color, -1)         # 彩色内圈
        
        # 3. 标注 query 帧的原始点
        if query_point is not None and frame_idx == query_frame_idx:
            qx, qy = int(query_point[0]), int(query_point[1])
            cv2.drawMarker(frame, (qx, qy), (0, 255, 255), 
                          cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.putText(frame, "Query", (qx + 10, qy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 4. 左上角显示帧号
        cv2.putText(frame, f"Frame {frame_idx:02d} / {num_track_frames - 1}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 5. 显示当前坐标
        cv2.putText(frame, f"({cx}, {cy})",
                   (cx + dot_radius + 4, cy + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        writer.write(frame)
        frame_idx += 1
    
    cap.release()
    writer.release()
    print(f"可视化视频已保存: {output_path}")
    return output_path

# ==========================================
# 使用示例
# ==========================================

# 读取特征
spatial, temporal, meta = load_features("extracted_features/1_svd.pt")
print(f"Spatial: {spatial.shape}")   # (1, 25, C, feat_h, feat_w)

# 原始视频分辨率（需要和提取时一致）
orig_h, orig_w = 480, 360

# 在第 0 帧指定一个查询点（像素坐标）
query_px, query_py = 300, 180   # 想要追踪的点
query_frame = 0

tracks = track_point(
    spatial, 
    query_px=query_px, 
    query_py=query_py, 
    query_frame_idx=query_frame,
    orig_h=orig_h, 
    orig_w=orig_w
)

print(f"Tracks shape: {tracks.shape}")  # (25, 2)
for i, (x, y) in enumerate(tracks):
    print(f"  Frame {i:02d}: ({x:.1f}, {y:.1f})")

visualize_tracking_on_video(
    video_path="1_read.mp4",
    tracks=tracks,
    query_frame_idx=0,
    query_point=(300, 180),
    output_path="tracked_output.mp4",
    trail_length=8,
    color_map=True,
)