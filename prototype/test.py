import torch

def test_pt_file(pt_path: str):
    data = torch.load(pt_path, map_location="cpu")

    print(f"\n=== Testing file: {pt_path} ===")

    # ---------------------------
    # 1. 检查 keys
    # ---------------------------
    required_keys = ["feat", "points", "occluded", "video", "meta"]
    for k in required_keys:
        if k not in data:
            print(f"[ERROR] Missing key: {k}")
        else:
            print(f"[OK] Key exists: {k}")

    # ---------------------------
    # 2. feat 检查
    # ---------------------------
    feat = data.get("feat")
    if feat is not None:
        print("\n[feat]")
        print(" shape:", feat.shape)
        print(" dtype:", feat.dtype)

        if feat.ndim != 4:
            print("[ERROR] feat should be 4D (F, C, H', W')")
        if feat.dtype != torch.float16:
            print("[WARN] feat dtype is not float16")

    # ---------------------------
    # 3. points 检查
    # ---------------------------
    points = data.get("points")
    if points is not None:
        print("\n[points]")
        print(" shape:", points.shape)
        print(" dtype:", points.dtype)

        if points.ndim != 3 or points.shape[-1] != 2:
            print("[ERROR] points should be (N, F, 2)")

        # normalize 检查
        min_val = points.min().item()
        max_val = points.max().item()

        print(f" value range: [{min_val:.4f}, {max_val:.4f}]")

        if min_val < 0 or max_val > 1:
            print("[ERROR] points NOT normalized to [0, 1]")
        else:
            print("[OK] points normalized")

    # ---------------------------
    # 4. occluded 检查
    # ---------------------------
    occluded = data.get("occluded")
    if occluded is not None:
        print("\n[occluded]")
        print(" shape:", occluded.shape)
        print(" dtype:", occluded.dtype)

        if occluded.dtype != torch.bool:
            print("[ERROR] occluded should be bool tensor")

    # ---------------------------
    # 5. video 检查
    # ---------------------------
    video = data.get("video")
    if video is not None:
        print("\n[video]")
        print(" shape:", video.shape)
        print(" dtype:", video.dtype)

        if video.ndim != 4 or video.shape[-1] != 3:
            print("[ERROR] video should be (F, H, W, 3)")

        if video.dtype != torch.uint8:
            print("[WARN] video is not uint8")

        min_val = video.min().item()
        max_val = video.max().item()
        print(f" value range: [{min_val}, {max_val}]")

        if min_val < 0 or max_val > 255:
            print("[ERROR] video values out of range [0,255]")

    # ---------------------------
    # 6. meta 检查
    # ---------------------------
    meta = data.get("meta")
    if meta is not None:
        print("\n[meta]")
        print(" type:", type(meta))
        if not isinstance(meta, dict):
            print("[ERROR] meta should be a dict")

    print("\n=== Done ===\n")
import pickle

def inspect_structure(obj, indent=0, max_depth=3):
    """递归打印对象结构"""
    prefix = "  " * indent
    
    if indent > max_depth:
        print(prefix + "...")
        return
    
    if isinstance(obj, dict):
        print(prefix + f"dict with {len(obj)} keys")
        for k in list(obj.keys())[:10]:  # 最多看前10个key
            print(prefix + f"  key: {k}")
            inspect_structure(obj[k], indent + 2, max_depth)
    
    elif isinstance(obj, list):
        print(prefix + f"list with length {len(obj)}")
        if len(obj) > 0:
            print(prefix + "  first element:")
            inspect_structure(obj[0], indent + 2, max_depth)
    
    elif hasattr(obj, "shape"):  # numpy / torch tensor
        print(prefix + f"{type(obj)} with shape {obj.shape}")
    
    else:
        print(prefix + f"{type(obj)}: {str(obj)[:100]}")
        

def inspect_pkl(path, max_depth=3):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print("=== Top-level structure ===")
    inspect_structure(data, max_depth=max_depth)
    
    return data

test_pt_file("/content/output/kubric_first/kubric_0000.pt")

# import pickle
# import numpy as np

# def check_points_normalization(pkl_path, name="dataset"):
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)

#     print(f"\n=== Checking {name} ===")

#     # 统一拿到 sample 列表
#     if isinstance(data, dict):
#         samples = list(data.values())
#         print(f"Format: dict with {len(samples)} samples")
#     elif isinstance(data, list):
#         samples = data
#         print(f"Format: list with {len(samples)} samples")
#     else:
#         raise ValueError("Unknown data format")

#     all_min = []
#     all_max = []

#     for i, sample in enumerate(samples[:10]):  # 只检查前10个，加快速度
#         pts = sample['points']  # (N, T, 2)

#         all_min.append(pts.min())
#         all_max.append(pts.max())

#         print(f"Sample {i}: min={pts.min():.4f}, max={pts.max():.4f}")

#     global_min = float(np.min(all_min))
#     global_max = float(np.max(all_max))

#     print("\n--- Summary ---")
#     print(f"Global min: {global_min:.4f}")
#     print(f"Global max: {global_max:.4f}")

#     # 自动判断
#     if global_max <= 1.5 and global_min >= -0.5:
#         print("✅ Likely NORMALIZED (range ~ [0, 1])")
#     elif global_max > 10:
#         print("❗ Likely PIXEL coordinates (not normalized)")
#     else:
#         print("⚠️ Ambiguous range, need manual check")

#     return global_min, global_max

