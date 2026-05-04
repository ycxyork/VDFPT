import torch
import cv2
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from diffusers import StableVideoDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.models.attention_processor import Attention


def load_video_frames(video_path, num_frames=50, height=512, width=512):
    """读取视频，采样并调整大小"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        raise ValueError(f"视频帧数 ({total_frames}) 少于要求的帧数 ({num_frames})")

    indices = list(range(num_frames))
    frames = []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        if len(frames) >= num_frames: break
            
    cap.release()
    
    # 归一化到 [-1, 1] 供 VAE 使用
    # Shape: (Frames, H, W, C) -> (Frames, C, H, W)
    pixel_values = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return pixel_values


# ==========================================
# 1. 定义一个自定义的 Attention Processor
#    用来拦截 Temporal Attention 的权重矩阵
# ==========================================
class TemporalMapProcessor:
    def __init__(self, original_processor, target_storage_list):
        self.original_processor = original_processor
        self.target_storage_list = target_storage_list  # 保存列表的引用

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # SVD Temporal Attn 是 Self-Attention, hidden_states shape: (Batch*HW, Frames, Dim)
        
        batch_size, sequence_length, _ = hidden_states.shape
        # 手动计算 Q 和 K 来获取 Attention Map
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Matrix Multiplication: (Batch*HW, Heads, Frames, Frames)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_probs = attn_scores.softmax(dim=-1)
        
        # [关键] 保存 Attention Map (取 Head 平均)
        # Shape: (Batch*HW, Frames, Frames) -> 例如 (1024, 14, 14)
        self.target_storage_list.append(attn_probs.mean(1).detach().cpu())
        
        # 继续原本的计算
        return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

# ==========================================
# 2. 主类：SVD 特征提取器
# ==========================================
class SVDFeatureExtractor:
    def __init__(self, pipe):
        self.pipe = pipe
        self.unet = pipe.unet

        self.feats = {
            "spatial": [],
            "temporal": []
        }
        
        self.hook_handles = [] 
        self.original_processors = {}
    
    def print_info(self):
        print("UNet Structure:")
        print(self.unet.up_blocks[2].attentions[2]) # 打印 UNet 的 Up-blocks 结构，确认层级

    def clean_hooks(self):
        """清理钩子和恢复 Processor，防止显存泄漏或影响下一次推理"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        for layer, proc in self.original_processors.items():
            layer.set_processor(proc)
        self.original_processors = {}
        
        self.feats["spatial"] = []
        self.feats["temporal"] = []

    def spatial_hook(self, module, input, output):
        # output shape: (Batch * Frames, 640, H, W)
        feat = output[0] if isinstance(output, tuple) else output
        self.feats["spatial"].append(feat.detach().cpu())

    def register_hooks(self):
        """注册所有钩子"""
        self.clean_hooks()

        target_block = self.unet.up_blocks[2]
    
        # A. 注册 Spatial Hook
        # 对应你 log 里的 (0-2): 3 x Transformer... 我们取索引 2
        handle = target_block.attentions[2].register_forward_hook(self.spatial_hook)
        self.hook_handles.append(handle)
        
        # B. 注册 Temporal Hook
        # 深入结构: attentions[2] -> temporal_transformer_blocks[0] -> attn1
        target_attn_layer = target_block.attentions[2].temporal_transformer_blocks[0].attn1
        
        # 替换 Processor
        original_proc = target_attn_layer.processor
        self.original_processors[target_attn_layer] = original_proc

        target_attn_layer.set_processor(
            TemporalMapProcessor(original_proc, self.feats["temporal"])
        )
        
        print("Hooks registered successfully!")

    def get_features(self, video_frames, decode_chunk_size=8):
        # 运行 Pipeline (此处简化，实际需要调用 pipe 的 __call__ 或 encode 逻辑)
        # 假设 video_frames 已经处理好
        
        # 这是一个伪代码调用，你需要按照 SVD Pipeline 的标准输入传参
        # 重点是：一旦 pipe(image=...) 运行，hooks 就会自动触发
        with torch.no_grad():
             # 使用 pipeline 正常推理 (可以只加噪一步，类似 DIFT)
            self.pipe(video_frames, decode_chunk_size=decode_chunk_size, num_inference_steps=1)
            
        return 

@torch.no_grad()
def run_svd_feature_extraction(video_path, extractor, target_t=261):
    device = extractor.pipe.device
    dtype = extractor.pipe.unet.dtype
    
    print(f"1. 读取视频: {video_path}")
    # 1. 准备视频 Tensor
    # (Frames, C, H, W)
    video_tensor = load_video_frames(video_path)
    video_tensor = video_tensor.to(device, dtype=dtype)
    num_frames, _, height, width = video_tensor.shape
    
    print("2. VAE 编码 (将视频像素压缩为 Latents)")
    # Input: (Frames, 3, H, W) -> VAE -> (Frames, 4, H/8, W/8)
    latents = extractor.pipe.vae.encode(video_tensor).latent_dist.sample()
    latents = latents * extractor.pipe.vae.config.scaling_factor
    
    # 调整为 SVD 输入格式: (Batch, Frames, Channels, H, W)
    # Batch=1
    latents = latents.unsqueeze(0) 
    batch_size = latents.shape[0]

    print(f"   Latent Shape: {latents.shape}")

    print("3. 准备条件 (Conditioning)")
    # SVD 需要第一帧作为 Image Embedding (Context)
    # 我们取视频的第一帧，重新预处理为 CLIP 的输入格式
    # CLIP 期望是 224x224 左右，pipeline 内部有 feature_extractor
    
    # 从 tensor 转回 PIL 给 pipeline 的 image_processor 用 (稍微绕一下为了安全)
    # 也可以直接插值 resize
    first_frame_tensor = (video_tensor[0] + 1.0) / 2.0  # [-1,1] -> [0,1]
    first_frame_pil = Image.fromarray((first_frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    
    # 获取 image embedding
    image_embeddings = extractor.pipe._encode_image(
        image=first_frame_pil, device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False
    )
    print(f"image_embeddings shape: {image_embeddings.shape}")

    print(f"4. 注入噪声 (t={target_t})")
    # 生成随机噪声
    noise = torch.randn_like(latents)
    # 获取 Timestep tensor
    timesteps = torch.tensor([target_t] * latents.shape[0], device=latents.device)
    
    # 加噪: Latent_t = Latent_0 * alpha + Noise * sigma
    noisy_latents = extractor.pipe.scheduler.add_noise(latents, noise, timesteps)

    print("5. 构建 Time IDs (SVD 特有参数)")
    first_frame = video_tensor[0:1]  # (1, 3, H, W)
    condition_latent = extractor.pipe.vae.encode(first_frame).latent_dist.sample()
    condition_latent = condition_latent * extractor.pipe.vae.config.scaling_factor
    noise_aug = torch.randn_like(condition_latent) * 0.02
    condition_latent = condition_latent + noise_aug
    # (1, 4, H/8, W/8) -> (1, Frames, 4, H/8, W/8)
    condition_latent = condition_latent.unsqueeze(0).repeat(1, num_frames, 1, 1, 1)

    # 拼接通道: (1, Frames, 8, H/8, W/8)
    unet_input = torch.cat([noisy_latents, condition_latent], dim=2)
    print(f"   UNet input shape: {unet_input.shape}")
    # SVD UNet 需要 added_time_ids (fps, motion_bucket_id, noise_aug_strength)
    # 我们需要手动构造这些参数
    def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        passed_add_time_ids = passed_add_time_ids.repeat(batch_size, 1)
        return passed_add_time_ids

    # 使用默认参数
    added_time_ids = _get_add_time_ids(
        fps=7, 
        motion_bucket_id=127, 
        noise_aug_strength=0.02, 
        dtype=dtype, 
        batch_size=batch_size
    )

    print("6. UNet 前向传播 & 特征抓取")
    # 注册 Hook
    extractor.register_hooks()
    
    # 手动调用 UNet
    extractor.pipe.unet(
        unet_input,
        timesteps,
        encoder_hidden_states=image_embeddings,
        added_time_ids=added_time_ids,
        return_dict=False,
    )
    
    # 获取结果
    spatial_feat = extractor.feats["spatial"][0]
    temporal_feat = extractor.feats["temporal"][0]
    
    extractor.clean_hooks()
    
    return spatial_feat, temporal_feat


if __name__ == "__main__":

    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.vae.decoder = None
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    extractor = SVDFeatureExtractor(pipe)

    video_path = "1.mp4"
    try:
        spatial, temporal = run_svd_feature_extraction(
            video_path, extractor, target_t=261
        )

        print("\n=== 提取完成 ===")
        # Spatial: (Batch * Frames, C, H, W)
        # Reshape to (Batch, Frames, C, H, W)
        b_f, c, h, w = spatial.shape
        num_frames = 50
        spatial = spatial.reshape(1, num_frames, c, h, w)
        print(f"Spatial Features: {spatial.shape} (用于 DIFT 匹配)")

        # Temporal: (Batch * H * W, Frames, Frames)
        # Reshape to (Batch, H, W, F, F)
        seq_len, f1, f2 = temporal.shape
        spatial_h, spatial_w = spatial.shape[-2], spatial.shape[-1]
        temporal = temporal.reshape(-1, spatial_h, spatial_w, f1, f2)
        print(f"Temporal Map:     {temporal.shape} (用于运动预测)")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = "extracted_features"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{video_name}.pt")

        torch.save({
            "spatial": spatial,         # (1, Frames, C, H, W)
            "temporal": temporal,       # (1, H, W, Frames, Frames)
            "video_path": video_path,
            "target_t": 261,
            "num_frames": num_frames,
        }, save_path)

    except Exception as e:
        print(f"Error: {e}")
