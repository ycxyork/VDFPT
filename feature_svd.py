import torch
import torch.nn as nn
import numpy as np
import gc
import os
import cv2
from typing import Optional, Union, List, Dict, Any
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from PIL import Image
from PIL import Image as PILImage
import math

VERBOSE = True
def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ============================================================
# Temporal Attention Processor
# ============================================================
class TemporalMapProcessor:
    def __init__(self, original_processor, unet_ref):
        self.original_processor = original_processor
        self.unet_ref = unet_ref

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        # hidden_states shape: (B*H*W, F, Dim)
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states) # (B*H*W, F, inner_dim)
        key   = attn.to_k(hidden_states) # (B*H*W, F, inner_dim)

        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # (B*H*W, heads, F, F)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_probs  = attn_scores.softmax(dim=-1)

        self.unet_ref._last_temporal_map = attn_probs.mean(1).detach().cpu()

        return self.original_processor(
            attn, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )


# ============================================================
# MySVDUNet with custom forward for feature extraction
# ============================================================
class MySVDUNet(UNetSpatioTemporalConditionModel):
    # def init_feature_extraction(self, up_ft_block: int = 2):
    #     """
    #     Initialize feature extraction state.
    #     Must be called once after loading weights via from_pretrained().
 
    #     Args:
    #         up_ft_block: Index of the up-sampling block from which to extract features.
    #                      Valid range: [0, len(self.up_blocks) - 1].
    #                      Deeper blocks (lower index) have stronger temporal fusion
    #                      but lower spatial resolution.
    #     """
    #     self._up_ft_block = up_ft_block

        # TemporalMapProcessor registration (disabled for now).
        # ------------------------------------------------------------------
        # self._last_temporal_map = None
        #
        # target_attn = (self.up_blocks[up_ft_block]
        #                    .attentions[-1]
        #                    .temporal_transformer_blocks[0]
        #                    .attn1)
        # original_proc = target_attn.processor
        # target_attn.set_processor(
        #     TemporalMapProcessor(original_proc, self)
        # )
        # print(f"TemporalMapProcessor registered on up_blocks[{up_ft_block}]")
        # ------------------------------------------------------------------

    def forward(
        self,
        sample: torch.Tensor,                # (B, F, 8, h, w)
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor, # (B, 1, dim)
        added_time_ids: torch.Tensor,        # (B, 3)
        up_ft_indices: List[int] = None,
    ):
        """
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
        """
        if up_ft_indices is None:
            raise ValueError("Must specify up_ft_indices for feature extraction")

        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            if isinstance(timestep, float):
                dtype = torch.float32
            else: dtype = torch.int32
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
       
       # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        # SVD: added_time_ids → time_ids embedding
        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb # (B, 1280)

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_frames, dim=0, output_size=encoder_hidden_states.shape[0] * num_frames
        )

        # 2. pre-process: conv_in accept (B*F, C, h, w)
        sample = self.conv_in(sample)

        # 3. down
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            if i > max(up_ft_indices):
                break  # only forward up to the max index we need

            is_final_block = (i == len(self.up_blocks) - 1)
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )

            if i in up_ft_indices:
                # sample shape: (B*F, C, H, W)
                BF, C, H, W = sample.shape
                feat = sample.detach().reshape(batch_size, num_frames, C, H, W)
                up_ft[i] = feat  # (B, F, C, H, W)
                log(f"Captured up_ft at block {i}: {feat.shape}")

        # Temporal map extraction (disabled for now).
        # ------------------------------------------------------------------
        # temporal_map = None
        # if self._last_temporal_map is not None:
        #     BHW, F_map, _ = self._last_temporal_map.shape
        #     HW = BHW // batch_size
        #     H_feat = W_feat = int(HW ** 0.5)  # assumes square feature map
        #     temporal_map = self._last_temporal_map.reshape(
        #         batch_size, H_feat, W_feat, F_map, F_map
        #     )
        # log(f"Captured temporal_map: {temporal_map.shape}")
        # ------------------------------------------------------------------

        return {'spatial': up_ft}


# ============================================================
# OneStep SVD Pipeline for feature extraction
# ============================================================
class OneStepSVDPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        video_tensor: torch.Tensor,   # (F, 3, H, W), [-1,1]
        sigma: float,
        up_ft_indices: List[int],
        num_frames: int = 25,
        ensemble_size: int = 1,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        cin: float = 2.5,
    ):
        """
        Run a single diffusion step through the SVD UNet and return
        intermediate up-block features.
 
        Args:
            video_tensor:      (F, 3, H, W) float tensor in [-1, 1].
            sigma:             EDM noise level at which to extract features
                               Typical range for feature extraction: [0.2, 2.0].
            up_ft_indices:     List of up-block indices to capture.
            num_frames:        Number of frames to process (must be <= F).
            ensemble_size:     Number of independent noisy copies to average over.
            fps:               FPS conditioning value (will be decremented by 1
                               to match SVD training convention).
            motion_bucket_id:  Motion bucket conditioning value.
            noise_aug_strength: Noise augmentation strength for image conditioning.
 
        Returns:
            dict with key 'spatial': {block_idx: (E, F, C, H', W') tensor}
        """
        device = self._execution_device
        real_batch_size = 1
        assert video_tensor.shape[0] >= num_frames, \
            f"video_tensor has {video_tensor.shape[0]} frames, dismatch num_frames={num_frames}"


        log("^^^pipelne step1: Encode input image...")
        # ---- 1. Encode first frame → CLIP image embedding ----
        first_frame_np = ((video_tensor[0] + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        first_frame_pil = PILImage.fromarray(first_frame_np)
        image_embeddings = self._encode_image(
            image=first_frame_pil,
            device=device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
        ) # (1, 1, 1024)
        image_embeddings = image_embeddings.repeat(ensemble_size, 1, 1)  # (E, 1, 1024)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        fps = fps - 1


        log("^^^pipelne step2: Encode input image to latents...")
        # ---- 2. Encode first frame → VAE latent (image conditioning) ----
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        first_frame_processed = video_tensor[0:1].to(device=device, dtype=self.vae.dtype) # (1, 3, H, W), [-1,1]

        conditioning_noise    = torch.randn_like(first_frame_processed)
        first_frame_processed = first_frame_processed + noise_aug_strength * conditioning_noise

        image_latents = self._encode_vae_image(
            image=first_frame_processed,
            device=device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
        )  # (1, 4, h, w)，mode() 输出，不含 scaling_factor
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # expand to (E, F, 4, h, w): same conditioning image for every ensemble member and every frame
        # (1, 4, h, w) → (1, F, 4, h, w) → (E, F, 4, h, w)
        image_latents = image_latents.unsqueeze(1).repeat(ensemble_size, num_frames, 1, 1, 1)


        log("^^^pipelne step3: Prepare latents and noise...")
        # ---- 3. Encode all video frames → VAE latents ----
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_frames = video_tensor[:num_frames].to(device=device, dtype=self.vae.dtype)  # (F, 3, H, W), [-1,1]

        latents = self.vae.encode(video_frames).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # (F, 4, h, w)
        latents = latents.to(image_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # ensemble repeat: (F, 4, h, w) → (E, F, 4, h, w)
        latents = latents.unsqueeze(0).repeat(ensemble_size, 1, 1, 1, 1)

        _, _, C, h, w = latents.shape

        # each ensemble add independent noise to the latents
        diffusion_noise = torch.randn(latents.shape, device=device, dtype=torch.float32).to(latents.dtype) # (E, F, 4, h, w)

        sigma_tensor  = torch.tensor(sigma, device=device, dtype=latents.dtype)
        noisy_latents = latents + sigma_tensor * diffusion_noise  # (E, F, 4, h, w)

        c_noise = 0.25 * math.log(sigma)
        timesteps_vec = torch.full((ensemble_size,), c_noise, device=device, dtype=torch.float32)
        #timesteps_vec = torch.full((ensemble_size,), sigma, device=device, dtype=torch.float32) # (E,)

        log("^^^pipelne step4: Concatenate image_latents over channels dimension...")
        # ---- 4. Concatenate image_latents over channels dimension ----
        #c_in = 1.0 / ((sigma ** 2 + 0.5 ** 2) ** 0.5) #c_in = 1.0 / ((sigma ** 2 + 1) ** 0.5)
        c_in = 2.5
        scaled_noisy_latents = noisy_latents * c_in
        unet_input = torch.cat([scaled_noisy_latents, image_latents], dim=2)  # (E, F, 8, h, w)


        log("^^^pipelne step5: Get added time ids...")
        # ---- 5. Get Added Time IDs ----
        added_time_ids = self._get_add_time_ids(
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            dtype=image_embeddings.dtype,
            batch_size=1,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
        )  # (1, 3)
        added_time_ids = added_time_ids.repeat(ensemble_size, 1).to(device)  # (E, 3)


        log("^^^pipelne step6: Forward through UNet...")
        # ---- 6. UNet forward ----
        output = self.unet(
            unet_input,                              # (E, F, 8, h, w)
            timesteps_vec,                           # (E,)
            encoder_hidden_states=image_embeddings, # (E, 1, 1024)
            added_time_ids=added_time_ids,           # (E, 3)
            up_ft_indices=up_ft_indices,
        )

        return output


# ============================================================
# SVDFeaturizer
# ============================================================
class SVDFeaturizer:
    def __init__(self, svd_id='stabilityai/stable-video-diffusion-img2vid-xt'):
        unet = MySVDUNet.from_pretrained(svd_id, subfolder='unet')
        unet.to(dtype=torch.float16)

        pipe = OneStepSVDPipeline.from_pretrained(svd_id, unet=unet)
        pipe.vae.decoder = None
        pipe.scheduler = EulerDiscreteScheduler.from_pretrained(svd_id, subfolder='scheduler')
        gc.collect()
        pipe = pipe.to('cuda')
        pipe.enable_attention_slicing()
        self.pipe = pipe

    @torch.no_grad()
    def forward(
        self,
        video_tensor: torch.Tensor,  # (F, 3, H, W), [-1, 1]
        sigma: float,
        up_ft_index: int = 2,
        ensemble_size: int = 4,
        num_frames: int = 25,
    ):
        """
        Args:
            video_tensor
            t
            up_ft_index
            ensemble_size
            num_frames

        Returns:
            spatial_feat:  (F, C, H', W')
        """
        output = self.pipe(
            video_tensor=video_tensor,
            sigma=sigma,
            up_ft_indices=[up_ft_index],
            num_frames=num_frames,
            ensemble_size=ensemble_size,
        )

        spatial_feat = output['spatial'][up_ft_index]  # (E, F, C, H, W)
        #temporal_map = output['temporal']             # (E, H, W, F, F)
        spatial_feat = spatial_feat.mean(0)  # (F, C, H, W)

        return spatial_feat

def load_video_frames(video_path, num_frames=25, height=512, width=512, t=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if t is None:
        t = total_frames - 1
    assert t < total_frames, \
        f"t={t} exceeds total frames {total_frames}"
    assert t >= num_frames - 1, \
        f"Range [0, {t}] has only {t+1} frames, less than num_frames={num_frames}"

    indices = set(np.linspace(0, t, num_frames, dtype=int))
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

    assert len(frames) == num_frames, \
        f"Expected {num_frames} frames, got {len(frames)}"
    
    # ---- Save sampled frames as video ----
    base, ext = os.path.splitext(video_path)
    save_path = f"{base}_read.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_out = 25
    out = cv2.VideoWriter(save_path, fourcc, fps_out, (width, height))
    for frame in frames:
        # frames are RGB, cv2 expects BGR
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    log(f"Saved sampled frames to: {save_path}")

    # (F, H, W, 3) → (F, 3, H, W), normalize to [-1, 1]
    pixel_values = (torch.tensor(np.array(frames))
                        .permute(0, 3, 1, 2)
                        .float() / 127.5 - 1.0)
    return pixel_values

if __name__ == '__main__':
    video_path = 'eval_videos/5.mp4'

    print("---Initialize SVDFeaturizer...---")
    featurizer = SVDFeaturizer()

    print("\n---Loading Video...---")
    video_tensor = load_video_frames(video_path, num_frames=25, height=512, width=512, t=25)
    print(f"video_tensor shape: {video_tensor.shape}")

    print("\n---Extracting Features...---")
    spatial_feat = featurizer.forward(
        video_tensor=video_tensor,
        t=261,
        up_ft_index=2,
        ensemble_size=4,
        num_frames=25,
    )

    print("\n---Finished Extracting...---")
    print(f"Spatial:  {spatial_feat.shape}")   # (1, 25, 640, 64, 64)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = "extracted_features"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{video_name}_svd.pt")

    torch.save({
        "spatial": spatial_feat,         # (1, Frames, C, H, W)
        "video_path": video_path,
    }, save_path)