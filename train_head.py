"""Train a TAP-Net-style prediction head from pre-extracted features.

Usage:
python train_head.py \
    --feat_dir /content/output \
    --output_dir checkpoints/head_train
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


HEAD_KWARGS = {
    'hidden_dim': 16,
    'occlusion_dim': 32,
    'softmax_temperature': 10.0,
    'soft_argmax_radius': 5.0,
}

OPTIMIZER_KWARGS = {
    'lr': 2e-3,
    'weight_decay': 1e-2,
    'betas': (0.9, 0.95),
    'eps': 1e-8,
}

POSITION_LOSS_WEIGHT = 0.05
HUBER_DELTA = 4.0


class TAPStylePredictionHead(nn.Module):
    """Small TAP-Net-style prediction head over query-to-video cost volumes."""
    def __init__(
        self,
        hidden_dim: int = 16,
        occlusion_dim: int = 32,
        softmax_temperature: float = 10.0,
        soft_argmax_radius: float = 5.0,
    ):
        super().__init__()
        self.softmax_temperature = softmax_temperature
        self.soft_argmax_radius = soft_argmax_radius

        self.shared = nn.Sequential(
            nn.Conv3d(1, hidden_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.position_head = nn.Conv3d(
            hidden_dim, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.occlusion_conv = nn.Sequential(
            nn.Conv3d(
                hidden_dim,
                occlusion_dim,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.ReLU(inplace=True),
        )
        self.occlusion_mlp = nn.Sequential(
            nn.Linear(occlusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cost_volume: torch.Tensor,
        query_points: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_count, num_frames, feat_h, feat_w = cost_volume.shape
        x = self.shared(cost_volume.unsqueeze(1))

        heatmap_logits = self.position_head(x).squeeze(1)
        heatmap = F.softmax(
            heatmap_logits.flatten(-2) * self.softmax_temperature,
            dim=-1,
        ).view(q_count, num_frames, feat_h, feat_w)
        tracks = self.soft_argmax(heatmap, image_size=image_size)

        query_frames = torch.round(query_points[:, 0]).long()
        query_xy = query_points[:, [2, 1]].to(tracks.dtype)
        tracks[torch.arange(q_count, device=tracks.device), query_frames] = query_xy

        occ_feat = self.occlusion_conv(x).mean(dim=(-2, -1))
        occ_feat = occ_feat.permute(0, 2, 1)
        occlusion_logits = self.occlusion_mlp(occ_feat).squeeze(-1)

        return tracks, occlusion_logits

    def soft_argmax(
        self,
        heatmap: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        _, _, feat_h, feat_w = heatmap.shape
        img_h, img_w = image_size
        device = heatmap.device
        dtype = heatmap.dtype

        y_grid, x_grid = torch.meshgrid(
            torch.arange(feat_h, dtype=dtype, device=device) + 0.5,
            torch.arange(feat_w, dtype=dtype, device=device) + 0.5,
            indexing='ij',
        )
        coords = torch.stack([x_grid, y_grid], dim=-1)

        flat_idx = torch.argmax(heatmap.flatten(-2), dim=-1)
        center_y = (flat_idx // feat_w).to(dtype=dtype) + 0.5
        center_x = (flat_idx % feat_w).to(dtype=dtype) + 0.5

        dist2 = (
            (x_grid[None, None] - center_x[:, :, None, None]) ** 2
            + (y_grid[None, None] - center_y[:, :, None, None]) ** 2
        )
        valid = (dist2 < self.soft_argmax_radius ** 2).to(dtype)
        weights = heatmap * valid
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

        points_grid = torch.sum(weights[..., None] * coords[None, None], dim=(-3, -2))
        points = points_grid.clone()
        points[..., 0] = points_grid[..., 0] * (img_w / feat_w)
        points[..., 1] = points_grid[..., 1] * (img_h / feat_h)
        return points


def compute_cost_volume(
    spatial_feat: torch.Tensor,
    query_points: torch.Tensor,
    img_H: int,
    img_W: int,
) -> torch.Tensor:
    """Build query-to-video cosine cost volumes from feature maps."""
    spatial_feat = F.normalize(spatial_feat.float(), p=2, dim=1)
    query_frames = torch.round(query_points[:, 0]).long()
    query_y = query_points[:, 1]
    query_x = query_points[:, 2]

    nx = query_x / img_W * 2 - 1.0
    ny = query_y / img_H * 2 - 1.0
    coords = torch.stack([nx, ny], dim=-1).view(-1, 1, 1, 2)

    query_frame_feats = spatial_feat[query_frames]
    query_vecs = F.grid_sample(
        query_frame_feats,
        coords,
        mode='bilinear',
        align_corners=False,
    ).view(query_points.shape[0], spatial_feat.shape[1])
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    return torch.einsum('qc,tchw->qthw', query_vecs, spatial_feat)


@torch.no_grad()
def predict_head(
    head: TAPStylePredictionHead,
    spatial_feat: torch.Tensor,
    query_points_arr: np.ndarray,
    img_H: int,
    img_W: int,
    query_chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict tracks and occlusions for all query points with a trained head."""
    all_tracks = []
    all_occluded = []
    device = spatial_feat.device

    for start in range(0, query_points_arr.shape[0], query_chunk_size):
        query_np = query_points_arr[start:start + query_chunk_size]
        query_points = torch.from_numpy(query_np).to(device=device, dtype=torch.float32)
        cost_volume = compute_cost_volume(
            spatial_feat=spatial_feat,
            query_points=query_points,
            img_H=img_H,
            img_W=img_W,
        )
        tracks, occlusion_logits = head(
            cost_volume,
            query_points=query_points,
            image_size=(img_H, img_W),
        )
        all_tracks.append(tracks.detach().cpu().numpy().astype(np.float32))
        all_occluded.append((occlusion_logits.detach().cpu().numpy() > 0).astype(np.bool_))

    return np.concatenate(all_tracks, axis=0), np.concatenate(all_occluded, axis=0)


def load_head_checkpoint(
    head: nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get('model_state_dict')
            or checkpoint.get('state_dict')
            or checkpoint.get('model')
            or checkpoint
        )
    else:
        state_dict = checkpoint

    if any(key.startswith('module.') for key in state_dict):
        state_dict = {
            key.removeprefix('module.'): value
            for key, value in state_dict.items()
        }
    head.load_state_dict(state_dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def squeeze_batch(value: np.ndarray) -> np.ndarray:
    if value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def load_feature_sample(pt_path: Path) -> dict:
    data = torch.load(pt_path, map_location='cpu')
    return {
        'feat': data['feat'],
        'query_points': squeeze_batch(as_numpy(data['query_points'])).astype(np.float32),
        'target_points': squeeze_batch(as_numpy(data['target_points'])).astype(np.float32),
        'occluded': squeeze_batch(as_numpy(data['occluded'])).astype(bool),
        'meta': data.get('meta', {}),
    }


def sample_queries(
    query_points: np.ndarray,
    target_points: np.ndarray,
    occluded: np.ndarray,
    max_queries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_queries <= 0 or query_points.shape[0] <= max_queries:
        return query_points, target_points, occluded
    indices = np.random.choice(query_points.shape[0], size=max_queries, replace=False)
    return query_points[indices], target_points[indices], occluded[indices]


def huber_position_loss(
    pred_tracks: torch.Tensor,
    target_tracks: torch.Tensor,
    target_occluded: torch.Tensor,
    coord_scale: float,
    delta: float,
) -> torch.Tensor:
    visible_mask = (~target_occluded.bool()).to(pred_tracks.dtype)
    error = (pred_tracks - target_tracks) * coord_scale
    dist = torch.sqrt(torch.sum(error * error, dim=-1) + 1e-12)
    loss = torch.where(
        dist < delta,
        0.5 * dist * dist,
        delta * (dist - 0.5 * delta),
    )
    return (loss * visible_mask).mean()


def occlusion_loss(
    occlusion_logits: torch.Tensor,
    target_occluded: torch.Tensor,
) -> torch.Tensor:
    target = target_occluded.to(dtype=occlusion_logits.dtype)
    return F.binary_cross_entropy_with_logits(
        occlusion_logits,
        target,
        reduction='mean',
    )


def save_checkpoint(
    output_dir: Path,
    head: TAPStylePredictionHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args,
    name: str,
) -> None:
    torch.save(
        {
            'model_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'config': {
                **vars(args),
                'head_kwargs': HEAD_KWARGS,
                'optimizer_kwargs': OPTIMIZER_KWARGS,
                'position_loss_weight': POSITION_LOSS_WEIGHT,
                'huber_delta': HUBER_DELTA,
            },
        },
        output_dir / name,
    )


def train_feature_file(
    pt_path: Path,
    head: TAPStylePredictionHead,
    optimizer: torch.optim.Optimizer,
    args,
    device: torch.device,
    global_step: int,
    epoch: int,
    output_dir: Path,
) -> tuple[dict, int]:
    sample = load_feature_sample(pt_path)
    meta = sample['meta']
    spatial_feat = sample['feat'].to(device=device, dtype=torch.float32)

    query_points_np, target_points_np, occluded_np = sample_queries(
        sample['query_points'],
        sample['target_points'],
        sample['occluded'],
        max_queries=args.num_queries_per_video,
    )

    eval_size = int(meta.get('eval_size', args.eval_size))
    num_queries = query_points_np.shape[0]
    total_loss = 0.0
    total_pos = 0.0
    total_occ = 0.0

    optimizer.zero_grad(set_to_none=True)

    for start in range(0, num_queries, args.query_chunk_size):
        end = min(start + args.query_chunk_size, num_queries)
        chunk_weight = float(end - start) / float(num_queries)
        query_points = torch.from_numpy(query_points_np[start:end]).to(
            device=device,
            dtype=torch.float32,
        )
        target_points = torch.from_numpy(target_points_np[start:end]).to(
            device=device,
            dtype=torch.float32,
        )
        target_occluded = torch.from_numpy(occluded_np[start:end]).to(device=device)

        cost_volume = compute_cost_volume(
            spatial_feat=spatial_feat,
            query_points=query_points,
            img_H=eval_size,
            img_W=eval_size,
        )
        pred_tracks, occlusion_logits = head(
            cost_volume,
            query_points=query_points,
            image_size=(eval_size, eval_size),
        )
        pos_loss = huber_position_loss(
            pred_tracks=pred_tracks,
            target_tracks=target_points,
            target_occluded=target_occluded,
            coord_scale=256.0 / float(eval_size),
            delta=HUBER_DELTA,
        )
        occ_loss = occlusion_loss(
            occlusion_logits=occlusion_logits,
            target_occluded=target_occluded,
        )
        weighted_pos_loss = POSITION_LOSS_WEIGHT * pos_loss
        loss = weighted_pos_loss + occ_loss

        (loss * chunk_weight).backward()
        total_loss += float(loss.detach().cpu()) * chunk_weight
        total_pos += float(weighted_pos_loss.detach().cpu()) * chunk_weight
        total_occ += float(occ_loss.detach().cpu()) * chunk_weight

        del query_points, target_points, target_occluded
        del cost_volume, pred_tracks, occlusion_logits, weighted_pos_loss, loss

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
    optimizer.step()
    global_step += 1

    if args.save_every > 0 and global_step % args.save_every == 0:
        save_checkpoint(
            output_dir=output_dir,
            head=head,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            name=f'head_step_{global_step:06d}.pt',
        )

    del spatial_feat
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'vid_name': meta.get('vid_name', pt_path.stem),
        'num_queries': int(num_queries),
        'loss': total_loss,
        'position_loss': total_pos,
        'occlusion_loss': total_occ,
    }, global_step


def main(args) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_dir = Path(args.feat_dir)
    pt_files = sorted(feat_dir.glob('*.pt'))
    if args.max_videos > 0:
        pt_files = pt_files[:args.max_videos]
    if not pt_files:
        raise FileNotFoundError(f'No .pt files found in {feat_dir}')

    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                **vars(args),
                'head_kwargs': HEAD_KWARGS,
                'optimizer_kwargs': OPTIMIZER_KWARGS,
                'position_loss_weight': POSITION_LOSS_WEIGHT,
                'huber_delta': HUBER_DELTA,
                'num_pt_files': len(pt_files),
            },
            f,
            indent=2,
        )

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Using device: {device}')
    print(f'Training on {len(pt_files)} feature files from {feat_dir}')

    head = TAPStylePredictionHead(**HEAD_KWARGS).to(device)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), **OPTIMIZER_KWARGS)

    global_step = 0
    history_path = output_dir / 'train_log.jsonl'

    for epoch in range(args.epochs):
        random.shuffle(pt_files)
        epoch_stats = []
        pbar = tqdm(pt_files, desc=f'Epoch {epoch + 1}/{args.epochs}')

        for pt_path in pbar:
            stats, global_step = train_feature_file(
                pt_path=pt_path,
                head=head,
                optimizer=optimizer,
                args=args,
                device=device,
                global_step=global_step,
                epoch=epoch,
                output_dir=output_dir,
            )

            stats['epoch'] = epoch
            stats['global_step'] = global_step
            epoch_stats.append(stats)
            with open(history_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')

            pbar.set_postfix(
                step=global_step,
                loss=f"{stats['loss']:.4f}",
                pos=f"{stats['position_loss']:.2f}",
                occ=f"{stats['occlusion_loss']:.3f}",
                q=stats['num_queries'],
            )

        valid_stats = [s for s in epoch_stats if s['num_queries'] > 0]
        if valid_stats:
            print(
                f"Epoch {epoch + 1}: "
                f"loss={np.mean([s['loss'] for s in valid_stats]):.4f}, "
                f"pos={np.mean([s['position_loss'] for s in valid_stats]):.4f}, "
                f"occ={np.mean([s['occlusion_loss'] for s in valid_stats]):.4f}"
            )

        save_checkpoint(
            output_dir=output_dir,
            head=head,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            name='head_latest.pt',
        )

    save_checkpoint(
        output_dir=output_dir,
        head=head,
        optimizer=optimizer,
        epoch=args.epochs,
        global_step=global_step,
        args=args,
        name='head_final.pt',
    )
    print(f'Done. Saved checkpoints to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a TAP-style prediction head from pre-extracted features.'
    )
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Directory containing pre-extracted .pt feature files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for checkpoints and train logs.')
    parser.add_argument('--eval_size', type=int, default=256,
                        help='Fallback coordinate resolution if a feature file has no meta eval_size.')
    
    parser.add_argument('--num_queries_per_video', type=int, default=-1,
                        help='Randomly sample this many queries per video; <=0 uses all.')
    parser.add_argument('--query_chunk_size', type=int, default=64,
                        help='Number of queries processed at once while accumulating one video step.')
    
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=-1.0)
    parser.add_argument('--save_every', type=int, default=400)
    parser.add_argument('--max_videos', type=int, default=-1,
                        help='Limit number of .pt files used for training; <=0 uses all.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
