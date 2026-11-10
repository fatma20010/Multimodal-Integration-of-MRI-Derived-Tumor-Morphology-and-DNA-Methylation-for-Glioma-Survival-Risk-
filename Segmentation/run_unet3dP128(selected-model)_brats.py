#!/usr/bin/env python3
"""
run_unet3d_brats.py
===================
Example usage for the trained 3D U-Net segmentation model:

    best_unet3d_brats2021_p128_bf32_rtx5090.pth

The architecture below is reconstructed to match the checkpoint EXACTLY
(module names == state_dict keys), so the weights load with strict=True.

Checkpoint contents
-------------------
    epoch                : 37
    best_val_dice        : 0.8771
    model_state_dict     : the weights (loaded here)
    optimizer_state_dict : (only needed to resume training)
    scheduler_state_dict : (only needed to resume training)

Model
-----
    3D U-Net, base features = 32, depth = 3 + bottleneck
    block   = [Conv3d -> InstanceNorm3d(affine=False) -> LeakyReLU] x2
    down    = MaxPool3d(2)
    up      = ConvTranspose3d(stride 2)
    in_channels  = 4   (T1, T1ce, T2, FLAIR)  -- ORDER MUST MATCH TRAINING
    out_channels = 4   (0=background, 1=NCR, 2=ED, 3=ET)
    aux_head2/aux_head3 : deep-supervision heads, used in TRAINING only.

Requirements
------------
    pip install torch numpy nibabel scipy

Typical usage
-------------
    # 1) Full inference on one BraTS case (4 NIfTI modalities) -> saved mask
    python run_unet3d_brats.py \
        --ckpt best_unet3d_brats2021_p128_bf32_rtx5090.pth \
        --t1   case_t1.nii.gz  --t1ce case_t1ce.nii.gz \
        --t2   case_t2.nii.gz  --flair case_flair.nii.gz \
        --out  case_seg_pred.nii.gz

    # 2) Also evaluate against a ground-truth mask (prints WT/TC/ET Dice)
    python run_unet3d_brats.py ... --gt case_seg.nii.gz

    # 3) No data on hand? Verify the model loads and runs on random input:
    python run_unet3d_brats.py --ckpt best_unet3d_brats2021_p128_bf32_rtx5090.pth --smoke_test
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. MODEL DEFINITION  (must mirror the checkpoint's module names)
# ----------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Two (Conv3d -> InstanceNorm3d -> LeakyReLU) stages.

    The submodule is named ``block`` so its parameters are stored as
    ``<name>.block.0.weight`` / ``<name>.block.3.weight`` -- exactly the
    keys found in the checkpoint. InstanceNorm3d uses affine=False (its
    default), which is why the checkpoint holds no norm parameters.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),   # .0
            nn.InstanceNorm3d(out_ch),                            # .1 (no params)
            nn.LeakyReLU(inplace=True),                           # .2
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),  # .3
            nn.InstanceNorm3d(out_ch),                            # .4 (no params)
            nn.LeakyReLU(inplace=True),                           # .5
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base=32):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8]          # 32, 64, 128, 256

        # Encoder
        self.enc1 = ConvBlock(in_channels, f[0])          # 4  -> 32
        self.enc2 = ConvBlock(f[0], f[1])                 # 32 -> 64
        self.enc3 = ConvBlock(f[1], f[2])                 # 64 -> 128
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f[2], f[3])           # 128 -> 256

        # Decoder (ConvTranspose3d upsampling + skip concat)
        self.up3  = nn.ConvTranspose3d(f[3], f[2], kernel_size=2, stride=2)  # 256->128
        self.dec3 = ConvBlock(f[3], f[2])                 # concat(128+128)=256 -> 128
        self.up2  = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)  # 128->64
        self.dec2 = ConvBlock(f[2], f[1])                 # concat(64+64)=128  -> 64
        self.up1  = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)  # 64->32
        self.dec1 = ConvBlock(f[1], f[0])                 # concat(32+32)=64   -> 32

        # Heads
        self.final_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)   # 32 -> 4
        # Deep-supervision heads (training only; defined so strict load works)
        self.aux_head2  = nn.Conv3d(f[1], out_channels, kernel_size=1)   # 64 -> 4
        self.aux_head3  = nn.Conv3d(f[2], out_channels, kernel_size=1)   # 128-> 4

    def forward(self, x, deep_supervision=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.final_conv(d1)

        if deep_supervision:                      # only useful during training
            return out, self.aux_head2(d2), self.aux_head3(d3)
        return out


def load_model(ckpt_path, device="cpu"):
    """Instantiate UNet3D and load the trained weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = UNet3D(in_channels=4, out_channels=4, base=32)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if "best_val_dice" in ckpt:
        print(f"[load] epoch={ckpt.get('epoch','?')}  "
              f"best_val_dice={ckpt['best_val_dice']:.4f}")
    return model


# ----------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------
def znorm_nonzero(vol):
    """Per-modality z-score over the non-zero (brain) region -- BraTS standard."""
    vol = vol.astype(np.float32)
    mask = vol > 0
    if mask.sum() > 0:
        mean = vol[mask].mean()
        std = vol[mask].std() + 1e-8
        vol = (vol - mean) / std
        vol[~mask] = 0.0
    return vol


def load_case(t1, t1ce, t2, flair):
    """Load 4 NIfTI modalities -> (4, D, H, W) float32 tensor + reference nibabel image.

    IMPORTANT: the channel order (T1, T1ce, T2, FLAIR) must match how the
    model was trained. Change it here if your training order differed.
    """
    import nibabel as nib
    imgs = [nib.load(p) for p in (t1, t1ce, t2, flair)]
    ref = imgs[0]
    arrs = [znorm_nonzero(im.get_fdata()) for im in imgs]
    x = np.stack(arrs, axis=0)                # (4, D, H, W)
    return torch.from_numpy(x).float(), ref


# ----------------------------------------------------------------------
# 3. SLIDING-WINDOW INFERENCE (Gaussian-weighted, optional TTA)
# ----------------------------------------------------------------------
def _gaussian_map(patch, sigma_scale=0.125, device="cpu"):
    """3D Gaussian importance map: center voxels weigh more than edges,
    which suppresses seams between overlapping patches."""
    coords = [torch.arange(p, device=device) - (p - 1) / 2.0 for p in patch]
    g = torch.ones(patch, device=device)
    for i, c in enumerate(coords):
        sigma = patch[i] * sigma_scale
        line = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        shape = [1, 1, 1]; shape[i] = patch[i]
        g = g * line.view(shape)
    return (g / g.max()).clamp_min(1e-4)


@torch.no_grad()
def sliding_window_predict(model, x, patch=(128, 128, 128), overlap=0.5,
                           tta=False, device="cpu"):
    """
    x     : (4, D, H, W) tensor (already normalized)
    returns: (num_classes, D, H, W) averaged softmax probabilities
    """
    model.eval()
    C, D, H, W = x.shape
    patch = tuple(min(patch[i], (D, H, W)[i]) for i in range(3))

    # pad so every dim is at least one patch
    pad = [max(0, patch[2] - W), 0, max(0, patch[1] - H), 0, max(0, patch[0] - D), 0]
    x = F.pad(x.unsqueeze(0), pad, mode="constant", value=0).squeeze(0)
    _, Dp, Hp, Wp = x.shape

    step = [max(1, int(p * (1 - overlap))) for p in patch]

    def starts(dim, p, s):
        ss = list(range(0, dim - p + 1, s))
        if ss[-1] != dim - p:
            ss.append(dim - p)
        return ss

    zs = starts(Dp, patch[0], step[0])
    ys = starts(Hp, patch[1], step[1])
    xs = starts(Wp, patch[2], step[2])

    n_classes = 4
    prob_sum = torch.zeros((n_classes, Dp, Hp, Wp), device=device)
    wsum     = torch.zeros((1, Dp, Hp, Wp), device=device)
    gauss    = _gaussian_map(patch, device=device).unsqueeze(0)   # (1, pd, ph, pw)

    x = x.to(device)
    flips = [None]
    if tta:                                   # identity + 3 single-axis flips
        flips += [(1,), (2,), (3,)]           # spatial dims of the (1,C,pd,ph,pw) tensor

    n_patches = len(zs) * len(ys) * len(xs)
    done = 0
    for z in zs:
        for y in ys:
            for xx in xs:
                patch_in = x[:, z:z + patch[0], y:y + patch[1], xx:xx + patch[2]].unsqueeze(0)
                acc = torch.zeros((n_classes,) + patch, device=device)
                for fl in flips:
                    inp = torch.flip(patch_in, dims=fl) if fl else patch_in
                    logits = model(inp)                        # (1, 4, pd, ph, pw)
                    prob = F.softmax(logits, dim=1)
                    if fl:
                        prob = torch.flip(prob, dims=fl)
                    acc += prob.squeeze(0)
                acc /= len(flips)
                prob_sum[:, z:z + patch[0], y:y + patch[1], xx:xx + patch[2]] += acc * gauss
                wsum[:,  z:z + patch[0], y:y + patch[1], xx:xx + patch[2]] += gauss
                done += 1
        print(f"\r[infer] patch {done}/{n_patches}", end="", flush=True)
    print()

    prob = prob_sum / wsum.clamp_min(1e-8)
    return prob[:, :D, :H, :W]                 # crop back to original size


# ----------------------------------------------------------------------
# 4. POST-PROCESSING
# ----------------------------------------------------------------------
def largest_component_per_class(label_map, classes=(1, 2, 3)):
    """Keep only the largest 3D connected component of each tumour class,
    removing spurious specks -- the report's post-processing step."""
    from scipy import ndimage
    out = label_map.copy()
    for c in classes:
        binary = (label_map == c)
        if binary.sum() == 0:
            continue
        lab, n = ndimage.label(binary)
        if n <= 1:
            continue
        sizes = ndimage.sum(binary, lab, index=range(1, n + 1))
        keep = np.argmax(sizes) + 1
        out[(binary) & (lab != keep)] = 0
    return out


# ----------------------------------------------------------------------
# 5. EVALUATION (BraTS region Dice)
# ----------------------------------------------------------------------
def region_dice(pred, gt):
    """Dice for the 3 BraTS evaluation regions.
        WT = {1,2,3},  TC = {1,3},  ET = {3}
    If your ground truth encodes ET as label 4, remap 4->3 before calling.
    """
    def dice(a, b):
        a, b = a.astype(bool), b.astype(bool)
        inter = np.logical_and(a, b).sum()
        s = a.sum() + b.sum()
        return 1.0 if s == 0 else 2.0 * inter / s
    regions = {
        "WT": (np.isin(pred, [1, 2, 3]), np.isin(gt, [1, 2, 3])),
        "TC": (np.isin(pred, [1, 3]),    np.isin(gt, [1, 3])),
        "ET": (pred == 3,                gt == 3),
    }
    return {k: dice(p, g) for k, (p, g) in regions.items()}


# ----------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="3D U-Net BraTS inference example")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t1"); ap.add_argument("--t1ce")
    ap.add_argument("--t2"); ap.add_argument("--flair")
    ap.add_argument("--out", default="seg_pred.nii.gz")
    ap.add_argument("--gt", default=None, help="optional ground-truth mask for Dice")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (flips)")
    ap.add_argument("--no_postproc", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke_test", action="store_true",
                    help="load model + run on random input, no data needed")
    args = ap.parse_args()

    model = load_model(args.ckpt, device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters: {n_params:,}")

    # -- smoke test: prove the model loads and produces a valid output shape --
    if args.smoke_test:
        x = torch.randn(1, 4, 128, 128, 128, device=args.device)
        with torch.no_grad():
            y = model(x)
        print(f"[smoke] input {tuple(x.shape)} -> output {tuple(y.shape)} "
              f"(expected (1, 4, 128, 128, 128))")
        print("[smoke] OK: weights loaded and forward pass works.")
        return

    # -- real inference --
    for name, val in [("t1", args.t1), ("t1ce", args.t1ce),
                      ("t2", args.t2), ("flair", args.flair)]:
        if val is None:
            ap.error(f"--{name} is required for inference (or use --smoke_test)")

    import nibabel as nib
    x, ref = load_case(args.t1, args.t1ce, args.t2, args.flair)
    print(f"[data] volume {tuple(x.shape)}")

    prob = sliding_window_predict(model, x, overlap=args.overlap,
                                  tta=args.tta, device=args.device)
    pred = prob.argmax(0).cpu().numpy().astype(np.uint8)

    if not args.no_postproc:
        pred = largest_component_per_class(pred)

    nib.save(nib.Nifti1Image(pred, ref.affine, ref.header), args.out)
    print(f"[save] segmentation -> {args.out}")
    vox = np.prod(ref.header.get_zooms()[:3]) / 1000.0  # cm^3 per voxel
    for c, nm in [(1, "NCR"), (2, "ED"), (3, "ET")]:
        print(f"[vol] {nm}: {(pred == c).sum() * vox:8.2f} cm^3")

    if args.gt:
        gt = nib.load(args.gt).get_fdata().astype(np.int16)
        gt[gt == 4] = 3                        # BraTS raw ET label 4 -> 3
        d = region_dice(pred, gt)
        print(f"[dice] WT={d['WT']:.4f}  TC={d['TC']:.4f}  ET={d['ET']:.4f}  "
              f"mean={np.mean(list(d.values())):.4f}")


if __name__ == "__main__":
    main()
