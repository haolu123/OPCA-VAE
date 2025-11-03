#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_poke_tools.py  —  PCA-VAE probing utilities (warm-build + clean version)

Features
--------
1) BASIS (single or batch):
   - Visualize what each PCA component decodes to by constructing a one-hot Y,
     mapping back via your PCA layer, and decoding to RGB.
   - Accepts a single index (e.g., "7"), a list ("0,2,5"), or a range ("0-9" or "0..9").
   - Saves individual component images AND an optional grid summary image.

2) EDIT (with side-by-side comparison, supports delta scanning):
   - Encode an image → get Y → nudge one component → decode back.
   - Saves a 3-panel comparison: [Input | Reconstruct | Edited-Reconstruct].
   - Spatial PCA supports apply_scope='all' or 'pixel' (with --hw "h,w").
   - Delta scanning: pass a range (e.g., --delta -1.0..1.0 --step 0.1) to produce a series
     and optionally assemble them into a grid strip.

Notes
-----
- Warm-builds PCA layer (codes/xN_mean/mapper/decoder or codes_spatial/x_mean_spatial) BEFORE loading weights,
  to avoid missing/unexpected keys.
- No torchvision dependency (uses PIL + numpy for image I/O and grid assembly).
"""

import os, math, argparse
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

# ---------- Helpers: parsing / image utils ----------

def parse_comp_arg(spec: str) -> List[int]:
    """
    Parse component spec:
      - "7" -> [7]
      - "0,2,5" -> [0,2,5]
      - "0-9" or "0..9" -> [0,1,...,9]
    """
    s = spec.strip()
    if "," in s:
        out = []
        for p in s.split(","):
            p = p.strip()
            if not p:
                continue
            out.append(int(p))
        seen, uniq = set(), []
        for v in out:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        return uniq
    if "-" in s or ".." in s:
        sep = ".." if ".." in s else "-"
        a, b = s.split(sep)
        a, b = int(a), int(b)
        if a <= b:
            return list(range(a, b + 1))
        else:
            return list(range(b, a + 1))
    return [int(s)]


def parse_delta_arg(spec: str, step: float) -> List[float]:
    """
    Parse delta spec:
      - "0.5" -> [0.5]
      - "-1.0..1.0" or "-1.0-1.0" with --step 0.1 -> [-1.0, -0.9, ..., 1.0]
      - "0.0,0.5,1.0" -> [0.0, 0.5, 1.0]
    """
    s = spec.strip()
    if "," in s:
        vals = [float(x.strip()) for x in s.split(",") if x.strip()]
        return vals
    if "-" in s or ".." in s:
        sep = ".." if ".." in s else "-"
        a_str, b_str = s.split(sep)
        a, b = float(a_str), float(b_str)
        if step <= 0:
            raise ValueError("--step must be > 0 for delta ranges")
        lo, hi = (a, b) if a <= b else (b, a)
        n = int(round((hi - lo) / step))
        vals = [round(lo + i * step, 10) for i in range(n + 1)]
        if abs(vals[-1] - hi) > 1e-9:
            vals.append(hi)
        return vals if a <= b else list(reversed(vals))
    return [float(s)]


def to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] in [0,1] -> uint8 [H,W,3]."""
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def concat_h(images: List[Image.Image]) -> Image.Image:
    """Concatenate PIL images horizontally with tallest alignment."""
    if not images:
        raise ValueError("No images to concatenate.")
    heights = [im.height for im in images]
    widths  = [im.width for im in images]
    H = max(heights)
    W = sum(widths)
    out = Image.new("RGB", (W, H), (0, 0, 0))
    x = 0
    for im in images:
        if im.height != H:
            im = im.resize((im.width, H), Image.BILINEAR)
        out.paste(im, (x, 0))
        x += im.width
    return out


def make_labeled_tile(img: Image.Image, label: str, bar: int = 26) -> Image.Image:
    """Add a top bar with text label (no custom fonts to avoid dependencies)."""
    W, H = img.size
    out = Image.new("RGB", (W, H + bar), (0, 0, 0))
    out.paste(img, (0, bar))
    drawer = ImageDraw.Draw(out)
    drawer.rectangle([0, 0, W, bar], fill=(30, 30, 30))
    drawer.text((6, 4), label, fill=(220, 220, 220))
    return out


def assemble_grid(images: List[Image.Image], cols: int = 10, label_prefix: Optional[str] = None):
    """Assemble a labeled grid image from equal-sized tiles."""
    if not images:
        raise ValueError("No images to assemble into grid.")
    tiles = []
    for i, im in enumerate(images):
        lab = f"{label_prefix}{i}" if label_prefix is not None else ""
        tiles.append(make_labeled_tile(im, lab) if lab else im)
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        grid.paste(tile, (c * tw, r * th))
    return grid


# ---------- Model loading (warm-build before load) ----------

def _warm_build_pca(model, cfg, device):
    """Run a dummy forward pass to force-build PCA layer params/buffers."""
    in_ch  = int(cfg["model_params"].get("in_channels", 3))
    img_sz = int(cfg["model_params"].get("img_size", 256))
    x = torch.zeros(1, in_ch, img_sz, img_sz, device=device)
    try:
        model.forward(x, status='Valid')
    except Exception:
        z = model.encode(x)[0]
        model.vq_layer(z, status='Valid', current_lr=None)


def _load_model_from_cfg_and_ckpt(cfg_path: str, ckpt_path: str, device: str = "cuda") -> nn.Module:
    """
    Build model from YAML and load Lightning .ckpt weights (handles 'model.' prefix).
    Warm-builds PCA buffers BEFORE loading to avoid missing/unexpected keys.
    """
    if not torch.cuda.is_available():
        device = "cpu"

    import sys
    sys.path.append(os.path.dirname(cfg_path))
    sys.path.append(os.path.dirname(ckpt_path))
    sys.path.append(os.getcwd())

    from omegaconf import OmegaConf
    from models import vae_models  # provided by your repo

    cfg = OmegaConf.load(cfg_path)
    model = vae_models[cfg["model_params"]["name"]](**cfg["model_params"])
    model.to(device).eval()
    _warm_build_pca(model, cfg, device)  # <-- build pca internals first

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch >= 2.4
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")  # older torch

    sd = state.get("state_dict", state)

    new_sd = {}
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith("model."):
            new_sd[k[6:]] = v
        else:
            new_sd[k] = v

    load_result = model.load_state_dict(new_sd, strict=False)
    print(f"[load] missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
    return model, cfg


@torch.no_grad()
def _latent_shape_from_spatial_codes(vq_layer) -> tuple:
    L, D, Q = vq_layer.codes_spatial.shape
    H = int(math.isqrt(L))
    W = H
    if H * W != L:
        H, W = L, 1
    return D, H, W


@torch.no_grad()
def _infer_DHW(model: nn.Module):
    dev = next(model.parameters()).device
    size = getattr(model, "img_size", 256)
    dummy = torch.zeros(1, 3, size, size, device=dev)
    enc = model.encode(dummy)[0]
    _, D, H, W = enc.shape
    return D, H, W


@torch.no_grad()
def _ensure_pca_built(model: nn.Module):
    vq = model.vq_layer
    dev = next(model.parameters()).device
    if hasattr(vq, "codes_spatial") or hasattr(vq, "codes"):
        return
    D, H, W = _infer_DHW(model)
    dummy_latent = torch.zeros(1, D, H, W, device=dev)
    _ = vq(dummy_latent, status="Valid", current_lr=None)


def _prep_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


# ---------- Core ops: basis decode / edit ----------

@torch.no_grad()
def decode_from_onehot_Y(model: nn.Module, comp: int, scale: float = 1.0) -> Image.Image:
    dev = next(model.parameters()).device
    vq = model.vq_layer

    if hasattr(vq, "codes_spatial"):
        L, D, Q = vq.codes_spatial.shape
        if not (0 <= comp < Q):
            raise ValueError(f"comp out of range [0, {Q-1}]")
        Y = torch.zeros(1, L, Q, device=dev)
        Y[:, :, comp] = scale
        x_rec_centered = torch.einsum("blq,ldq->bld", Y, vq.codes_spatial)
        x_rec = x_rec_centered + vq.x_mean_spatial.unsqueeze(0)  # [1,L,D]
        D_, H, W = _latent_shape_from_spatial_codes(vq)
        x_latent = x_rec.view(1, H, W, D_).permute(0, 3, 1, 2).contiguous()
    else:
        if not hasattr(vq, "codes") or not hasattr(vq, "xN_mean"):
            _ensure_pca_built(model)
        N, Q = vq.codes.shape[0], vq.codes.shape[1]
        if not (0 <= comp < Q):
            raise ValueError(f"comp out of range [0, {Q-1}]")
        Y = torch.zeros(1, Q, device=dev)
        Y[:, comp] = scale
        xN_centered = Y @ vq.codes.T
        xN = xN_centered + vq.xN_mean.unsqueeze(0)  # [1,N]

        if getattr(vq, "_use_mapper", False):
            if (getattr(vq, "mapper", None) is None) or (getattr(vq, "decoder", None) is None):
                D_, H_, W_ = _infer_DHW(model)
                vq._build_heads(D_ * H_ * W_, dev)
            x_hat_flat = vq.decoder(xN)  # [1, D*H*W]
            D, H, W = _infer_DHW(model)
            x_latent = x_hat_flat.view(1, D, H, W)
        else:
            D, H, W = _infer_DHW(model)
            if xN.numel() != D * H * W:
                raise RuntimeError(f"Classic PCA shape mismatch: xN has {xN.numel()} elements but D*H*W={D*H*W}. "
                                   f"Set _use_mapper=True during training or ensure codes were learned on DHW.")
            x_latent = xN.view(1, D, H, W)

    img = model.decode(x_latent).clamp(-1, 1)
    grid = (img.add(1).div(2)).squeeze(0).detach().cpu()
    pil = Image.fromarray(to_uint8_rgb(grid))
    return pil


@torch.no_grad()
def reconstruct_image(model: nn.Module, image_path: str) -> Image.Image:
    dev = next(model.parameters()).device
    inp = _prep_image(image_path, model.img_size).to(dev)
    x = model.encode(inp)[0]
    out = model.decode(x).clamp(-1, 1)
    grid = (out.add(1).div(2)).squeeze(0).detach().cpu()
    return Image.fromarray(to_uint8_rgb(grid))


@torch.no_grad()
def edit_image_on_component(model: nn.Module, image_path: str, comp: int, delta: float,
                            apply_scope: str = "all", hw: Optional[Tuple[int, int]] = None) -> Image.Image:
    dev = next(model.parameters()).device
    vq = model.vq_layer
    inp = _prep_image(image_path, model.img_size).to(dev)

    encoding = model.encode(inp)[0]  # [1,D,H,W]
    x_hat, _, Y = vq(encoding, status="Valid", current_lr=None)

    if hasattr(vq, "codes_spatial"):
        L, D, Q = vq.codes_spatial.shape
        if not (0 <= comp < Q):
            raise ValueError(f"comp out of range [0, {Q-1}]")
        Y_mod = Y.clone()
        if apply_scope == "all":
            Y_mod[:, :, comp] += delta
        elif apply_scope == "pixel":
            if hw is None:
                raise ValueError("For apply_scope='pixel', please pass --hw h,w")
            D_, H, W = _latent_shape_from_spatial_codes(vq)
            h, w = hw
            if not (0 <= h < H and 0 <= w < W):
                raise ValueError(f"(h,w) out of bounds 0<=h<{H}, 0<=w<{W}")
            l = h * W + w
            Y_mod[:, l, comp] += delta
        else:
            raise ValueError("apply_scope must be 'all' or 'pixel'")

        x_rec_centered = torch.einsum("blq,ldq->bld", Y_mod, vq.codes_spatial)
        x_rec = x_rec_centered + vq.x_mean_spatial.unsqueeze(0)
        D_, H, W = _latent_shape_from_spatial_codes(vq)
        x_latent = x_rec.view(1, H, W, D_).permute(0, 3, 1, 2).contiguous()
    else:
        if not hasattr(vq, "codes") or not hasattr(vq, "xN_mean"):
            _ensure_pca_built(model)
        N, Q = vq.codes.shape[0], vq.codes.shape[1]
        if not (0 <= comp < Q):
            raise ValueError(f"comp out of range [0, {Q-1}]")
        Y_mod = Y.clone()
        Y_mod[:, comp] += delta
        xN_centered = Y_mod @ vq.codes.T
        xN = xN_centered + vq.xN_mean.unsqueeze(0)

        if getattr(vq, "_use_mapper", False):
            if (getattr(vq, "mapper", None) is None) or (getattr(vq, "decoder", None) is None):
                D_, H_, W_ = _infer_DHW(model)
                vq._build_heads(D_ * H_ * W_, dev)
            x_hat_flat = vq.decoder(xN)
            D, H, W = _infer_DHW(model)
            x_latent = x_hat_flat.view(1, D, H, W)
        else:
            D, H, W = _infer_DHW(model)
            if xN.numel() != D * H * W:
                raise RuntimeError(f"Classic PCA shape mismatch: xN has {xN.numel()} elements but D*H*W={D*H*W}. "
                                   f"Set _use_mapper=True during training or ensure codes were learned on DHW.")
            x_latent = xN.view(1, D, H, W)

    img = model.decode(x_latent).clamp(-1, 1)
    grid = (img.add(1).div(2)).squeeze(0).detach().cpu()
    return Image.fromarray(to_uint8_rgb(grid))


# ---------- CLI ----------

def run_basis(args):
    model, cfg = _load_model_from_cfg_and_ckpt(args.config, args.ckpt)
    _ensure_pca_built(model)

    comps = parse_comp_arg(args.comp)
    os.makedirs(args.out_dir, exist_ok=True)

    tiles = []
    for c in comps:
        pil = decode_from_onehot_Y(model, c, args.scale)
        out_path = os.path.join(args.out_dir, f"basis_comp{c}.png")
        pil.save(out_path)
        tiles.append(pil)

    print(f"[BASIS] Saved {len(tiles)} images to: {args.out_dir}")

    if args.grid_out:
        grid = assemble_grid(tiles, cols=args.grid_cols, label_prefix="c")
        grid.save(args.grid_out)
        print(f"[BASIS] Saved grid: {args.grid_out}")

def run_edit(args):
    model, cfg = _load_model_from_cfg_and_ckpt(args.config, args.ckpt)
    _ensure_pca_built(model)

    # Prepare input once
    inp_img = Image.open(args.image).convert("RGB").resize((model.img_size, model.img_size), Image.BILINEAR)

    # Parse comps and deltas
    comps  = parse_comp_arg(args.comp)          # supports "7", "0,2,5", "0..9"
    deltas = parse_delta_arg(args.delta, args.step)

    os.makedirs(args.out_dir, exist_ok=True)

    # For scan_grid: we’ll create rows, one per component:
    # [Input (cX), Edited(cX +d1), Edited(cX +d2), ...]
    grid_tiles: List[Image.Image] = []
    row_width = 1 + len(deltas)  # enforce 1 row per component

    for c in comps:
        # Start a new row with the input tile (labeled by component)
        row_tiles = [make_labeled_tile(inp_img, f"Input (c{c})")]

        for d in deltas:
            edited = edit_image_on_component(
                model, args.image, c, d,
                args.apply_scope,
                tuple(map(int, args.hw.split(","))) if (args.hw and args.apply_scope == "pixel") else None
            )

            # Save per-delta output
            sign = "+" if d >= 0 else ""
            if args.compare.lower() != "off":
                # Two-panel: [Input | Edited]
                pair = concat_h([
                    make_labeled_tile(inp_img, "Input"),
                    make_labeled_tile(edited, f"Edited (c{c} {sign}{d})"),
                ])
                out_path = os.path.join(
                    args.out_dir, f"edit_c{c}_d{str(d).replace('.','p').replace('-','m')}.png"
                )
                pair.save(out_path)
            else:
                # Edited-only file
                out_path = os.path.join(
                    args.out_dir, f"edit_only_c{c}_d{str(d).replace('.','p').replace('-','m')}.png"
                )
                edited.save(out_path)

            # Row for scan grid: add edited-only (labeled) tile
            row_tiles.append(make_labeled_tile(edited, f"c{c} {sign}{d}"))
            print(f"[EDIT] Saved: {out_path}")

        # Append this component’s row to the overall grid tiles
        grid_tiles.extend(row_tiles)

    # Build scan_grid with fixed columns so each component occupies exactly one row
    if args.scan_grid and grid_tiles:
        grid = assemble_grid(grid_tiles, cols=row_width, label_prefix=None)
        grid.save(args.scan_grid)
        print(f"[EDIT] Saved scan grid: {args.scan_grid}")


def main():
    p = argparse.ArgumentParser(description="PCA-VAE Basis Visualization & Latent Editing")
    p.add_argument("--config", required=True, help="Path to YAML config used for training")
    p.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt (or state dict)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_basis = sub.add_parser("basis", help="Decode PCA basis image(s) from one-hot Y")
    p_basis.add_argument("--comp", type=str, required=True, 
                         help='Component index spec: "7" | "0,2,5" | "0-9" or "0..9"')
    p_basis.add_argument("--scale", type=float, default=1.0, help="Coefficient scale for one-hot")
    p_basis.add_argument("--out_dir", type=str, required=True, help="Directory to save individual basis images")
    p_basis.add_argument("--grid_out", type=str, default="", help="Optional combined grid output path (PNG)")
    p_basis.add_argument("--grid_cols", type=int, default=10, help="Columns in grid summary")
    p_basis.set_defaults(func=run_basis)

    p_edit = sub.add_parser("edit", help="Encode an image, nudge one PCA component, and compare (supports delta scan)")
    p_edit.add_argument("--image", required=True, help="Path to input image")
    p_edit.add_argument("--comp", type=str, required=True,
                    help='Component spec: "7" | "0,2,5" | "0-9" or "0..9"')
    p_edit.add_argument("--delta", type=str, required=True,
                        help="Delta value or range: '0.5' | '-1.0..1.0' | '-1.0-1.0' | '0.0,0.5,1.0'")
    p_edit.add_argument("--step", type=float, default=0.1, help="Step when --delta is a range")
    p_edit.add_argument("--apply_scope", choices=["all", "pixel"], default="all",
                        help="Spatial PCA only: apply to all positions or a single (h,w)")
    p_edit.add_argument("--hw", type=str, default=None, help="When --apply_scope=pixel, pass 'h,w'")
    p_edit.add_argument("--compare", type=str, default="on", help="on/off — save triptych comparison or only edited")
    p_edit.add_argument("--out_dir", required=True, help="Directory for outputs (files will be auto-named)")
    p_edit.add_argument("--scan_grid", type=str, default="", help="Optional path to save a grid of all triptychs")
    p_edit.add_argument("--scan_cols", type=int, default=6, help="Columns for the scan grid")
    p_edit.set_defaults(func=run_edit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
