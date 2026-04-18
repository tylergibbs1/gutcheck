"""Render a side-by-side raw-vs-overlay video on a colonoscopy clip.

Each approach runs inference frame-by-frame. Output is an MP4 with:
  [raw frame] | [4-panel comparison grid] stacked per frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck import APPROACHES
from gutcheck.viz import overlay_mask, comparison_grid
from gutcheck.models.sam_wrapper import Sam3Wrapper
from gutcheck.models.pranet import PraNetWrapper
from gutcheck.models.dinov3_seg import DinoV3Segmenter
from scripts._common import (
    SAM3_DIR, SAM3_LORA_DIR,
    DINOV3_DIR, DINOV3_HEAD_PATH,
    PRANET_REPO, PRANET_CKPT,
    RESULTS,
)


def load_dinov3():
    ckpt = torch.load(str(DINOV3_HEAD_PATH), map_location="cpu", weights_only=False)
    seg = DinoV3Segmenter(hf_repo=str(DINOV3_DIR), local_dir=DINOV3_DIR, input_size=ckpt.get("input_size", 448))
    seg.head.load_state_dict(ckpt["head"])
    seg.cuda()
    seg.backbone.to(torch.bfloat16)
    seg.eval()
    return seg


@torch.inference_mode()
def dinov3_predict(seg, image_rgb):
    h0, w0 = image_rgb.shape[:2]
    enc = seg.processor(images=image_rgb, return_tensors="pt",
                        size={"height": seg.input_size, "width": seg.input_size})
    pv = enc["pixel_values"].cuda().to(torch.bfloat16)
    patch_tokens, h, w = seg.backbone_features(pv)
    logits = seg.head(patch_tokens.float(), h, w, (h0, w0))
    return (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()


def load_sam_lora():
    from peft import PeftModel
    wrap = Sam3Wrapper(SAM3_DIR, dtype=torch.bfloat16)
    wrap.model = PeftModel.from_pretrained(wrap.model, str(SAM3_LORA_DIR))
    wrap.model.eval()
    return wrap


def render_clip(clip_path: Path, out_path: Path, max_frames: int | None = None, stride: int = 1):
    sam_zs = Sam3Wrapper(SAM3_DIR)
    try:
        sam_lora = load_sam_lora()
    except Exception as e:
        print(f"[warn] no SAM LoRA: {e}")
        sam_lora = None
    pranet = PraNetWrapper(PRANET_REPO, PRANET_CKPT)
    dino = load_dinov3() if DINOV3_HEAD_PATH.exists() else None

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total = min(total, max_frames)

    writer = None
    pbar = tqdm(total=total, desc=f"render {clip_path.name}")
    frame_idx = 0
    written = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        if max_frames and written >= max_frames:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        preds = {}
        preds["sam_zs"] = sam_zs.predict(rgb)
        if sam_lora is not None:
            preds["sam_lora"] = sam_lora.predict(rgb)
        preds["pranet"] = pranet.predict(rgb)
        if dino is not None:
            preds["dinov3"] = dinov3_predict(dino, rgb)

        grid = comparison_grid(rgb, preds, gt=None, show_gt_panel=False)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        if writer is None:
            h, w = grid_bgr.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
        writer.write(grid_bgr)
        written += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    print(f"wrote {out_path} ({written} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="path to input clip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    if args.clip is None:
        # auto-find any clip under data/HyperKvasir
        candidates = sorted(
            list((RESULTS.parent / "data" / "HyperKvasir").rglob("*.mp4")) +
            list((RESULTS.parent / "data" / "HyperKvasir").rglob("*.avi"))
        )
        if not candidates:
            print("No clip found. Pass a path.")
            return
        args.clip = str(candidates[0])

    clip = Path(args.clip)
    out = Path(args.out) if args.out else RESULTS / "overlays" / "video" / (clip.stem + "_comparison.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    render_clip(clip, out, max_frames=args.max_frames, stride=args.stride)


if __name__ == "__main__":
    main()
