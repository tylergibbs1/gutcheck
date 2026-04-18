# GutCheck

Polyp segmentation paradigm comparison. Two-week exploration, video deliverable.

## Paradigms

| Tag | Approach | Color | Trainable params |
|-----|----------|-------|------------------|
| `sam_zs` | SAM 3.1 zero-shot, text prompt "polyp" | blue | 0 |
| `sam_lora` | SAM 3.1 + LoRA on mask decoder, 880 images | purple | ~1M |
| `pranet` | PraNet 2020 published checkpoint | orange | 0 (pretrained) |
| `dinov3` | Frozen DINOv3 ViT-L + MLP decoder, 880 images | green | ~2M (decoder only) |

## Datasets

| Use | Dataset | Size |
|-----|---------|------|
| Train | Kvasir-SEG train | 880 images |
| In-distribution test | Kvasir-SEG val | 120 images |
| Cross-dataset test | CVC-ClinicDB | 612 images |
| Real-world demo | HyperKvasir video clips | handful of raw clips |

## Run order (on an H100 pod)

```bash
bash scripts/setup_pod.sh
bash scripts/download_data.sh
bash scripts/download_models.sh

python scripts/eval_sam_zeroshot.py
python scripts/eval_pranet.py
python scripts/train_sam_lora.py
python scripts/eval_sam_lora.py
python scripts/train_dinov3.py
python scripts/eval_dinov3.py

python scripts/render_comparison_grid.py
python scripts/render_agreement_heatmap.py
python scripts/render_video_overlay.py
python scripts/summary_plot.py
```

All outputs land in `results/`. Metrics roll up to `results/metrics.json`.
