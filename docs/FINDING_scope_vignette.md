# What breaks SAM 3.1 on polyps

A small investigation built on top of the main GutCheck benchmark. Uses the same 100 Kvasir-SEG and 62 CVC-ClinicDB test images.

## The setup

SAM 3.1 zero-shot scores **Dice mean 0.549 / median 0.883** on CVC-ClinicDB. That gap between mean and median is the mystery: a subset of images are catastrophic (Dice < 0.2), while most are fine. 25 of 62 CVC images fall into the catastrophic bucket. 17 of 100 Kvasir images do.

PraNet from 2020 catastrophically fails on only **2** out of 162 — meaning 40 of SAM 3.1's 42 catastrophic failures are images a six-year-old specialist model handles correctly. The failures are about SAM 3.1, not about the images being intrinsically unsegmentable.

## Step 1 — Feature correlation

For every test image, computed 9 features: ground-truth polyp area, centrality, aspect, solidity; image brightness, saturation, specular-highlight fraction, scope-vignette-border fraction, Laplacian-variance sharpness.

Pearson r vs each approach's Dice:

| feature                | SAM 3.1 | PraNet  | DINOv3  |
|------------------------|---------|---------|---------|
| scope\_border\_frac    | **−0.313** | +0.015  | −0.289  |
| aspect                 | −0.183  | −0.015  | −0.144  |
| solidity               | +0.262  | +0.150  | +0.265  |
| specular\_frac         | +0.059  | **−0.223** | −0.030  |
| brightness             | +0.175  | −0.130  | +0.048  |

The dominant predictor of SAM 3.1 failure is **`scope_border_frac`** — the fraction of image-border pixels that are near-black (i.e. the dark vignette around a colonoscope view). PraNet is **immune** to this feature (r ≈ 0). They are failing on different things.

## Step 2 — Bucket statistics

SAM 3.1 images split by Dice bucket:

| bucket        | n   | area\_frac | scope\_border\_frac | brightness |
|---------------|-----|-----------|---------------------|-----------|
| catastrophic  | 42  | 0.118     | **0.728**               | 0.478     |
| weak          |  7  | 0.174     | 0.513               | 0.518     |
| ok            | 113 | 0.122     | 0.556               | 0.519     |

Polyp size and centrality are identical across buckets. The one variable that meaningfully shifts is scope-border fraction: catastrophic failures have ~30% more black-frame pixels than OK images.

## Step 3 — Confirmatory intervention

If the dark vignette really causes the failure, removing it should help. For each image, detected the bounding box of the non-dark connected region, cropped to that box, ran SAM 3.1 on the crop, pasted predictions back.

| dataset       | full-frame Dice | scope-cropped Dice | Δ      | n\_catastrophic | n\_recovered to ≥0.5 |
|---------------|-----------------|---------------------|--------|-----------------|-----------------------|
| Kvasir-SEG    | 0.748           | 0.748               | 0.000  | 17              | 0 (0%)                |
| CVC-ClinicDB  | 0.549           | **0.583**           | +0.034 | 25              | **5 (20%)**           |

Rescued images (all CVC):

| image | Dice full | Dice crop | vignette removed |
|-------|-----------|-----------|------------------|
| 266   | 0.00      | 0.89      | 20%              |
| 349   | 0.01      | 0.93      | 20%              |
| 459   | 0.00      | 0.96      | 20%              |
| 65    | 0.03      | 0.95      | 20%              |
| 73    | 0.00      | 0.56      | 20%              |

The crop also *hurt* 3 CVC images (Dice dropped 0.85→0.00 on two of them) — so it is not a universally safe preprocessing step. Net effect is positive.

## What this means

1. **SAM 3.1's failure mode on CVC-ClinicDB is partly explained by the dark scope vignette.** On the subset of CVC images where the black border surrounds a tight circular view, SAM 3.1 frequently latches onto the border geometry instead of the polyp. Cropping fixes ~20% of these failures and gains ~3.4 mean Dice on CVC. This is a cheap, reproducible bug.

2. **Kvasir failures are not explained by vignette.** SAM 3.1 has 17 catastrophic failures on Kvasir with no crop recovery. Features tested (polyp size, centrality, shape, brightness, specularity, blur) don't distinguish catastrophic from OK on Kvasir. A second, unidentified failure mode is present. Candidate hypotheses not yet tested: prompt sensitivity ("polyp" vs "mucosal lesion"), pathology class imbalance in the training data, the multiplex-checkpoint's 4 randomly-initialised conv layers in the FPN.

3. **PraNet's cross-dataset advantage is not just "better model."** 40 of SAM 3.1's 42 catastrophic failures are images PraNet scores >0.9 on. A specialist trained on polyp pixels has internalised "ignore the scope frame" as a prior; SAM 3.1, trained on natural-image concepts with rectangular framing, has not.

## For the video

This is a 60-second segment: the mean-vs-median gap sets up the mystery; the feature correlation names the suspect; the crop intervention confirms part of it; the residual failures are left open. The chart `scope_crop_summary.png` pairs a bar chart of the effect with a per-image scatter showing which individual images were rescued.

## Files

- `feature_vs_dice.png` — scatter of features vs per-approach Dice
- `sam31_specific_failure_gallery.png` — 12 catastrophic failures where PraNet succeeds (GT | SAM 3.1 | PraNet)
- `feature_dice_correlations.csv` — full correlation table
- `failure_overlap.csv` — which approaches share SAM 3.1's catastrophic failures
- `per_image_features.csv` — every test image with its features and all per-approach Dices
- `scope_crop_summary.png` — intervention bar + scatter
- `scope_crop/summary.json` — aggregate intervention result
- `scope_crop/per_image.csv` — per-image full-frame vs cropped Dice
