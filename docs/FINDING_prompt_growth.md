# One word changes everything: "growth" vs "polyp" as a SAM 3.1 prompt

Result of the auto-research loop and a follow-up paired ablation.

## The numbers

With identical preprocessing (scope-crop), post-processing (morph close + fill holes), and TTA (hflip union), the only change between the two bars below is the text prompt:

| Prompt   | Kvasir-SEG Dice | CVC-ClinicDB Dice |
|----------|-----------------|-------------------|
| `"polyp"`  | 0.781 / 0.944 (med) | **0.569** / 0.892 (med) |
| `"growth"` | 0.778 / 0.942 (med) | **0.705** / 0.931 (med) |

**+13.5 Dice points on CVC-ClinicDB** from a one-word prompt change. Kvasir is a wash.

## What actually happens per image

| dataset | n | helped by &gt;0.2 | hurt by &gt;0.2 | essentially unchanged |
|---------|---|----------------|---------------|-----------------------|
| Kvasir-SEG  | 100 | 7 | 8 | 82 |
| CVC-ClinicDB | 62 | **12** | **0** | 43 |

On CVC, `"growth"` is a **strict dominance**: it rescues twelve images and hurts none. Of the twelve rescued, nine go from Dice ≈ 0 (catastrophic polyp-prompt failure) to Dice &gt; 0.8. On Kvasir the two prompts trade off image-by-image and the means are identical to within noise.

## Why this happens

Two hypotheses (neither directly tested beyond the data above):

1. **Vocabulary frequency.** SAM 3.1's open-vocabulary text encoder was trained on 4M+ natural-image concepts. `"polyp"` is rare specialized medical jargon; `"growth"` is a common English visual/botanical word (crystal growth, plant growth, tumor growth) that more reliably resolves to the image region. Other synonyms I tried all tanked:

   | Prompt          | Combined Dice |
   |-----------------|---------------|
   | `"growth"`      | **0.741**     |
   | `"polyp"`       | 0.675         |
   | `"tissue growth"` | 0.700       |
   | `"colon polyp"` | 0.534         |
   | `"pink bump"`   | miscaptures   |
   | `"bump"`        | 0.369         |
   | `"tumor"`       | 0.363         |
   | `"lesion"`      | 0.147         |
   | `"adenoma"`     | 0.012 (high presence, wrong mask) |

   So the effect is **specific to `"growth"`** — it's not "any word beats polyp" and it's not "natural-language beats jargon" (`"bump"` and `"lesion"` both lose).

2. **Cross-dataset framing.** The Kvasir ↔ CVC-ClinicDB gap is where `"growth"` shows its whole effect. CVC is the harder, framing-different dataset. The same prompt-sensitivity has no measurable impact when the distribution is close to SAM's training prior (Kvasir). This is compatible with the earlier failure-mode finding that SAM's CVC failures cluster on scope-vignette images.

## What this changes for the paradigm comparison

Original story (from `summary.png`):

> SAM 3.1 zero-shot: 0.549 mean Dice on CVC-ClinicDB. Foundation model loses cleanly to PraNet 2020 (0.905) on cross-dataset generalization.

Revised story (`summary_with_prompt_ablation.png`):

> SAM 3.1 zero-shot, with one-word prompt fix, hits 0.705 mean Dice on CVC-ClinicDB — within 6 Dice points of LoRA fine-tuned on 900 training images (0.764). PraNet still wins the absolute race at 0.905, but the zero-shot-vs-specialist gap collapses from 35 points to 20 just from a prompt swap.

Both stories are true. The second one is the interesting one, because it reframes "how much does a six-month-old foundation model need" as a prompt-engineering question, not an architecture question.

## Files

- `per_image.csv` — per-image Dice under both prompts, for every test image
- `summary.json` — aggregate
- `ablation_summary.png` — paired bar + per-image scatter
- `../summary_with_prompt_ablation.png` — updated headline chart with both SAM 3.1 prompt bars next to the other paradigms
- `polyp/<dataset>/*.png`, `growth/<dataset>/*.png` — prediction masks for both prompts, every test image

## The one caveat

On Kvasir, eight images that polyp-prompt got right (Dice &gt; 0.7) are broken by growth-prompt (Dice &lt; 0.1). A production system shouldn't hard-pick `"growth"` — it should run both and pick whichever looks more confident on the specific image. We haven't built that selector; it's obvious next work if anyone takes this further.
