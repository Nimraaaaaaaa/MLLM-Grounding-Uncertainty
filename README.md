# Uncertainty and Verification for MLLM Visual Grounding under Image Corruptions

A small pilot study on **Qwen2.5-VL-3B-Instruct** and **RefCOCO**. It asks three questions:

1. How much does visual grounding accuracy drop under image corruptions?
2. Which uncertainty signal best detects grounding failures: a free signal (token entropy) or a costly one (sampling consistency)?
3. Does an uncertainty-triggered verification loop (sample, zoom, verify) improve accuracy, compared with a random trigger at equal compute?

> **Scope note:** this is a small-scale pilot (one model, one dataset, 150 test images, synthetic corruptions). It is meant to show a careful experimental workflow, not to make general claims.

---

## Key findings

| # | Finding |
|---|---|
| 1 | Corruptions hurt grounding substantially. Accuracy drops from 92.0% (clean) to 75.3% (blur) and 62.7% (random occlusion). |
| 2 | **Sampling consistency detects failures much better than token entropy.** Excluding the ill-posed `target_occ` condition, AUROC is 0.824 for consistency vs 0.665 for mean digit-token entropy. |
| 3 | **Uncertainty-gated verification did not improve accuracy** (gain +0.001, 95% CI [-0.013, +0.014] at a 30% trigger rate). It was also no better than a random trigger. This is a negative result. |
| 4 | Better answers existed among the candidates (oracle selection: 68.8% to 78.6%), but the self-verification step failed to select them (always-verify fixed 32 predictions and broke 39). |

---

## Method

- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct`, fp16, on a free Colab T4 GPU.
- **Task:** referring expression grounding. The model outputs a bounding box as JSON; a prediction counts as correct if IoU >= 0.5 (Acc@0.5).
- **Data:** `lmms-lab/RefCOCO` (val split). A dev set (100 images) and a test set (150 images), **with disjoint images** to avoid leakage. Hyperparameters and gating models are chosen on dev; test is only used for reporting.
- **Corruptions (6 conditions):** `clean`, `blur` (Gaussian, radius 5), `lowres` (downsample to 20% then upsample), `noise` (Gaussian, sigma 35), `rand_occ` (25% of the image covered by a gray box at a random place), `target_occ` (a random half of the ground-truth box covered).
- **Uncertainty signals:**
  - *Free* (single greedy pass): mean and max entropy over the coordinate digit tokens, negative log-probability, and min token probability.
  - *Costly* (K=4 extra samples at temperature 0.8): `u_cons` = 1 - mean IoU between sampled boxes and the greedy box; `u_agree` = 1 - fraction of samples with IoU > 0.5 to the greedy box.
  - *Learned combinations:* logistic regression on the free signals (`u_free`) and on all signals (`u_combo`), trained on dev.
- **Verification loop:** candidates are the greedy box, sampled boxes, and a zoomed-in re-grounding of the box. Each candidate is drawn as a red rectangle on the image, and the same model is asked whether it correctly encloses the expression. The Yes/No logit gap is the score, and the top-scoring candidate is returned.
- **Statistics:** 95% confidence intervals come from a cluster bootstrap over images (so the 6 conditions of one image are not treated as independent). AUROC is reported both pooled and per condition, because pooled AUROC can be inflated by differences between conditions.
- **Gating baselines:** random trigger (averaged over 20 seeds) and an oracle trigger (upper bound).

---

## Results (test set, 150 images x 6 conditions)

### RQ1: Effect of corruptions

| Condition | Acc@0.5 | mIoU | Mean digit entropy | Consistency |
|---|---|---|---|---|
| clean | 0.920 | 0.841 | 1.033 | 0.893 |
| blur | 0.753 | 0.660 | 1.371 | 0.731 |
| lowres | 0.813 | 0.753 | 1.199 | 0.816 |
| noise | 0.880 | 0.781 | 1.160 | 0.827 |
| rand_occ | 0.627 | 0.592 | 1.049 | 0.796 |
| target_occ | 0.133 | 0.379 | 1.075 | 0.774 |

`target_occ` is ill-posed: half of the target is hidden but the ground-truth box is still the full object, so low accuracy there reflects the task, not only the model. We therefore also report results without it.

### RQ2: Failure detection (AUROC, 95% cluster-bootstrap CI)

| Signal | All conditions | Excluding `target_occ` |
|---|---|---|
| Mean digit entropy (`u_Hd`) | 0.543 [0.497, 0.589] | 0.665 [0.600, 0.728] |
| Max digit entropy (`u_Hmax`) | 0.452 [0.412, 0.499] | 0.551 [0.491, 0.617] |
| Neg. log-prob (`u_nlp`) | 0.496 [0.448, 0.542] | 0.613 [0.544, 0.678] |
| Min token prob (`u_minp`) | 0.432 [0.389, 0.482] | 0.528 [0.463, 0.595] |
| Sampling consistency (`u_cons`) | 0.715 [0.683, 0.753] | **0.824** [0.790, 0.863] |
| Sampling agreement (`u_agree`) | 0.701 [0.669, 0.734] | 0.776 [0.736, 0.816] |
| Combined free signals (`u_free`) | 0.634 [0.590, 0.678] | 0.701 [0.632, 0.767] |
| Combined all signals (`u_combo`) | 0.779 [0.747, 0.811] | **0.838** [0.805, 0.875] |

Per-condition AUROC (below) shows the consistency advantage is not an artifact of pooling:

| Condition | `u_Hd` | `u_cons` | `u_combo` |
|---|---|---|---|
| clean | 0.717 | 0.839 | 0.877 |
| blur | 0.719 | 0.807 | 0.794 |
| lowres | 0.776 | 0.866 | 0.856 |
| noise | 0.660 | 0.856 | 0.843 |
| rand_occ | 0.676 | 0.789 | 0.805 |
| target_occ | 0.408 | 0.419 | 0.471 |

Token entropy alone is a weak detector (pooled AUROC near chance). `u_Hmax` and `u_minp` fall below 0.5 when `target_occ` is included. The cost trade-off matters: entropy is free, while consistency needs K=4 extra generations.

### RQ3: Does gated verification help? (30% of samples verified)

| Trigger | Accuracy | Gain vs base (0.688) | 95% CI | Extra calls / sample |
|---|---|---|---|---|
| entropy-based (free) | 0.689 | +0.001 | [-0.013, +0.014] | 2.46 |
| sampling consistency | 0.682 | -0.006 | [-0.022, +0.013] | 2.69 |
| combined | 0.686 | -0.002 | [-0.019, +0.011] | 2.64 |
| random | 0.684 | -0.003 | [-0.014, +0.007] | 2.23 |

All CIs include zero, so there is **no evidence that verification helps**, and no evidence that uncertainty-based triggering beats random triggering. Verifying every sample gave 68.0% accuracy (fixed 32, broke 39).

### Why did verification fail? (diagnostics)

**Was a correct box among the candidates?** Yes, often. Selecting the best candidate with an oracle would raise accuracy a lot:

| Condition | Greedy correct | Any candidate correct |
|---|---|---|
| clean | 0.920 | 0.967 |
| blur | 0.753 | 0.827 |
| lowres | 0.813 | 0.920 |
| noise | 0.880 | 0.940 |
| rand_occ | 0.627 | 0.720 |
| target_occ | 0.133 | 0.340 |
| **overall** | **0.688** | **0.786** |

**Can the verifier tell good boxes from bad ones?** Only moderately. The 3B model's Yes/No score, used to detect failures of the greedy box, has pooled AUROC 0.701, but it is close to chance on clean images (0.541) and on `target_occ` (0.533), and moderate on blur, lowres, noise and rand_occ (0.68 to 0.78).

So the headroom exists (about 10 accuracy points), but the same 3B model is not a reliable enough judge to capture it. We did not run experiments isolating the exact cause, so this is an interpretation, not a proven explanation.

### Ablation on dev (hard samples, n = 100 selected by high entropy)

| Config | Base acc | Verified acc | Gain | Avg cost | Fixed | Broken |
|---|---|---|---|---|---|---|
| A_samples | 0.57 | 0.52 | -0.05 | 6.87 | 6 | 11 |
| B_zoom | 0.57 | 0.55 | -0.02 | 2.50 | 2 | 4 |
| C_both | 0.57 | 0.50 | -0.07 | 8.24 | 5 | 12 |
| D_both_crop | 0.57 | 0.56 | -0.01 | 8.24 | 6 | 7 |
| E_both_2rounds | 0.57 | 0.49 | -0.08 | 9.50 | 4 | 12 |

All configurations had negative gain on dev. The final test run used `C_both` (samples + zoom, marked-box verification). Note that `D_both_crop` had the best dev gain, so the test-set config choice was not strictly the dev-optimal one. Since every config was negative, we do not expect this to change the conclusion, but it is a limitation.

---

## Limitations

- **One model** (3B parameters) and **one dataset** (RefCOCO val). Results may not transfer to larger models or other datasets.
- **Small sample:** 100 dev and 150 test images (900 test evaluations, but only 150 independent images).
- **Synthetic corruptions** may not match real-world degradation.
- `target_occ` is ill-posed (see above).
- **The verifier is the same model** as the grounder, which likely limits verification quality.
- A single run with a fixed seed; we did not measure variance across seeds.
- Cost is counted in model calls, not wall-clock time.

---

## Reproducing

Run on Google Colab with a T4 GPU (free tier is enough, but the full pipeline takes several hours).

1. Open the notebook `Uncertainty_aware_verification_loop_for_MLLM_visual_grounding.ipynb` in Colab and select **Runtime > Change runtime type > T4 GPU**.
2. Run the cells top to bottom: GPU check, setup and model load, logits hook, dataset, grounding functions, corruptions, base evaluation (dev, then test), analysis, verification loop, ablation, final test verification, RQ3 analysis, and the diagnostic cells.
3. Results are written as `.jsonl` and `.csv` files to Google Drive after every step. If the session disconnects, re-run the setup cells and the same evaluation cell; finished items are skipped.

Main dependencies: `transformers`, `accelerate`, `datasets`, `scikit-learn`, `pandas`, `matplotlib`, `torch`.

**Implementation notes**
- `repetition_penalty=1.0` is set so that the digit-token probabilities are not distorted.
- A small hook on the `lm_head` output replaces NaN/inf logits, which otherwise caused a CUDA assert during fp16 sampling on the T4.
- Corruptions are deterministic per (image, condition), so results are reproducible.

## Possible next steps

- Repeat on RefCOCOg and with a larger model (for example Qwen2.5-VL-7B, 4-bit).
- Use a separate, stronger verifier instead of the same 3B model.
- Learn a selection rule over candidates (rather than a Yes/No score) to capture the oracle headroom.
- Report results over multiple seeds.

---

## Author

**Nimra Abdulhaq** | nimraabdulhaqq@gmail.com
