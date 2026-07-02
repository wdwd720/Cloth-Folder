# OpenAI-generated towel images — the realism bridge (Layer B)

This pipeline generates **photorealistic hotel-towel images** with the OpenAI
image API and feeds them, after QC + pseudo-labeling + human review, into the
existing towel corner-pose training flow. It is the **middle layer** of a 3-layer
data strategy:

| Layer | Source | Role |
|------|--------|------|
| **A** | synthetic (`generate-towel-dataset`) | exact labels, pretraining + smoke tests |
| **B** | **OpenAI-generated (this doc)** | realism *bridge* — diverse lighting/texture/scenes, cheap to scale |
| **C** | real / external (`create-towel-real-dataset`, Kaggle/Open Images + Claude) | final realism, the thing that actually matters |

```
preview-openai-towel-prompts            (inspect the curriculum, free)
        │
generate-openai-towel-dataset --dry-run (cost preview, no API call)
        │   --no-dry-run --max-cost-usd N --yes-i-understand-this-uses-paid-api
        ▼
filter-generated-towel-images           (QC: corrupt/dup/dim/metadata)
        ▼
claude-label-towel-folder               (pseudo-labels — UNTRUSTED)
        ▼
auto-triage-pseudolabels (policy)       (cheaply approve/reject/keep-for-review)
        ▼
review-pseudolabels --source ...        (human reviews ONLY the leftovers)
        ▼
merge-towel-datasets  →  export-yolo-towel-pose --include-pseudolabels approved_only
        ▼
print-towel-training-command  /  recommend-towel-training-plan
```

## Two training tracks — the key efficiency fix

The old `mix_flat_wrinkled_negatives` task generated lots of folded / multiple /
hanging towels. Those are great for a *future rejection critic* but **wasteful for
the YOLO 4-corner pose model**, which only learns from single flat foldable towels.
So generation is now split into two explicit **tracks** (every prompt records its
`training_track`):

| Track | Tasks | What it's for |
|------|-------|---------------|
| **`pose_positive`** | `pose_positive_flat_only`, `pose_positive_diverse_v1` | single, flat/mildly-wrinkled, fully-visible towels, all 4 corners visible → the ONLY pose training data |
| **`critic_negative`** | `critic_negatives_v1` (and the legacy negatives) | folded stacks, rolled, multiple, hanging, occluded, not-a-towel → a future scene/rejection critic, **NOT pose data** |

- **Pose-positive prompts carry a hard constraint block** ("exactly one … flat …
  all four outer corners visible … not folded, not rolled, not stacked, not
  hanging, no people, no hands …") so the generator produces pose-usable images.
- `export-yolo-towel-pose` **excludes `training_track == critic_negative` by
  default** (pass `--include-critic-negatives` only to deliberately include them).
- `auto-triage-pseudolabels --policy pose_positive_strict` then cheaply approves /
  rejects / flags so a human only reviews genuinely uncertain cases.

## What this is for — and why it is NOT enough alone

Synthetic data is cheap and exactly labeled but looks fake; real hotel-towel data
is scarce. OpenAI-generated images bridge the gap: realistic lighting, fabric,
shadows, viewpoints and clutter at volume, for a few dollars. **But generated
images are not ground truth and not real:**

- the model can render impossible fabric, the wrong number of towels, or a
  "towel" that is really a bathrobe;
- its idea of a hotel towel is a *prior*, not your actual robot's camera;
- training on generated corners that are subtly wrong quietly degrades the model.

So generated images are treated exactly like external real images: **pseudo-label,
review, approve, and keep a real-only validation set.** Never ship a detector
validated only on generated data.

## ⚠️ Cost, privacy, and what not to trust

- **Cost.** Every image is a paid API call. `--dry-run` (the default) previews the
  prompts and an **approximate** cost with no charge. Real runs **refuse** unless
  you pass `--no-dry-run`, a `--max-cost-usd` cap, and the explicit
  `--yes-i-understand-this-uses-paid-api` flag, and the predicted cost is within
  the cap. Pricing here is approximate (gpt-image-1 is token-based) — verify at
  <https://openai.com/api/pricing/>. Rough: gpt-image-1 @ 1024² medium ≈ $0.04/img
  → ~$4 for 100, ~$20 for 500.
- **Privacy.** Prompts are sent to OpenAI; generated images come back from OpenAI.
  Do not embed private or proprietary content in prompts.
- **What not to trust.** `category_target` (what we *asked* for) is metadata, never
  a label. Every generated sample starts `usable_for_training: false`,
  `label_status: unlabeled`, `approval_status: pending`. Only human-reviewed or
  explicitly auto-approved pseudo-labels ever export.

## Setup

```bash
pip install -e ".[openai]"            # installs the openai SDK (lazy/optional)
export OPENAI_API_KEY=...             # read ONLY from env; never stored or printed
# (labeling also needs:  pip install -e ".[claude]"  and  export ANTHROPIC_API_KEY=...)
```
If the key or SDK is missing, the generator fails gracefully with setup steps — it
never makes a call and never prints the key.

## Prompt curriculum (categories + controllable mix)

The curriculum (`terafold/towel/prompt_bank.py`) groups categories by track. For
**pose** use the strict positive-only tasks; for **critic** data use
`critic_negatives_v1`; the legacy `mix_flat_wrinkled_negatives` (and `flat_only`,
`positives_only`, `hard_negatives`, `balanced`) remain for back-compat. Preview
before spending anything:

```bash
python3 -m terafold preview-openai-towel-prompts --task pose_positive_flat_only --num 20
```

`pose_positive_flat_only` mix: 50% flat top-down, 30% flat oblique, 20% mildly
wrinkled. `pose_positive_diverse_v1` is the same tracks with broader
surfaces/lighting/textures/colours.

**`pose_positive_unfolded_sheet_v1` is the strictest** (recommended when the model
still produces compact/folded-looking towels): 70% top-down, 20% near-top-down,
10% mild-wrinkle, all forced to be a **single-layer towel fully spread open as a
flat full-size rectangular sheet** (no folded stacks, no rolled/compact towels, no
thick stacked edges; towel occupies ~55–80% of the frame with all four corners
separated and visible). Probe with this task first if folded stacks appear.

## Generation QC

`filter-generated-towel-images` runs before labeling: file-validity / corrupt-image
removal, dimension checks, exact (content-hash) **and** perceptual (average-hash)
near-duplicate removal, and a metadata-completeness check. Kept images become a
candidate dataset (raw schema) that `claude-label-towel-folder` consumes directly.

## Recommended pose workflow — a 100-image pilot (copy-paste)

```bash
# 1. Generate STRICT pose positives (paid; ~$4 for 100 @ gpt-image-1 medium):
python3 -m terafold generate-openai-towel-dataset \
  --out data/towel_openai_pose_v0 --num-images 100 \
  --task pose_positive_flat_only --model gpt-image-1 --size 1024 \
  --max-cost-usd 8 --no-dry-run --yes-i-understand-this-uses-paid-api
#    (run once with --dry-run first to preview prompts + cost, no charge)

# 2. QC:
python3 -m terafold filter-generated-towel-images \
  --input data/towel_openai_pose_v0 --out data/towel_openai_pose_qc_v0

# 3. Claude-label (needs ANTHROPIC_API_KEY):
python3 -m terafold claude-label-towel-folder \
  --input data/towel_openai_pose_qc_v0 --out data/towel_openai_pose_labeled_v0 \
  --max-images 100 --min-confidence 0.75 --resume

# 4. Auto-triage (cheap approve/reject; preview with --dry-run first):
python3 -m terafold auto-triage-pseudolabels \
  --dataset data/towel_openai_pose_labeled_v0 --policy pose_positive_strict --dry-run
python3 -m terafold auto-triage-pseudolabels \
  --dataset data/towel_openai_pose_labeled_v0 --policy pose_positive_strict --apply

# 5. Manually review ONLY the remaining uncertain cases:
python3 -m terafold review-pseudolabels \
  --dataset data/towel_openai_pose_labeled_v0 --source openai_generated

# 6. Export approved pose labels only:
python3 -m terafold export-yolo-towel-pose \
  --data data/towel_openai_pose_labeled_v0 --out data/yolo_towel_pose_openai_pose_v0 \
  --include-pseudolabels approved_only

# 7. Merge with synthetic:
python3 -m terafold merge-towel-datasets \
  --inputs data/towel_synth_640_v1,data/towel_openai_pose_labeled_v0 \
  --out data/towel_master_pose_v0

# 8. Export master:
python3 -m terafold export-yolo-towel-pose \
  --data data/towel_master_pose_v0 --out data/yolo_towel_pose_master_v0 \
  --include-pseudolabels approved_only

# 9. Print the training command (does NOT auto-launch a job):
python3 -m terafold print-towel-training-command \
  --data data/yolo_towel_pose_master_v0 --model yolo11n-pose.pt --epochs 100 --imgsz 640
```

A **500-image** batch is identical — bump `--num-images 500 --max-cost-usd 25` and
add `--resume` (an interrupted run reuses already-saved images, never re-paid).

The exporter reports per-source (`synthetic`, `real_user`, `external`, `kaggle`,
`openimages`, `roboflow`, `openai_generated`, `claude_pseudolabel`),
per-`training_track`, and pseudo-label approval-status breakdowns. It
**excludes unapproved pseudo-labels in every mode** and **excludes
`critic_negative` samples by default**.

## auto-triage policy `pose_positive_strict`

Auto-**reject** if: state is folded_success / partially_folded / multiple_towels /
bad_view / not_towel; confidence < 0.75; missing or out-of-bounds corners; invalid
polygon; polygon area too tiny; towel fills/cuts the image border; risk reasons
mention hanging / folded / rolled / multiple / occluded / person / hands /
off-frame; or `training_track == critic_negative`.
Auto-**approve** only if: state is flat_unfolded / wrinkled_unfolded; confidence ≥
0.88; all 4 corners present + in bounds; plausible geometry; source is
openai_generated / real_user / external; no severe risk. **Everything else stays
`review_needed`.** Writes `auto_triage_report.md` with counts + reason histogram +
source/category/track breakdowns. `--dry-run` previews without touching labels.

## Critic data — don't throw the mixed batch away

Folded / hanging / rolled / multiple / not-towel images are **not** pose data, but
they are exactly the negatives a future **scene/rejection critic** needs ("is this
one flat foldable towel before I plan a fold?"). So:

- Generate them deliberately with `--task critic_negatives_v1` (track
  `critic_negative`); they never enter pose export by default.
- Turn a reviewed/triaged dataset's rejected + critic samples into a two-class
  critic set:

```bash
python3 -m terafold build-towel-critic-from-review \
  --dataset data/towel_openai_labeled_v1 --out data/towel_critic_openai_v1
```

This writes a manifest of `{image, state, approval_status, rejection_reason,
source, training_track, target_class}` where `target_class ∈ {usable, not_usable}`.
Keep it for the critic; keep it **out** of the YOLO 4-corner pose set.

## Recommended data mix / training strategy

- **Stage 0 — baseline:** synthetic only (smoke + pretrain).
- **Stage 1 — realism bridge:** synthetic + approved OpenAI-generated.
- **Stage 2 — real fine-tune:** add approved real/external towels.
- **Always:** keep a **real-only validation set** and a small curated **gold eval
  set** that are NEVER trained on, so generated data can't inflate your metrics.

First-pass target *ranges* (guidance, not gates): ~1000–2000 synthetic, ~500–2000
OpenAI-generated, ~50–300 approved real/external, plus a ~30–100 image gold set.
`recommend-towel-training-plan` prints these alongside your current composition.

## Local vs RunPod training — handoff plan

The dataset + commands are identical; only the machine differs. This sprint never
launches training automatically — it only prints the command. Recommended handoff:

```bash
# --- Local: package the exported YOLO dataset ---
tar -czf yolo_towel_pose_master_v0.tar.gz data/yolo_towel_pose_master_v0

# --- On RunPod (or any cloud GPU pod): ---
pip install ultralytics
tar -xzf yolo_towel_pose_master_v0.tar.gz
yolo pose train model=yolo11n-pose.pt \
  data=data/yolo_towel_pose_master_v0/towel_pose.yaml \
  epochs=100 imgsz=640 batch=16 project=runs/towel_pose name=towel_pose_master_v0

# --- After training ---
# 1. copy runs/towel_pose/towel_pose_master_v0/weights/best.pt back to your laptop
# 2. run local inference on a held-out generated/real test image:
python3 -m terafold infer-towel-pose --image ~/Downloads/hotel_towel_test.jpg \
  --weights best.pt --out runs/towel_pose/test.json --overlay-out runs/towel_pose/test.png
# 3. compare to the synthetic-only baseline (corner error / IoU) on the SAME
#    real-only gold set, using evaluate-towel-pose, before trusting the new model.
```

`print-towel-training-command` prints the exact local/Ultralytics command; the pod
runs the same one. Keep a real-only gold eval set so you can honestly compare the
generated-augmented model against the synthetic-only baseline.

## Hardware / safety: unaffected

This is a **perception-data** pipeline. It does not touch the robot motion safety
gates, the hardware unlock ladder, contact-fold logic, or the real-image ghost-fold
safety semantics. Contact folding stays LOCKED. No hardware is required or moved.
