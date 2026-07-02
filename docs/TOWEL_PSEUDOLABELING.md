# External towel images + Claude pseudo-labeling

This pipeline grows the towel **corner-pose** training set with *external real
images* — ingested from Kaggle / Open Images — and pseudo-labels them with
**Claude Vision**, while keeping a human in the loop and the hardware safety gates
untouched.

It plugs into the existing real-image-first pipeline (see README §5c): pseudo-labels
become normal towel labels (`tl/tr/br/bl` corners + state) that flow through
`merge-towel-datasets` → `export-yolo-towel-pose` → YOLO pose training — but only
**after approval**.

```
import-kaggle-towel-dataset ─┐
                             ├─▶ filter-towel-images ─▶ claude-label-towel-folder ─▶ review/approve ─▶ export-yolo-towel-pose
import-openimages-towels ────┘   (cheap pre-filter)     (Claude Vision, geometry)    (human / gate)     (--include-pseudolabels approved_only)
```

> ⚠️ **Pseudo-labels can be wrong.** Nothing Claude produces is trusted by
> default: every label starts `usable_for_training: false`, and the YOLO exporter
> **excludes unapproved pseudo-labels** unless you explicitly pass
> `--include-pseudolabels approved_only` *and* the label was approved.

---

## 0. One-time setup

### Kaggle

```bash
pip install kaggle kagglehub
# then add credentials (either one):
#   * place kaggle.json at ~/.kaggle/kaggle.json   (chmod 600)
#   * or export KAGGLE_USERNAME=... and KAGGLE_KEY=...
# get the token at https://www.kaggle.com/settings -> "Create New API Token"
```

If neither `kaggle` nor `kagglehub` is installed (or credentials are missing), the
importer fails **gracefully** and prints these exact steps — it never touches the
network implicitly.

### Open Images

Automatic bulk download is intentionally **not** implemented. Download the `Towel`
class once, then ingest the local folder:

```bash
pip install fiftyone
python -c "import fiftyone.zoo as foz; foz.load_zoo_dataset('open-images-v7', split='train', label_types=['detections'], classes=['Towel'], max_samples=2000, dataset_dir='open_images_towel')"
# images land under open_images_towel/data/
```

**Open Images gives image-level labels / bounding boxes — NOT towel corners.** You
still need Claude labeling + human review to get the 4 corners this pipeline trains on.

### Anthropic API key (for Claude labeling)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...      # get one at https://console.anthropic.com/
```

The key is read **only** from the `ANTHROPIC_API_KEY` environment variable. It is
never hard-coded, never written to disk, and never printed. If it is missing,
`claude-label-towel-folder` exits with setup instructions and does nothing.

---

## ⚠️ Cost, privacy & "why human review matters"

- **Cost.** Each image is one Claude Vision request. The labeler records token
  usage and a *rough* cost estimate in the dataset manifest (`usage`,
  `cost_estimate_usd`) — the estimate is approximate; verify current
  [Anthropic pricing](https://www.anthropic.com/pricing). Filter first
  (`filter-towel-images`) and cap with `--max-images` so you don't spend tokens on
  junk. Use `--resume` to avoid re-paying for images already labeled.
- **Privacy.** Images you ingest are **sent to the Anthropic API**. Do not send
  private, personal, or non-redistributable photos. Only ingest datasets whose
  license permits your use; provenance + a `license_note` are recorded in every
  manifest, but verifying the license is your responsibility.
- **Why human review matters.** Claude is a strong labeler but **not perfect**: it
  can hallucinate corners on occluded/folded towels, mislabel decorative towel
  sculptures as flat towels, or be over-confident. Training a corner-pose detector
  on wrong corners quietly degrades the model. So pseudo-labels are gated: a human
  (`review-pseudolabels`) or an explicit high-confidence geometry gate
  (`approve-high-confidence-pseudolabels`) must approve before export.

---

## 1. Exact Kaggle pipeline

```bash
# Ingest a Kaggle towel dataset into a RAW image dataset (no labels yet).
python3 -m terafold import-kaggle-towel-dataset \
  --dataset owner/dataset-slug \
  --out data/towel_kaggle_raw_v0 \
  --max-images 2000

# Already downloaded it? Ingest the local export offline (no Kaggle client needed):
python3 -m terafold import-kaggle-towel-dataset \
  --dataset owner/dataset-slug --src-dir ~/Downloads/kaggle_towels --out data/towel_kaggle_raw_v0
```

Recursively finds JPG/JPEG/PNG/WebP, skips tiny images (`--min-size`),
deduplicates by content hash, and writes `manifest.json` with the source, dataset
slug, original path, license note, image hash, and width/height. Originals are
copied (bytes, never decoded); your source folder is untouched.

## 2. Exact Open Images / manual pipeline

```bash
# Prints the manual FiftyOne / official-downloader steps (and explains box != corners):
python3 -m terafold import-openimages-towels --out data/towel_openimages_raw_v0 --max-images 2000

# After downloading the 'Towel' class, ingest the local folder:
python3 -m terafold import-openimages-towels \
  --src-dir open_images_towel/data --out data/towel_openimages_raw_v0 --max-images 2000
```

Output uses the **same raw image manifest schema** as the Kaggle importer.

## 3. Filter to likely-towel candidates (cheap, before paying for Claude)

```bash
python3 -m terafold filter-towel-images \
  --input data/towel_kaggle_raw_v0 \
  --out data/towel_candidates_v0 \
  --mode filename_or_vlm \
  --max-images 1000
```

Keeps images whose filename/path contains a towel keyword (`towel`, `bath_towel`,
`hotel_towel`, `white_towel`, `linen`, ...), marks obvious non-image / unreadable
files, deduplicates, and writes a candidate manifest. **Originals are never
deleted.** (`--mode filename` skips the optional VLM screen entirely.)

## 3b. Crop-first localization for REAL cluttered photos (recommended)

Real phone photos of a towel on a bed confuse the one-shot corner labeler: Claude
keeps placing the 4 "corners" on the **bed / headboard / room rectangle** instead of
the small striped towel. Fix it by making localization its own cheap step — find the
towel's box first, crop to it, *then* label the crop. Corners labeled on the crop are
remapped back to the original full image automatically.

```bash
export ANTHROPIC_API_KEY=...

# 1) Localize the target towel ONLY (ignore bed/pillow/blanket/headboard/floor/body),
#    save a tight padded crop + remap metadata, and an overlay on each full image:
python3 -m terafold crop-real-towel-candidates \
  --input data/towel_real_v1 \
  --out data/towel_real_v1_crops \
  --target-description "beige and white striped towel"

# 2) Label the crops with the striped-towel-only policy. Corners are remapped back to
#    the full images; you get overlays on BOTH the crop and the original photo:
python3 -m terafold claude-label-towel-folder \
  --input data/towel_real_v1_crops \
  --out data/towel_real_labeled_v1_crops \
  --min-confidence 0.65 \
  --label-policy outer_visible_corners_striped_towel_only \
  --force
```

The crop step asks Claude for **only a bounding box** for the target object and self-
reports `is_target`; a box that is actually the bed/pillow/room (or low confidence, or
not found) is **rejected** — no crop is written, and the rejection is recorded in the
crop manifest. Each kept crop stores `crop` metadata (`x_off`, `y_off`, `scale_x/y`)
so the labeler can map crop-space corners back to original-image coordinates. The
`outer_visible_corners_striped_towel_only` policy reuses the strict visible-outer-
corner rules, pins the target to the striped towel, and rejects (`not_striped_towel`,
state `not_towel`) when the crop is not the towel. Labels land in **crop** coordinates
on the copied crop image, with the remapped full-image corners under `full_image` and
a full-image overlay at `full_overlays/<id>_full_overlay.png`.

## 4. Pseudo-label with Claude Vision

```bash
export ANTHROPIC_API_KEY=...
python3 -m terafold claude-label-towel-folder \
  --input data/towel_candidates_v0 \
  --out data/towel_pseudolabeled_v0 \
  --max-images 500 \
  --model claude-opus-4-8 \
  --min-confidence 0.75 \
  --resume
```

For each image Claude returns **strict JSON only**: `state`, `usable_for_training`,
`bbox`, 4 `corners` (`tl/tr/br/bl`) + visibility, `fold_axis`, `grasp_edge`,
`place_edge`, `confidence`, `risk_reasons`, `notes`. The prompt asks Claude to label
**one main towel**, prefer a flat/slightly-wrinkled white/off-white hotel towel, and
mark folded stacks / decorative / unclear towels as not usable. Claude is **never**
asked for robot commands — only pixel geometry.

Each response is validated strictly: malformed JSON is rejected; coordinates are
checked against the image bounds (and clamped); the corner quad is checked for
impossible (self-crossing / degenerate) geometry. A label is routed to
`review_needed` when confidence `< --min-confidence`, the state is
`bad_view`/`multiple_towels`/`not_towel`/`folded_success`, corners are missing, the
bbox is invalid, or the geometry is impossible. Every image gets a label JSON, an
overlay (`overlays/<id>_overlay.png`), and a confidence; token usage + a rough cost
estimate are saved to the manifest. `--resume` skips images already labeled in
`--out`.

## 4b. Second opinion: OpenAI Vision labeler

When Claude's corners drift onto the bed / headboard / crop border instead of the
beige/white striped towel, label the same images with **OpenAI Vision** and compare.
It produces the **identical towel label schema**, validation, overlays, and review
gating — only the backend differs (OpenAI's Responses API with image input). Test on
just a few images first with `--max-images`:

```bash
export OPENAI_API_KEY=...
python3 -m terafold openai-label-towel-folder \
  --input data/towel_real_v1 \
  --out data/towel_real_labeled_openai_test \
  --model gpt-5.5 \
  --max-images 3 \
  --label-policy striped_towel_visible_outer_corners \
  --force
```

OpenAI returns strict JSON with `is_target_towel_visible`, `reject_reason`, `state`,
`corners` (`tl/tr/br/bl`) + `visible`, `confidence`, and `risks`. The prompt pins the
target to the striped towel, lists the distractors to ignore (bed sheet, mattress
cover, pillow, blanket, headboard, floor, body/feet), and gives the visible-outer-
corner rule (label the towel's current visible outline — **not** the image/crop
border, bed rectangle, or a bounding box; for a folded towel label the folded
rectangle; for a rounded corner, the intersection of the two outer edges). The reject
rule (`is_target_towel_visible=false` → `not_striped_towel`, state `not_towel`) keeps
bed/background labels out of training. Labels are stamped `source: openai_pseudolabel`
/ `label_backend: openai`, start `usable_for_training: false`, and flow through the
same `review-pseudolabels` / `approve-high-confidence-pseudolabels` gates as Claude's.
The key is read only from `OPENAI_API_KEY` (never stored or printed).

## 4c. High-precision multi-stage labeling (`--multipass`)

Single-pass corner labeling on a full real-world photo still drifts toward loose,
bounding-box-ish corners — the model sees the whole scene and brackets the towel.
`--multipass` trades tokens for accuracy: it spends **multiple vision calls per
image** to pin the 4 corners tightly on the striped towel. Use `--max-images 3` to
test a few first (the goal is accurate labels, not cheap ones).

```bash
export OPENAI_API_KEY=...
python3 -m terafold openai-label-towel-folder \
  --input data/towel_real_v1 \
  --out data/towel_real_labeled_openai_multipass_test \
  --model gpt-5.5 \
  --max-images 3 \
  --label-policy striped_towel_visible_outer_corners \
  --multipass \
  --force
```

Per image the pipeline runs: **(1) localize** the striped towel only (tight bbox,
ignore bed/mattress/pillow/blanket/headboard/floor/body) → **(2) crop + zoom**
(`--crop-padding`, saved crop) → **(3) corner-label the crop** (4 visible outer
corners, not the crop/image border, not a bbox; rounded corner = edge intersection)
→ **(4) remap** the crop keypoints back to original image coordinates → **(5) verify**
(`--verify`/`--no-verify`): the original + proposed keypoints go back to the model,
which answers *all on the towel? any on background? just a loose bbox? tight enough
for YOLO?* and **accepts / corrects / rejects** → **(6) automatic geometry checks**:
hard-reject if the corners fill the crop/image border, sit on the image edge, form an
impossible quad, or spill far outside the detected towel region; flag axis-aligned /
slightly-outside corners for review.

Each image saves a full artifact trail for auditing: `localize_overlays/<id>.png`
(detected bbox + crop region), `crops/<id>_crop.png`, `crop_overlays/<id>_crop.png`
(corners on the crop), `overlays/<id>_overlay.png` (final corners remapped onto the
full image), `verify/<id>_verify.json`, and `labels/<id>.json` (corners in **original
image coordinates**). Labels still start `usable_for_training: false` and route
through the same review/approve gates.

## 5. Review / approve, then export & train

```bash
# A) Human review one-by-one (approve / reject / edit state / edit corners):
python3 -m terafold review-pseudolabels --dataset data/towel_pseudolabeled_v0
#    -> writes review_report.md (approved / rejected / needs-review + common reasons)

# B) OR auto-approve ONLY high-confidence, geometrically-sound labels (rest stay for review):
python3 -m terafold approve-high-confidence-pseudolabels \
  --dataset data/towel_pseudolabeled_v0 --min-confidence 0.90 --max-geometry-error 0.10

# Merge with your real dataset (dedup + split), then export approved pseudo-labels:
python3 -m terafold merge-towel-datasets \
  --inputs data/towel_real_v0,data/towel_pseudolabeled_v0 --out data/towel_combined_v0
python3 -m terafold export-yolo-towel-pose \
  --data data/towel_combined_v0 --out data/yolo_towel_pose_v1 \
  --include-pseudolabels approved_only

# Train + dry-run exactly as in README §5c:
python3 -m terafold print-towel-training-command --data data/yolo_towel_pose_v1 --model yolo26n-pose.pt --epochs 100 --imgsz 640
```

The exporter reports a per-source breakdown (`real_user`, `kaggle`, `openimages`,
`roboflow`, `synthetic`, `claude_pseudolabel`). **Pseudo-labels are excluded by
default**; `--include-pseudolabels approved_only` adds only approved / auto-approved
ones — unapproved pseudo-labels can never be exported, in any mode.

---

## Safety invariants

- The hardware safety gates and contact-fold lock are **untouched** — this is a
  perception/data pipeline only.
- Claude is asked for image geometry only, never robot/motor/serial commands.
- API keys are read only from `ANTHROPIC_API_KEY`; never stored, never printed.
- Pseudo-labels are never `usable_for_training` until approved; the YOLO exporter
  double-gates on both the approval status and the `--include-pseudolabels` policy.
