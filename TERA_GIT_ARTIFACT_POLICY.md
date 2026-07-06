# TeraFold — Git Hygiene & Artifact Policy

Generated: 2026-07-05 — read-only audit (branch `main`, HEAD `2a16901`). Companion to [`REPO_AUDIT_TERA_ROBOTICS.md`](REPO_AUDIT_TERA_ROBOTICS.md), [`TERA_INTEGRATION_PLAN.md`](TERA_INTEGRATION_PLAN.md), [`TERA_NEXT_STEPS_V6.md`](TERA_NEXT_STEPS_V6.md).

**TL;DR — two urgent actions, both safe:**
1. **Restore a `/data/` ignore rule.** An uncommitted `.gitignore` edit deleted it, so 8.8 GB of `data/` is untracked *and* un-ignored — one `git add -A` from a gigabyte commit.
2. **Run `git gc --prune=now`.** `.git` has bloated to **1.5 GB, of which ~1.45 GB is unreachable garbage** from a prior staging accident. GC reclaims it; committed history is clean (2.86 MB of blobs).

**Good news:** no secrets are committed or in history; git *history* is clean (largest blob ever = 128 KB).

---

## 1. Repository size — where the 1.5 GB actually is

| Measure | Value | Meaning |
|---|---|---|
| `du -sh .git` | **1.5 GB** | On-disk git store |
| Loose objects | 4508 objects, ~1.45 GiB | All loose (nothing packed) |
| **Reachable** blob bytes (all history) | **2.86 MiB** (415 blobs) | The actual committed content is tiny |
| **Unreachable** objects (`git fsck`) | **4081 blobs**, 1 commit, 11 trees | Dangling garbage — not in any commit |
| Largest blob **ever in history** | **128 KB** (`terafold/cli.py`) | No large file was ever committed |

**Diagnosis:** someone almost certainly ran `git add` on large files (`data/` images and/or a tarball) then reset/unstaged without committing. The blobs were written into `.git/objects` and never garbage-collected. The **committed history is clean and small** — this is purely reclaimable cruft.

**Fix (safe — only touches unreachable objects):**
```bash
git gc --prune=now
# if anything survives (reflog), then:
git reflog expire --expire=now --all && git gc --prune=now
```
Expected result: `.git` drops from 1.5 GB to well under 20 MB.

---

## 2. Largest **tracked** files (all fine — no LFS needed for source)

| Size | File |
|---|---|
| 128 KB | `terafold/cli.py` (89 commands in one file — a maintainability smell, not a git problem) |
| 52 KB | `terafold/sim/fold_sim.py` |
| 52 KB | `deep-research-report.md` |
| 44 KB | `terafold/towel/openai_multipass.py`, `terafold/towel/claude_pseudolabel.py` |
| 40 KB | `terafold/data/hf_streaming.py`, `README.md` |

Every tracked file is small text/code. **Git LFS is not needed for anything currently tracked.**

---

## 3. The uncommitted `.gitignore` edit — show, explain, decide

```diff
-# TeraFold runtime artifacts (large / regenerable — never commit)
-# /data holds raw/generated datasets (GBs); GitHub caps files at 100 MB anyway.
-# (anchored to repo root so terafold/data — a Python package — stays tracked)
-/data/
+# TeraFold runtime artifacts (regenerable — never commit)
 runs/
 *.pt
 *.ckpt
```
The edit **deleted the `/data/` rule** (and the very comment warning that `data/` is GBs and GitHub caps files at 100 MB) and added `*.tar.gz`. Confirmed effect: `git check-ignore data/towel_real_v1/images/000001.jpg` → **NOT IGNORED**; `git status` shows `?? data/`.

**Recommendation: do NOT commit this edit as-is. Restore a `/data/` ignore** (see the consolidated block in §7). If the intent was to start tracking a *curated* subset of `data/`, do that deliberately with a force-add of specific small files (`git add -f`) plus an off-repo/LFS plan for anything large — never by un-ignoring the whole 8.8 GB tree. Note the `/data/` anchoring is correct and must be preserved: it ignores the root `data/` dataset dir while keeping `terafold/data/` (a Python package) tracked.

---

## 4. What should / should not be committed

**COMMIT (source of truth):** `terafold/` (all `.py`), `scripts/`, `configs/` (small YAML), `docs/` (`.md`), `tests/`, `README.md`, `goal.md`, `deep-research-report.md`, `pyproject.toml`, `.gitignore`, and the four `TERA_*`/`REPO_AUDIT_*` reports. **Add a `LICENSE` file** (Apache-2.0 is declared but the file is missing). Optionally add small tracked evidence: `docs/brev_status/*_STATUS.json` (Brev milestone evidence) and a `vendor/README.md` (see §6).

**NEVER COMMIT:**

| Path / class | Size | Why |
|---|---|---|
| `data/` | 8.8 GB | Datasets; regenerable or precious→back up off-repo (§5). **Currently un-ignored — fix first.** |
| `runs/` | 77 MB | Run artifacts + test pollution. Already ignored. Back up the 6 weight files separately. |
| `*.tar.gz` (root) | ~1.9 GB | `yolo_towel_pose_master_v0.tar.gz` (1.45 GB) and `yolo_towel_pose_synth`… dup `data/`; `terafold_yam_sprint_CODE_ONLY.tar.gz` dups git (+ ships `__pycache__`). Already ignored. |
| `*.pt`/`*.ckpt`/`*.safetensors` | — | Weights (incl. `yolo11n-pose.pt`, `runs/**/*.pt`). Already ignored. |
| `*.mp4` | 12 files in `runs/sim/` | Kinematic-animation videos. Already ignored. |
| `vendor/` | 35 MB | Windows venv + Waveshare SDK. Already ignored; rescue the SDK + scripts (§6). |
| `*.egg-info/`, caches | — | Build/tool caches. Already ignored; **`terafold.egg-info` is correctly not tracked.** |

**Binary artifact classes on disk (verified):** 12 `.mp4` (in gitignored `runs/`); **zero** `.usd`/`.usda`/`.rrd`/`.npz` anywhere locally. So beyond `.pt` and `.mp4` (both ignored), there are no stray binary classes to worry about *yet* — but the incoming Brev work will add `.usd` scenes and `.rrd` traces, which the block in §7 pre-emptively ignores.

**Git LFS verdict: not needed.** Nothing tracked warrants it, and the large artifacts are *regenerable or belong in dataset/model storage*, not git. Prefer **external storage** (HF dataset/model repos, S3/GCS, or a synced drive) over LFS: it avoids LFS bandwidth costs and keeps git for code. Only reconsider LFS if you later decide to version a small, stable set of curated real images.

---

## 5. Precious vs regenerable — and the backup that must happen first

Roughly **2/3 of the 8.8 GB `data/` tree is derivable copies.** Exactly one thing is irreplaceable.

| Class | Dirs | Regenerable? | Action |
|---|---|---|---|
| **PRECIOUS** | `data/raw_towel_real_v1/` — 105 iPhone **HEIC** originals, 298 MB | **No** — real-world capture | **Back up NOW** off-repo (see below). Single copy on this laptop; no HEIC tarball exists. |
| API-COST | OpenAI-generated (`towel_openai_pose_v0` etc., ~$6.9) + Claude labels (~$0.9) | Only by re-spending API $ | Keep off-repo; low priority |
| REGENERABLE | synthetic (`towel_synth_*`), YOLO exports, merges, HF/OpenImages downloads | Yes (seeds recorded) | Safe to delete/prune |
| MOCK | `data/episodes/*` (`MockRobot dry_run`), `lerobot/*`, `yam_episodes/*` (`dummy:true`) | Yes | Schema reference only |

**Also single-copy and unbacked-up (outside `data/`):**
- **6 trained weight files (~20 MB)** in gitignored `runs/` — the two YOLO `best.pt`/`last.pt` pairs + two keypoint `model.pt`. A `rm -rf runs/` (tempting, given pytest pollutes it) destroys all six plus the training-metric CSVs.
- **Waveshare bring-up scripts + `scservo_sdk`** in gitignored `vendor/` — see §6.

**Backup recommendation (do before any cleanup or git surgery):**
```bash
# precious raw photos + the trained weights + the vendor bring-up work, off-repo
cp -a data/raw_towel_real_v1/ ~/tera-backup/raw_towel_real_v1/            # 105 HEIC
cp -a runs/towel_pose runs/pose runs/keypoints_v0 runs/keypoints_so101_ft_v0 ~/tera-backup/weights/
cp -a vendor/waveshare/STServo_Python/stservo-env/{scservo_sdk,safe_*.py,sms_sts/ping_mac.py} ~/tera-backup/vendor/
# then verify the HEICs also still exist in Photos/iCloud (UNKNOWN whether they do)
```

---

## 6. Secrets scan — clean, with recommendations

**Result: no secrets are committed or in git history.** Every match for key-like patterns is one of: a doc placeholder (`export ANTHROPIC_API_KEY=...`), an env-only read (`os.environ`/`getenv`), a status string (`"no_api_key"`), or a docstring promising keys are never stored/printed. Specifically verified:
- No `.env*` file anywhere in the repo.
- No key-like strings in tracked `.py/.sh/.json/.yaml/.toml/.md` beyond env reads and doc placeholders.
- No keys in the untracked `data/` manifests (which log OpenAI/Claude runs — only token counts and model names, e.g. `claude-opus-4-8`, `gpt-5.5`).
- Only a `"sk-test"` literal, in a test.

**The codebase is genuinely disciplined about keys:** `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `KAGGLE_*` are read from env only, missing keys degrade to a graceful status, and error strings redact the key (`openai_generation._sanitize`).

**Recommendations:** (a) add a tracked `.env.example` listing the env-var *names* only, to document what is needed without ever holding a value; (b) confirm `.gitignore` covers `.env` (the block in §7 does); (c) before any *new* paid pipeline run, re-verify the model IDs (`claude-opus-4-8`, `gpt-5.5`, `gpt-image-1`) are still valid — manifests prove they worked in late June 2026 but they are post-cutoff names.

**Vendor rescue (compliance + durability):** `vendor/` is correctly gitignored, but the repo's *real-hardware backend depends on it* (`waveshare_sms_sts_backend.py` sys-path-injects `vendor/waveshare/.../scservo_sdk`), so a fresh clone silently loses all Waveshare capability. And the 8 local bring-up scripts (`safe_*.py`, `ping_mac.py`) — the only record of how real motion was first exercised, incl. the ID→joint mapping procedure — live *only* there. **Recommendation:** keep the Windows venv guts ignored, but (1) commit a tracked `vendor/README.md` (via `!vendor/README.md` negation) recording the Waveshare download URL and the zip sha256 `f39b7a07…`; (2) move the 8 bring-up scripts into a tracked `docs/hardware-bringup/`; (3) **check the Waveshare SDK license before** vendoring `scservo_sdk` into a tracked path — no LICENSE/COPYING was found in its tree. Note: the `safe_*.py` "safe" prefix means *gated and tiny*, not read-only — 4 of the 6 send real single-servo motion.

---

## 7. Recommended consolidated `.gitignore` (ready to paste)

Restores `/data/`, keeps the good existing rules, and pre-emptively covers the incoming Brev artifact classes (`.usd`, `.rrd`, `.npz`). Adjust the `!` negations to taste.

```gitignore
# ---- Python ----
__pycache__/
*.py[cod]
*.egg-info/
*.egg
.eggs/
build/
dist/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/

# ---- Virtual environments ----
.venv/
venv/
env/

# ---- Secrets (never commit) ----
.env
.env.*
!.env.example

# ---- TeraFold runtime artifacts (large / regenerable — never commit) ----
# /data holds raw+generated datasets (GBs); GitHub caps files at 100 MB.
# Anchored to repo root so terafold/data (a Python package) stays tracked.
/data/
runs/
tera_checkpoints/
*.pt
*.pth
*.ckpt
*.safetensors
*.mp4

# ---- Isaac / sim / viz artifacts (incoming from Brev) ----
*.usd
*.usda
*.usdc
*.rrd
*.npz

# ---- Archive snapshots (dataset tarballs exceed GitHub's limit; code tarballs duplicate git) ----
*.tar.gz
*.zip

# ---- OS / editor ----
.DS_Store
.idea/
.vscode/
*.swp

# ---- Keep small sample/config assets if explicitly added with `git add -f` ----
!configs/*.yaml
!configs/**/*.yaml

# ---- Vendored SDKs / virtualenvs (machine-specific) — but keep the README pointer ----
vendor/
!vendor/README.md
```

**Note on `*.zip`:** this newly ignores `vendor/waveshare/STServo_Python.zip` (already inside the ignored `vendor/`, so no change) — harmless. If you ever want to track a specific config or sample that a broad rule catches, use `git add -f <path>`.

---

## 8. One-page action list

1. **Back up** the 105 HEIC photos + 6 weight files + vendor bring-up scripts off-repo (§5). *Do this first.*
2. **Restore `/data/`** in `.gitignore` using the §7 block. Verify `git status` no longer shows `?? data/`.
3. **`git gc --prune=now`** to reclaim ~1.45 GB (§1). Verify `du -sh .git`.
4. **Add a `LICENSE`** file (Apache-2.0) and a `.env.example` (names only).
5. **Rescue vendor** — tracked `vendor/README.md` (URL + sha256) and move `safe_*.py` into `docs/hardware-bringup/`; check the SDK license (§6).
6. **Prune ~5–6 GB** of derivable duplicate copies in `data/` after backup, keeping only sources of truth (HEIC + manifests + generators). Optional.
7. For the Brev merge, commit small `STATUS.json` evidence, keep weights/USD/`.rrd` off-repo. See [`TERA_INTEGRATION_PLAN.md`](TERA_INTEGRATION_PLAN.md).
