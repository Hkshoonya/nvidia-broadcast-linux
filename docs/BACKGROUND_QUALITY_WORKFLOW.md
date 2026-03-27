# Background Quality Workflow

Use the local artifact workflow when tuning background replacement quality.

## 1. Capture a Small Real Dataset

```bash
./.venv/bin/python scripts/capture_quality_dataset.py \
  --camera /dev/video0 \
  --count 8 \
  --interval 1.0 \
  --label baseline \
  --note "face-on, motion, glasses"
```

This creates a timestamped directory under `artifacts/quality_dataset/` with:

- `frame_001.png`, `frame_002.png`, ...
- `manifest.json`

The optional `--label` is appended to the timestamped folder name.

Try to include:

- neutral face-on framing
- hand motion
- head turn
- hair edge against bright background
- glasses
- dark scene

## 2. Evaluate One Frame

```bash
./.venv/bin/python scripts/eval_background_quality.py \
  --input artifacts/quality_dataset/<session>/frame_001.png \
  --preset doczeus \
  --mode replace \
  --out artifacts/quality_eval/frame_001_doczeus
```

Artifacts written:

- `raw.png`
- `alpha.png`
- `replacement_matte.png`
- `edge_aware_matte.png`
- `composite.png`
- `metrics.json`

For green-screen evaluation:

```bash
./.venv/bin/python scripts/eval_background_quality.py \
  --input artifacts/quality_dataset/<session>/frame_001.png \
  --preset doczeus \
  --mode remove \
  --out artifacts/quality_eval/frame_001_doczeus_remove
```

This writes `greenscreen_matte.png` instead of the replace-specific mattes.

## 3. Sweep Modes on One Frame

```bash
./.venv/bin/python scripts/sweep_background_modes.py \
  --input artifacts/quality_dataset/<session>/frame_001.png \
  --effect-mode replace \
  --out artifacts/mode_sweep/frame_001
```

This compares steady-state timing and edge metrics across app presets.

## 4. Evaluate the Whole Dataset

```bash
./.venv/bin/python scripts/eval_background_dataset.py \
  --dataset artifacts/quality_dataset/<session> \
  --effect-mode replace \
  --out artifacts/dataset_eval
```

This aggregates mean inference time, composite time, and edge-band metrics
for each mode across all captured frames.

Run the same command with `--effect-mode remove` for green-screen mode. Results
are stored under `artifacts/dataset_eval/replace/<session>/...` and
`artifacts/dataset_eval/remove/<session>/...`.

## 5. Prepare a Training Bundle

```bash
./.venv/bin/python scripts/prepare_matting_training_bundle.py \
  --dataset artifacts/quality_dataset/<session> \
  --preset doczeus \
  --models birefnet isnet rvm \
  --variants replace remove \
  --out artifacts/training_bundle
```

This creates a timestamp-aligned bundle with:

- `images/`
- `masks/replace/` and `masks/remove/`
- `trimaps/replace/` and `trimaps/remove/`
- `train.jsonl`
- `val.jsonl`
- `summary.json`

The model list is tried in priority order per frame. This is meant for
pseudo-label generation, so slower CPU fallback models are acceptable if they
produce tighter masks.

## 6. Train a Lightweight Learned Refiner

```bash
./.venv/bin/python scripts/train_edge_refiner.py \
  --bundle artifacts/training_bundle/<bundle> \
  --variant replace \
  --epochs 20 \
  --install
```

Run the same command with `--variant remove` to train the green-screen path.
When `--install` is used, the exported ONNX is copied into:

- `models/edge_refiner_replace.onnx`
- `models/edge_refiner_remove.onnx`

The runtime keeps these disabled by default. To test them, launch the app with:

```bash
NVBROADCAST_ENABLE_LEARNED_REFINER=1 ./.venv/bin/python -m nvbroadcast
```

That gate is intentional: the learned refiners should only be enabled after
they beat the heuristic baseline on the captured dataset.

## 7. Rank Hard Cases

```bash
./.venv/bin/python scripts/analyze_training_bundle.py \
  --bundle artifacts/training_bundle/<bundle> \
  --variant replace
```

This ranks the worst coarse-vs-teacher examples by edge-band L1 error so you
can see which captured frames are still failing hardest and collect more data
for those cases specifically.

## Current Interpretation

- `DocZeus` is the quality-first GPU mode.
- `Killer` and `Zeus` are speed modes.
- `edge_mid_fraction` and `edge_low_mid_fraction` are useful halo proxies:
  lower is usually better if the silhouette does not collapse.
- Green-screen mode now uses its own `greenscreen_matte`; compare that metric
  separately from replacement-mode mattes.
