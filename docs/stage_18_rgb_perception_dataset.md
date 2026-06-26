# Stage 18: RGB Perception Dataset & Detector Training

> **Version**: v0.1.0 (Design)
> **Date**: 2026-05-21

---

## 1. Motivation

Stage 17 introduced oracle scene inventory — ground-truth object state read
directly from the Robosuite simulator. This gives an upper bound on
scene-aware parsing but does not generalise to the real world.

Stage 18 builds the first step toward **vision-based perception**: a labelled
RGB dataset with oracle-generated ground truth, collected from the same
simulation environment used for E2E evaluation. The dataset enables training
a lightweight object detector that can eventually replace the oracle.

**Key principle**: Labels are generated from simulator state, not manually
annotated. This is feasible because the oracle has full access to object
poses, colours, and identities.

---

## 2. Dataset Schema

### 2.1 Location

`generated/datasets/rgb_perception_v0_1/`

### 2.2 Dataset Structure

```
rgb_perception_v0_1/
├── dataset_meta.json          # Global metadata
├── annotations.jsonl          # All annotations (one JSON object per image)
├── images/
│   ├── step_000000.png
│   ├── step_000001.png
│   └── ...
├── train.txt                  # Image filenames for training split
└── val.txt                    # Image filenames for validation split
```

### 2.3 Annotation Schema (annotations.jsonl)

Each line is a JSON object with the following fields:

| Field | Type | Description |
|:------|:-----|:------------|
| `image_path` | str | Relative path to PNG image |
| `episode_id` | int | Episode identifier (0-indexed reset) |
| `step_id` | int | Step within the episode |
| `camera_name` | str | Camera used ("agentview") |
| `objects` | list[dict] | Detected objects in the scene |
| `scene_source` | str | `"oracle"` |
| `image_width` | int | Width in pixels |
| `image_height` | int | Height in pixels |

Each object dict:

| Field | Type | Description |
|:------|:-----|:------------|
| `object_id` | str | "cube1", "cube2", "table" |
| `category` | str | "block" or "table" |
| `color` | str | "red", "green", "blue", etc. |
| `position` | dict | `{"x": ..., "y": ..., "z": ...}` in sim coordinates |
| `bbox` | dict or null | `{"x1": int, "y1": int, "x2": int, "y2": int}` pixel coords (optional) |
| `center` | dict or null | `{"x": int, "y": int}` pixel center (optional) |
| `visibility` | str or null | "visible", "partially_visible", "occluded" |

### 2.4 dataset_meta.json

```json
{
  "dataset_name": "rgb_perception_v0_1",
  "dataset_version": "0.1.0",
  "created_at": "2026-05-21",
  "source": "Robosuite Stack env, oracle labels",
  "total_images": 100,
  "image_size": {"height": 512, "width": 512},
  "camera": "agentview",
  "num_episodes": null,
  "objects": ["cube1", "cube2", "table"],
  "categories": ["block", "table"],
  "colors": {"cube1": "red", "cube2": "green"},
  "split": {"train": 80, "val": 20},
  "has_bbox": false,
  "has_center": false,
  "purpose": "RGB perception training — object detection from oracle labels",
  "notes": "Smoke dataset for detector proof-of-concept. Bbox/center are optional and can be added via projection."
}
```

---

## 3. Collection Strategy

### 3.1 Procedure

1. Initialise Robosuite Stack environment with `render_mode="offscreen"`.
2. For each image:
   - Call `env.render_rgb(camera="agentview", height=512, width=512)` for RGB.
   - Call `oracle_perception.perceive_scene(env)` for ground truth.
   - Save image to `images/step_NNNNNN.png`.
   - Append annotation to `annotations.jsonl`.
3. After collection, split into `train.txt` / `val.txt` (80/20 random).
4. Write `dataset_meta.json`.

### 3.2 Scene Variation

The collection script should capture a range of states:

- **Random initial positions**: Cubes placed at various (x, y) on the table.
- **Grasped states**: Gripper holding a cube (simulate mid-episode).
- **Stacked states**: One cube on top of another (on relation active).
- **Empty table**: No cubes (just table).
- **Single cube**: Only one cube present.

Variation is achieved by:
1. Running short random-action episodes (random policy for N steps).
2. Saving frames periodically from each episode.

### 3.3 Camera Configuration

- Camera: `"agentview"` (robosuite default overhead view).
- Resolution: 512×512 (balance between detail and storage).
- Format: PNG (lossless, small for sim renders).

---

## 4. Collection Script

### 4.1 CLI

```
python -m vlm_nsm.experiments.collect_rgb_perception_dataset \
  --output_dir generated/datasets/rgb_perception_v0_1 \
  --num_images 100 \
  --camera agentview \
  --height 512 \
  --width 512 \
  --seed 0
```

### 4.2 Implementation

- Uses `RobosuiteEnvWrapper` with `render_mode="offscreen"`.
- Generates `num_images` by running random-action rollouts.
- Calls `render_rgb()` and `perceive_scene()` at each step.
- Saves images and annotations.
- Handles render failures gracefully.

See `src/vlm_nsm/experiments/collect_rgb_perception_dataset.py`.

---

## 5. Detector Training Plan

### 5.1 Recommended Architecture

**YOLOv8n** (nano) for initial proof-of-concept:
- 3.2M parameters — fast training and inference.
- Native PyTorch, easily exportable to ONNX/TensorRT.
- Single-class detection initially (detect "block" vs "table").
- Extend to instance-level (cube1 vs cube2) later.

Alternative: Lightweight CNN with regression head:
- Smaller than YOLO, runs at higher FPS.
- Directly regresses object centre + category from RGB.
- Easier to integrate into the existing pipeline.

### 5.2 Training Pipeline

1. **Split**: 80/20 train/val.
2. **Preprocessing**: Normalise to [0, 1], resize to 224×224 if using CNN.
3. **Augmentation**: Random horizontal flip, brightness jitter, small crop.
4. **Loss**: Classification (cross-entropy) + bbox regression (MSE / IoU loss).
5. **Epochs**: 100 (YOLO default) or until validation loss plateaus.
6. **Evaluation**: Compare predicted scene vs oracle scene on val set.

### 5.3 Evaluation Protocol

| Metric | Target |
|:-------|:-------|
| mAP@0.5 | > 0.90 (sim is easy) |
| mAP@0.5:0.95 | > 0.75 |
| Object classification accuracy | > 0.95 |
| Position MAE (sim units) | < 0.02 |

### 5.4 Integration Path

```
RGB image → Detector → DetectedObject list → SceneInfo → scene_description → Parser
```

Three-way comparison:
1. **No scene**: Baseline (Stage 15) — parser has no scene awareness.
2. **Oracle scene**: Upper bound (Stage 17) — ground truth from sim.
3. **Predicted scene**: Stage 18 — detector output replaces oracle.

If the predicted-scene performance is close to oracle, the pipeline is ready
for real-world testing.

---

## 6. Implementation Order

### Phase 1: Smoke Dataset (S18-B)
- [ ] Collection script that saves RGB + oracle annotations.
- [ ] Validate schema and data integrity.
- [ ] Verify with mock env (no Robosuite) in tests.

### Phase 2: Full Dataset (S18-C)
- [ ] Collect 500+ images with scene variation.
- [ ] Add bbox projection from 3D poses (requires camera intrinsics).
- [ ] Add visibility heuristics.

### Phase 3: Detector Training (S18-D)
- [ ] Train YOLOv8n on smoke dataset.
- [ ] Evaluate against oracle.
- [ ] Export for pipeline integration.

### Phase 4: Scene-Aware Parser with Vision (S18-E)
- [ ] Replace oracle in E2E with predicted scene.
- [ ] Compare parsing accuracy: no-scene vs oracle vs predicted.

---

## 7. Limitations

1. **Sim-only**: Dataset is generated from Robosuite, not real images.
   Distribution shift to real-world is unknown.
2. **Simple scenes**: Only 2 blocks + table. Clutter, varied lighting,
   and diverse backgrounds not represented.
3. **Single camera**: Only agentview camera. Multi-view fusion not covered.
4. **No bbox yet**: Bounding box projection requires camera matrix extraction,
   which is deferred to Phase 2.
5. **No temporal data**: Each frame is independent. Video/temporal consistency
   not exploited.
6. **Assets are robosuite-specific**: Visual appearance (textures, lighting)
   matches robosuite default rendering, which may differ from real-world
   deployment.

---

## 8. Files

| File | Purpose |
|:-----|:--------|
| `docs/stage_18_rgb_perception_dataset.md` | This design document |
| `src/vlm_nsm/experiments/collect_rgb_perception_dataset.py` | Dataset collection script |
| `tests/test_rgb_perception_dataset_schema.py` | Schema validation tests |
| `generated/datasets/rgb_perception_v0_1/` | Collected dataset (after S18-B) |
