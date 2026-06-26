"""VLA Dataset — HDF5-based visual-action dataset for Diffusion Policy training.

Each episode is stored as a single HDF5 file with:
  /observations/agentview    uint8 (T, H, W, 3)
  /observations/frontview    uint8 (T, H, W, 3)
  /actions_4d                float32 (T, 4)   [dx, dy, dz, gripper]
  /proprio/eef_pos           float32 (T, 3)
  /proprio/gripper_open      float32 (T, 1)
  /symbolic/...              int8 (T,)        symbolic condition fields
  /success                   bool
  /failure_type              str

Design:
  - Lazy HDF5 loading via memory-mapped arrays (no per-frame PNG IO).
  - Phase-aware weighted sampling for contact-rich phases.
  - Returns observation history windows + action chunks for Diffusion training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Constants ───────────────────────────────────────────────────────────────

# Phase weights for phase-aware sampling.
# Higher weight → more samples from this phase.
# grasp_close and lift_verify are contact-critical and under-represented.
_DEFAULT_PHASE_WEIGHTS: dict[str, float] = {
    "reach_pick": 0.5,
    "descend_align": 0.8,
    "micro_align": 1.5,
    "grasp_close": 2.0,
    "lift_verify": 2.0,
    "transport": 0.5,
    "descend_place": 0.8,
    "micro_place": 1.5,
    "release_verify": 1.0,
    "retreat": 0.5,
}


# ── Episode Manifest ────────────────────────────────────────────────────────

class VLADatasetManifest:
    """Manages the list of episodes and their metadata."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.jsonl"
        self.episodes: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            # Auto-discover: scan for .hdf5 files
            for ep_path in sorted(self.root_dir.glob("episodes/ep_*.hdf5")):
                self.episodes.append({"episode_path": str(ep_path)})
        else:
            with open(self.manifest_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.episodes.append(json.loads(line))

    def success_mask(self) -> np.ndarray:
        return np.array([ep.get("success", True) for ep in self.episodes], dtype=bool)

    def filter_success_only(self) -> "VLADatasetManifest":
        manifest = VLADatasetManifest.__new__(VLADatasetManifest)
        manifest.root_dir = self.root_dir
        manifest.manifest_path = self.manifest_path
        manifest.episodes = [ep for ep in self.episodes if ep.get("success", True)]
        return manifest

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.episodes[idx]


# ── HDF5 Episode Reader ─────────────────────────────────────────────────────

class HDF5EpisodeReader:
    """Reads a single episode HDF5 file with memory-mapped arrays."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Episode file not found: {self.path}")
        self._h5: h5py.File | None = None

    def open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    @classmethod
    def from_handle(cls, h5_handle: h5py.File) -> "HDF5EpisodeReader":
        """Create a reader from an already-open HDF5 file handle."""
        reader = cls.__new__(cls)
        reader.path = Path(h5_handle.filename)
        reader._h5 = h5_handle
        return reader

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def num_steps(self) -> int:
        h5 = self.open()
        return h5["actions_4d"].shape[0]

    def read_agentview(self) -> np.ndarray:
        return self.open()["observations/agentview"][:]  # (T, H, W, 3)

    def read_frontview(self) -> np.ndarray:
        return self.open()["observations/frontview"][:]  # (T, H, W, 3)

    def read_actions(self) -> np.ndarray:
        return self.open()["actions_4d"][:]  # (T, 4)

    def read_eef_pos(self) -> np.ndarray:
        return self.open()["proprio/eef_pos"][:]  # (T, 3)

    def read_gripper(self) -> np.ndarray:
        return self.open()["proprio/gripper_open"][:]  # (T, 1)

    def read_phase_ids(self) -> np.ndarray:
        return self.open()["symbolic/phase_id"][:]  # (T,)

    def read_symbolic_field(self, field: str) -> np.ndarray | None:
        path = f"symbolic/{field}"
        h5 = self.open()
        if path in h5:
            return h5[path][:]
        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ── Torch Dataset ───────────────────────────────────────────────────────────

class VLADataset(Dataset):
    """Torch Dataset for VLA Diffusion Policy training.

    Each item yields:
      - agentview: (obs_history, 3, H, W)  — image sequence
      - frontview: (obs_history, 3, H, W)
      - proprio: (obs_history, 7)           — [eef_pos(3), gripper(1), zeros(3)]
      - symbolic_cond: (sym_dim,)            — symbolic condition at current step
      - action_chunk: (action_chunk_len, 4)  — target action sequence
      - phase_id: int                        — current phase label
      - index: int                           — global sample index

    Supports phase-aware weighted sampling via `sample_weights`.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        split: str = "train",
        obs_history_len: int = 2,
        action_chunk_len: int = 16,
        action_horizon: int = 8,
        stride: int = 4,
        image_size: int = 128,
        success_only: bool = True,
        phase_weights: dict[str, float] | None = None,
        transform: Callable | None = None,
        symbolic_dim: int = 42,
        cache_in_memory: bool = False,
        operator_filter: str | None = None,  # "Pick", "PlaceOn", or None (all)
        use_real_symbolic: bool = True,  # Phase 2.0: real symbolic encoding
        symbolic_shuffle_seed: int = 0,  # If >0, apply fixed permutation to real dims (B4.6)
        zero_phase: bool = False,  # If True, zero out phase_onehot portion (ablation)
    ):
        """Initialize VLA Dataset.

        Args:
            root_dir: Path to dataset root (contains episodes/ directory).
            split: "train" or "val".
            obs_history_len: Number of past observations to include.
            action_chunk_len: Length of action chunk for Diffusion training.
            action_horizon: Number of actions actually executed before re-predict.
            stride: Stride for sliding window sampling.
            image_size: Target image size (H, W assumed equal).
            success_only: If True, only use successful episodes.
            phase_weights: Per-phase sampling weights. Uses _DEFAULT_PHASE_WEIGHTS if None.
            transform: Optional image transform/augmentation.
            symbolic_dim: Dimension of the symbolic condition vector.
            cache_in_memory: If True, load all data into RAM (fast but memory-heavy).
            operator_filter: If set, only include windows from specific phases.
                "Pick" → phase 0,1,2 (reach_pick, grasp, lift)
                "PlaceOn" → phase 3,4,5 (reach_place, release, retreat)
                None → all phases
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.obs_history_len = obs_history_len
        self.action_chunk_len = action_chunk_len
        self.action_horizon = action_horizon
        self.stride = stride
        self.image_size = image_size
        self.transform = transform
        self.symbolic_dim = symbolic_dim
        self.cache_in_memory = cache_in_memory

        # Phase weights
        self.phase_weights = phase_weights or _DEFAULT_PHASE_WEIGHTS

        # Phase 3.0a: per-operator data split
        self.operator_filter = operator_filter
        self._pick_phases = {0, 1, 2}     # reach_pick, grasp, lift
        self._place_phases = {3, 4, 5}    # reach_place, release, retreat

        # Phase 2.0: real symbolic condition encoder
        self.use_real_symbolic = use_real_symbolic
        self.zero_phase = zero_phase
        # B4.6: fixed random permutation of real symbolic dims (first 31 of 42)
        self._sym_perm = None
        if symbolic_shuffle_seed > 0:
            rng = np.random.RandomState(symbolic_shuffle_seed)
            self._sym_perm = rng.permutation(31)  # permute 31 real dims, leave padding
        self._sym_encoder = None
        self._obj_id_to_name = {0: "table", 1: "cube1", 2: "cube2", 3: "cube3"}
        self._relation_id_to_name = {0: "on", 1: "table", 2: "none"}

        # Load manifest
        manifest = VLADatasetManifest(root_dir)
        if success_only:
            manifest = manifest.filter_success_only()

        # Train/val split by episode
        n_total = len(manifest)
        n_train = max(int(n_total * 0.8), 1) if n_total >= 2 else n_total
        indices = np.random.RandomState(42).permutation(n_total)
        if split == "train":
            self._episode_indices = indices[:n_train]
        else:
            self._episode_indices = indices[n_train:]

        self._episodes = [manifest[int(i)] for i in self._episode_indices]

        # Build sample index
        self._samples: list[dict[str, Any]] = []
        self._build_index()

        # File handle cache: pre-open all HDF5 files to avoid per-sample open/close
        self._file_cache: dict[str, h5py.File] = {}
        self._mem_cache: dict[str, dict[str, np.ndarray]] = {}
        if cache_in_memory:
            self._preload_memory()
        else:
            self._preopen_files()

        # Normalization stats (computed externally or lazily)
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self._try_load_stats()

    def _preopen_files(self) -> None:
        """Pre-open all HDF5 files and keep them in cache for fast access."""
        seen = set()
        for ep in self._episodes:
            path = ep.get("episode_path", "")
            if path and path not in seen and Path(path).exists():
                try:
                    self._file_cache[path] = h5py.File(path, "r")
                    seen.add(path)
                except Exception:
                    pass

    def _preload_memory(self) -> None:
        """Load all episode data into RAM for maximum training speed."""
        import time
        t0 = time.time()
        seen = set()
        for ep in self._episodes:
            path = ep.get("episode_path", "")
            if not path or path in seen or not Path(path).exists():
                continue
            seen.add(path)
            try:
                with h5py.File(path, "r") as f:
                    self._mem_cache[path] = {
                        "agentview": f["observations/agentview"][:],
                        "frontview": f["observations/frontview"][:],
                        "actions": f["actions_4d"][:],
                        "eef_pos": f["proprio/eef_pos"][:],
                        "gripper": f["proprio/gripper_open"][:],
                        "phase_id": f["symbolic/phase_id"][:],
                    }
            except Exception:
                pass
        elapsed = time.time() - t0
        total_mb = sum(
            sum(a.nbytes for a in d.values()) for d in self._mem_cache.values()
        ) / 1024 / 1024
        print(f"Memory cache loaded: {len(self._mem_cache)} episodes, "
              f"{total_mb:.0f} MB in {elapsed:.1f}s")

    def _get_reader(self, ep_path: str) -> "HDF5EpisodeReader":
        """Get a reader that reuses cached file handles."""
        if ep_path in self._file_cache:
            return HDF5EpisodeReader.from_handle(self._file_cache[ep_path])
        return HDF5EpisodeReader(ep_path)

    def close_files(self) -> None:
        """Close all cached file handles."""
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_cache.clear()

    # ── Index Building ──────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Scan all episodes and build a flat list of (episode_idx, start_step)."""
        for ep_idx, ep in enumerate(self._episodes):
            ep_path = ep.get("episode_path", "")
            if not ep_path or not Path(ep_path).exists():
                continue
            try:
                with HDF5EpisodeReader(ep_path) as reader:
                    T = reader.num_steps
                    # Pre-read symbolic fields for fast access
                    pick_ids = reader.read_symbolic_field("pick_object_id")
                    tgt_ids = reader.read_symbolic_field("target_object_id")
                    rel_ids = reader.read_symbolic_field("relation_id")
                    min_start = self.obs_history_len - 1
                    max_start = T - self.action_chunk_len - 1
                    for t in range(min_start, max_start + 1, self.stride):
                        phases = reader.read_phase_ids()
                        phase_id = int(phases[t]) if t < len(phases) else -1
                        # Phase 3.0a: per-operator filter
                        if self.operator_filter == "Pick" and phase_id not in self._pick_phases:
                            continue
                        if self.operator_filter == "PlaceOn" and phase_id not in self._place_phases:
                            continue
                        sample = {
                            "episode_idx": ep_idx,
                            "start_step": t,
                            "phase_id": phase_id,
                        }
                        # Phase 2.0: store symbolic IDs in sample index
                        if pick_ids is not None and t < len(pick_ids):
                            sample["pick_id"] = int(pick_ids[t])
                        if tgt_ids is not None and t < len(tgt_ids):
                            sample["target_id"] = int(tgt_ids[t])
                        if rel_ids is not None and t < len(rel_ids):
                            sample["rel_id"] = int(rel_ids[t])
                        self._samples.append(sample)
            except Exception:
                continue

    def _try_load_stats(self) -> None:
        stats_path = self.root_dir / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.action_mean = np.array(stats["action_mean"], dtype=np.float32)
            self.action_std = np.array(stats["action_std"], dtype=np.float32)

    # ── Sampling weights ────────────────────────────────────────────────

    @property
    def sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler (phase-aware)."""
        weights = np.ones(len(self._samples), dtype=np.float64)
        for i, sample in enumerate(self._samples):
            phase_id = sample.get("phase_id", -1)
            # Map phase_id to name via reverse lookup
            phase_name = self._phase_id_to_name(phase_id)
            weights[i] = self.phase_weights.get(phase_name, 1.0)
        return weights

    @staticmethod
    def _phase_id_to_name(phase_id: int) -> str:
        return VLADataset._PHASE_ID_TO_NAME.get(phase_id, "unknown")

    # ── Dataset Interface ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._samples[idx]
        ep = self._episodes[sample["episode_idx"]]
        ep_path = ep.get("episode_path", "")
        start = sample["start_step"]
        phase_id = sample["phase_id"]

        # Prefer memory cache for speed
        if ep_path in self._mem_cache:
            cache = self._mem_cache[ep_path]
            obs_start = start - self.obs_history_len + 1
            obs_end = start + 1
            agentview = cache["agentview"][obs_start:obs_end]
            frontview = cache["frontview"][obs_start:obs_end]
            eef = cache["eef_pos"][obs_start:obs_end]
            gripper = cache["gripper"][obs_start:obs_end]
            act_end = start + self.action_chunk_len
            actions = cache["actions"][start:act_end]
        else:
            reader = self._get_reader(ep_path) if ep_path else HDF5EpisodeReader(ep_path)
            try:
                obs_start = start - self.obs_history_len + 1
                obs_end = start + 1
                agentview = reader.read_agentview()[obs_start:obs_end]
                frontview = reader.read_frontview()[obs_start:obs_end]
                eef = reader.read_eef_pos()[obs_start:obs_end]
                gripper = reader.read_gripper()[obs_start:obs_end]
                act_end = start + self.action_chunk_len
                actions = reader.read_actions()[start:act_end]
            finally:
                if ep_path not in self._file_cache and hasattr(reader, "close"):
                    reader.close()

        # Convert to tensors
        agentview_t = self._preprocess_image(agentview[-1])  # Use latest frame
        frontview_t = self._preprocess_image(frontview[-1])
        proprio_t = self._build_proprio(eef[-1], gripper[-1])
        # Phase 2.0: read pre-stored symbolic IDs from sample index (fast)
        _pick_id = sample.get("pick_id", 0)
        _tgt_id = sample.get("target_id", 0)
        _rel_id = sample.get("rel_id", 2)
        sym_cond_t = self._build_symbolic_cond(phase_id, _pick_id, _tgt_id, _rel_id,
                                                  use_real_symbolic=self.use_real_symbolic)
        # B4.6 ablation: apply fixed permutation to real dims (preserve stats, destroy semantics)
        if self._sym_perm is not None:
            real = sym_cond_t[:31].clone()
            sym_cond_t[:31] = real[self._sym_perm]
        # Per-dimension ablation: zero out phase_onehot (indices 8-17)
        if self.zero_phase:
            sym_cond_t[8:18] = 0.0
        action_chunk_t = torch.from_numpy(actions.astype(np.float32))

        # Normalize actions if stats available
        if self.action_mean is not None and self.action_std is not None:
            action_chunk_t = (action_chunk_t - torch.from_numpy(self.action_mean)) / (
                torch.from_numpy(self.action_std) + 1e-8
            )

        return {
            "agentview": agentview_t,          # (3, H, W)
            "frontview": frontview_t,          # (3, H, W)
            "proprio": proprio_t,              # (7,)
            "symbolic_cond": sym_cond_t,        # (sym_dim,)
            "action_chunk": action_chunk_t,     # (chunk, 4)
            "phase_id": phase_id,
            "index": idx,
        }

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) → float32 (3, H, W) normalized to [0, 1]."""
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(img)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor

    @staticmethod
    def _build_proprio(eef: np.ndarray, gripper: np.ndarray) -> torch.Tensor:
        """Build 7-dim proprio: [eef_xyz(3), gripper(1), zeros(3)]."""
        proprio = np.zeros(7, dtype=np.float32)
        proprio[:3] = eef
        proprio[3] = float(gripper.ravel()[0]) if gripper.size > 0 else 0.0
        return torch.from_numpy(proprio)

    # Phase name mapping (consistent with SymbolicConditionEncoder._PHASE_NAMES)
    _PHASE_ID_TO_NAME: dict[int, str] = {
        0: "reach_pick", 1: "descend_align", 2: "micro_align",
        3: "grasp_close", 4: "lift_verify", 5: "transport",
        6: "descend_place", 7: "micro_place", 8: "release_verify",
        9: "retreat",
    }

    @staticmethod
    def _build_symbolic_cond(phase_id: int,
                             pick_id: int = 0,
                             target_id: int = 0,
                             rel_id: int = 2,
                             use_real_symbolic: bool = True) -> torch.Tensor:
        """Build symbolic condition vector.

        Phase 2.0: uses real symbolic fields (pick, target, relation, phase)
        from HDF5 data via SymbolicConditionEncoder when available.

        Args:
            phase_id: Phase index (0-9).
            pick_id: Pick object int ID (0=table, 1=cube1, 2=cube2, ...)
            target_id: Target int ID.
            rel_id: Relation int ID (0=on, 1=table, 2=none).
            use_real_symbolic: If False, return zero vector (B4 baseline training).
        """
        if not use_real_symbolic:
            return torch.zeros(42, dtype=torch.float32)

        _obj_map = {0: "table", 1: "cube1", 2: "cube2", 3: "cube3"}
        _rel_map = {0: "on", 1: "table", 2: "none"}

        pick_name = _obj_map.get(pick_id, "cube1")
        target_name = _obj_map.get(target_id, "cube2")
        rel_name = _rel_map.get(rel_id, "on")
        # Fix C2: Use consistent phase name mapping matching SymbolicConditionEncoder._PHASE_NAMES
        phase_name = VLADataset._PHASE_ID_TO_NAME.get(phase_id, "reach_pick")

        try:
            from vlm_nsm.neuro_symbolic.symbolic_condition import SymbolicConditionEncoder
            from vlm_nsm.skills.skill_types import SkillCommand
            encoder = SymbolicConditionEncoder(objects=["cube1", "cube2"])
            sc = SkillCommand(skill_name="pick_place", object_id=pick_name,
                              target_object_id=target_name, relation=rel_name)
            sv = encoder.encode(sc, phase_name=phase_name)
            raw = sv.as_vector()
            # Pad to model dim (42) if needed
            model_dim = 42
            padded = np.zeros(model_dim, dtype=np.float32)
            padded[:len(raw)] = raw
            return torch.from_numpy(padded)
        except (ImportError, KeyError, ValueError, AttributeError) as e:
            # Fallback to phase-only encoding (should not happen in normal operation)
            import warnings
            warnings.warn(f"SymbolicConditionEncoder failed: {e}. Using phase-only fallback.")
            vec = np.zeros(42, dtype=np.float32)
            if 0 <= phase_id < 10:
                phase_offset = 2 + 3 + 3
                vec[phase_offset + phase_id] = 1.0
            return torch.from_numpy(vec)


# ── Normalization Stats ─────────────────────────────────────────────────────

def compute_vla_normalization_stats(
    root_dir: str | Path,
    max_episodes: int = 500,
) -> dict[str, np.ndarray]:
    """Compute action mean/std across the dataset.

    Args:
        root_dir: Path to dataset root.
        max_episodes: Maximum number of episodes to scan for stats.

    Returns:
        Dict with "action_mean" and "action_std" (each shape (4,)).
    """
    manifest = VLADatasetManifest(root_dir).filter_success_only()
    episodes = manifest.episodes[:max_episodes]

    all_actions: list[np.ndarray] = []
    for ep in episodes:
        ep_path = ep.get("episode_path", "")
        if not ep_path or not Path(ep_path).exists():
            continue
        try:
            with HDF5EpisodeReader(ep_path) as reader:
                actions = reader.read_actions()
                all_actions.append(actions)
        except Exception:
            continue

    if not all_actions:
        raise RuntimeError(f"No valid episodes found in {root_dir}")

    cat = np.concatenate(all_actions, axis=0)
    return {
        "action_mean": cat.mean(axis=0).astype(np.float32),
        "action_std": cat.std(axis=0).astype(np.float32),
    }


def save_normalization_stats(root_dir: str | Path, max_episodes: int = 500) -> None:
    """Compute and save normalization stats to root_dir/stats.json."""
    root = Path(root_dir)
    stats = compute_vla_normalization_stats(root, max_episodes)
    stats_serializable = {
        "action_mean": stats["action_mean"].tolist(),
        "action_std": stats["action_std"].tolist(),
    }
    (root / "stats.json").write_text(
        json.dumps(stats_serializable, indent=2), encoding="utf-8"
    )


# ── Dataset Factory ─────────────────────────────────────────────────────────

def create_vla_datasets(
    root_dir: str | Path,
    obs_history_len: int = 2,
    action_chunk_len: int = 16,
    action_horizon: int = 8,
    stride: int = 4,
    image_size: int = 128,
    success_only: bool = True,
) -> tuple[VLADataset, VLADataset]:
    """Create train and val VLA datasets.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    common = dict(
        root_dir=root_dir,
        obs_history_len=obs_history_len,
        action_chunk_len=action_chunk_len,
        action_horizon=action_horizon,
        stride=stride,
        image_size=image_size,
        success_only=success_only,
    )
    train_ds = VLADataset(**common, split="train")
    val_ds = VLADataset(**common, split="val")
    return train_ds, val_ds
