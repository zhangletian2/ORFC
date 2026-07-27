"""
ORFC-v1 data utilities: deterministic train/val/test splitting,
seed management, and feature manifest I/O.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


# ================================================================
#  Deterministic train / validation / test split
# ================================================================

def make_split(
    train_files,
    test_files,
    n_train: int,
    n_val: int,
    split_seed: int = 42,
):
    """Create mutually exclusive train / val / test splits.

    * train and val are drawn from *train_files* without overlap.
    * test comes from *test_files* (entirely separate pool).

    Returns
    -------
    split : dict
        keys = 'train', 'val', 'test';
        values = list of Path objects.
    manifest : dict
        JSON-serialisable record of the split (basenames only).
    """
    rng = np.random.RandomState(split_seed)
    n_pool = len(train_files)
    if n_train + n_val > n_pool:
        raise ValueError(
            f"n_train({n_train}) + n_val({n_val}) > pool({n_pool})")
    perm = rng.permutation(n_pool)
    train_idx = sorted(perm[:n_train])
    val_idx = sorted(perm[n_train:n_train + n_val])

    split = {
        'train': [train_files[i] for i in train_idx],
        'val':   [train_files[i] for i in val_idx],
        'test':  list(test_files),
    }

    manifest = {
        'split_seed': split_seed,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': len(test_files),
        'train_basenames': [Path(f).stem for f in split['train']],
        'val_basenames':   [Path(f).stem for f in split['val']],
        'test_basenames':  [Path(f).stem for f in split['test']],
    }
    return split, manifest


def save_manifest(manifest, path):
    """Atomically save a deterministic split manifest."""
    import tempfile
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.json.tmp')
    os.close(fd)
    try:
        with open(tmp, 'w') as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


# ================================================================
#  Git info
# ================================================================

def git_info(repo_dir=None):
    """Return (commit_hash, is_dirty) or ('unknown', None)."""
    import subprocess
    cwd = repo_dir or os.getcwd()
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ['git', 'diff', '--quiet'], cwd=cwd,
            stderr=subprocess.DEVNULL) != 0
        return commit, dirty
    except Exception:
        return 'unknown', None


def git_status_porcelain(repo_dir=None):
    """Return `git status --porcelain` output for tracked + untracked (§11.8)."""
    import subprocess
    cwd = repo_dir or os.getcwd()
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
        return out if out else None
    except Exception:
        return None


# ================================================================
#  Seed bookkeeping
# ================================================================

def make_seeds(base_seed: int = 42):
    """Derive deterministic per-purpose seeds from a single base seed."""
    rng = np.random.RandomState(base_seed)
    return {
        'base': base_seed,
        'split': int(rng.randint(0, 2**31)),
        'minibatch': int(rng.randint(0, 2**31)),
        'group_sampling': int(rng.randint(0, 2**31)),
        'init': int(rng.randint(0, 2**31)),
    }


def build_result_skeleton(config, seeds, manifest):
    """Create the initial result dict with all required metadata."""
    commit, dirty = git_info()
    return {
        'schema_version': '1.0',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': commit,
        'git_dirty': dirty,
        'config': config,
        'seeds': seeds,
        'split_manifest': manifest,
        'history': [],
        'per_group_metrics': {},
        'heldout_metrics': {},
    }
