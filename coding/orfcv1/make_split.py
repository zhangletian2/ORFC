"""Freeze the new 4500/500 train/dev split of the 5,000-image train pool.

Index convention matches build_cache.py: positions into the ASCENDING sorted
list of basenames in features/<pool>/dinov2_vitl14/<block>/.

blk05 and blk20 are asserted to carry byte-identical basename sets, so a single
index serves both blocks -- the "same images across blocks" property is free.
"""
import numpy as np
from pathlib import Path

ROOT = Path('/data4/workspace/zlt/featcodec/features/train/dinov2_vitl14')
OUT = Path('/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/artifacts/dinov2_vitl14/split')
SEED = 20260730
N_DEV = 500

sets = {b: sorted(p.stem for p in (ROOT / b).iterdir() if p.suffix == '.npy')
        for b in ('blk05', 'blk20')}
names = sets['blk05']
assert sets['blk05'] == sets['blk20'], 'block basename sets differ'
n = len(names)
assert n == 5000, n

rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
dev = np.sort(perm[:N_DEV])
train = np.sort(perm[N_DEV:])
assert len(set(dev.tolist()) & set(train.tolist())) == 0
assert len(train) + len(dev) == n

OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / 'train_idx_n4500.npy', train.astype(np.int64))
np.save(OUT / 'dev_idx_n500.npy', dev.astype(np.int64))
(OUT / 'train_names_n4500.txt').write_text(
    '\n'.join(names[i] for i in train) + '\n')
(OUT / 'dev_names_n500.txt').write_text(
    '\n'.join(names[i] for i in dev) + '\n')
print(f'seed={SEED} pool={n} train={len(train)} dev={len(dev)}')
print('train first5', [names[i] for i in train[:5]])
print('dev   first5', [names[i] for i in dev[:5]])
print('wrote', OUT)
