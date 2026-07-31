"""Phase-1 protocol package (plan v9).

Importing this package puts the repo root and the sibling ``orfc`` package on
``sys.path``, matching what every top-level script in the repo already does, so
``from opq import ...`` and ``from soft_pq import ...`` resolve the same way
here as they do from the repo root.

Module order is the protocol order: ``splits`` -> ``build`` -> ``verify`` ->
``cal_sweep`` -> ``unseal``.  ``config`` holds every frozen constant, ``engine``
the hard-quantisation evaluator, ``kmeans`` the initialisation numerics,
``frozen`` the checked loader and ``tail`` the ViT tail.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORFC = REPO.parent / "orfc"
for path in (str(REPO), str(ORFC)):
    if path not in sys.path:
        sys.path.insert(0, path)
