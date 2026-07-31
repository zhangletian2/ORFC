"""Plan v10: algorithmic bridge, then three matched joint-learning arms.

v9 is a sealed conclusion.  Nothing under ``phase1/`` outside this package is
edited by v10 -- not one byte -- and neither is ``../orfc/soft_pq.py``.  Every
new line of v10 lives here and *reads* v9's modules (``engine``, ``frozen``,
``tail``, ``splits``, ``unseal``) as a library.

Module order is the protocol order:

    build_holdout -> verify(holdout) -> search/bridge -> profile -> train
    -> verify(arms) -> unseal

``config`` holds every frozen constant, ``qhard`` the one piece of genuinely
new numerics (a hard straight-through quantised forward, no temperature).
"""

from .. import REPO, ORFC  # noqa: F401  -- puts repo + ../orfc on sys.path
