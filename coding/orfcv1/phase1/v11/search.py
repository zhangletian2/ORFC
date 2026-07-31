"""v11's outer search: v10's, re-exported.

Plan v11 section 11 is explicit that there must not be a second searcher.  A2's
outer loop and N16's bridge must call the *same* function, because the bridge's
whole purpose is to say something about the optimiser A2 will actually use; two
implementations that agree today would be two implementations that could drift.

So this module re-exports and adds nothing.  v11's differences from v10 are in
what the searcher is handed -- a codec warm-started from ORFC, with side modes
kept fresh by the menu-coverage auxiliary -- not in how it searches.

The constants match by construction: ``S_OUTER_MAX``, ``EPS_ACCEPT_MULT`` and
``REPLAY_REL_TOL`` have the same values in v10's and v11's config, and
``assert_search_constants_match_v10`` fails the import if that ever stops being
true rather than letting the two rounds diverge silently.
"""

from ..v10.search import (            # noqa: F401  -- re-exported unchanged
    legal_swaps,
    greedy_swap_search,
    changed_groups,
)

from ..v10 import config as _v10
from . import config as C


def assert_search_constants_match_v10():
    drift = {name: (getattr(_v10, name), getattr(C, name))
             for name in ("S_OUTER_MAX", "EPS_ACCEPT_MULT", "REPLAY_REL_TOL",
                          "EVAL_IMAGE_BATCH", "EVAL_PAIR_BUDGET")
             if getattr(_v10, name) != getattr(C, name)}
    if drift:
        raise SystemExit(
            f"INVALID_EXPERIMENT: v11 re-exports v10's searcher but its "
            f"constants have drifted: {drift}.  The bridge would then be "
            f"characterising a different optimiser from the one A2 runs.")


assert_search_constants_match_v10()
