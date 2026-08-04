"""Run one V20 profile with additive alpha=0.25 strict-fair training."""

import argparse

from .config import SPECS, activate


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", required=True, choices=tuple(SPECS))
    known, rest = parser.parse_known_args(argv)
    _, anchor = activate(known.profile)
    defaults = (
        ("--anchor", anchor.name),
        ("--fair-loss-alpha", "0.25"),
        ("--image-microbatch", "32"),
    )
    for flag, value in defaults:
        if flag not in rest:
            rest += [flag, value]
    for flag in ("--strict-fair-codec", "--stream-allocations"):
        if flag not in rest:
            rest.append(flag)
    from ..v12 import train
    train.main(rest)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
