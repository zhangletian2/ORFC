"""Run entropy-constrained joint training with the frozen V19 recipe."""

import argparse

from .config import FAIR_LOSS_ALPHA, RATE_LAMBDA, SPECS, activate


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--block", required=True, choices=tuple(SPECS))
    known, rest = parser.parse_known_args(argv)
    C = activate(known.block)
    defaults = (("--rate-lambda", str(RATE_LAMBDA)),
                ("--fair-loss-alpha", str(FAIR_LOSS_ALPHA)),
                ("--image-microbatch", "32"))
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
