"""Launch V15 with logical batch 32 and exact gradient microbatching."""

import argparse

from .config import SPECS, activate, default_microbatch


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--block", required=True, choices=tuple(SPECS))
    known, rest = parser.parse_known_args(argv)
    activate(known.block)
    from ..v12 import train
    if "--stream-allocations" not in rest:
        rest.append("--stream-allocations")
    if "--image-microbatch" not in rest:
        rest += ["--image-microbatch", str(default_microbatch(known.block))]
    train.main(rest)


if __name__ == "__main__":
    main()
