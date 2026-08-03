"""Joint policy/codec training with additive strict-fair codec loss."""

from ..v12 import config as C


def main(argv=None):
    C.PLAN = "v19_additive_strict_fair_alpha"
    C.output_dir = lambda anchor, run_id: (
        C.PHASE1 / "v19" / str(run_id) / anchor.name)
    from ..v12 import train
    args = list(argv or [])
    for flag in ("--strict-fair-codec", "--stream-allocations"):
        if flag not in args:
            args.append(flag)
    train.main(args)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
