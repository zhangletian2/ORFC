"""Run strict-fair equal, policy-weighted, or norm-matched training."""

from ..v12 import config as C


def main(argv=None):
    C.PLAN = "v17_strict_fair_weighting_control"
    C.output_dir = lambda anchor, run_id: (
        C.PHASE1 / "v17" / str(run_id) / anchor.name)
    from ..v12 import train
    args = list(argv or [])
    if "--strict-fair-codec" not in args:
        args.append("--strict-fair-codec")
    train.main(args)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
