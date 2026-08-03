"""Run blk20 R64/R96 with strict-fair shared-codec updates."""

from ..v12 import config as C


def main(argv=None):
    C.PLAN = "v16_strict_fair_shared_codec"
    C.output_dir = lambda anchor, run_id: (
        C.PHASE1 / "v16" / str(run_id) / anchor.name)
    from ..v12 import train
    args = list(argv or [])
    if "--strict-fair-codec" not in args:
        args.append("--strict-fair-codec")
    train.main(args)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
