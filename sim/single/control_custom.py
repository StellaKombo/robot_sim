##
# Custom controller: KCF MPC -> body-frame [vx, vy, wz].
##

import argparse
from control_base import run
from policy.config import G1Config
from utils.kcf_controller import (
    KCFCommand,
    REFERENCE_MODES,
    get_default_mpc_config,
    resolve_mpc_settings,
)


class CustomCommand(KCFCommand):
    """Run the configured MPC and reference independently of simulation trials."""

    def __init__(
        self,
        cmd_scale,
        *,
        settings=None,
        quiet=False,
        verbose=False,
        feedback_is_synchronous=False,
        reference_mode=None,
        prediction_model=None
    ):
        settings = resolve_mpc_settings(
            get_default_mpc_config() if settings is None else settings,
            reference_mode=reference_mode,
            prediction_model=prediction_model,
        )
        super().__init__(
            cmd_scale,
            settings=settings,
            quiet=quiet,
            verbose=verbose,
            feedback_is_synchronous=feedback_is_synchronous,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G1 custom controller: KCF iCEM MPC (pair with simulation.py)."
    )
    parser.add_argument(
        "--unicycle",
        action="store_true",
        help="Use unicycle prediction and its configured gains (iCEM unchanged)",
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference-mode', choices=REFERENCE_MODES,
                           help='Override the reference selected in mpc_config.py')
    reference.add_argument(
        "--lissajous",
        action="store_true",
        help="Track the configured Lissajous pose/twist trajectory instead of a fixed target",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cfg = G1Config()
    source = CustomCommand(
        cfg.cmd_scale,
        reference_mode=args.reference_mode or ("lissajous" if args.lissajous else None),
        prediction_model="unicycle" if args.unicycle else None,
    )
    try:
        run(source, cfg)
    finally:
        source.close()


if __name__ == "__main__":
    main()
