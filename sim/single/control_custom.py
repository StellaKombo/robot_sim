##
# Custom controller: KCF MPC -> body-frame [vx, vy, wz].
##

import argparse
from control_base import run
from policy.config import G1Config
from utils.kcf_controller import KCFCommand, get_default_mpc_config, resolve_mpc_settings
from utils.g1_kcf_state import g1_closed_loop_state


class CustomCommand(KCFCommand):
    """Select MPC settings; execution is shared with the existing trial runner."""

    def __init__(self, cmd_scale, *, settings=None, quiet=False, verbose=False,
                 feedback_is_synchronous=False,
                 reference_mode=None, prediction_model=None, relative_frame=False):
        settings = resolve_mpc_settings(
            get_default_mpc_config() if settings is None else settings,
            reference_mode=reference_mode,
            prediction_model=prediction_model,
            relative_frame=relative_frame,
        )
        super().__init__(cmd_scale, settings=settings, quiet=quiet, verbose=verbose,
                         feedback_is_synchronous=feedback_is_synchronous)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G1 custom controller: KCF L-BFGS-B MPC (pair with simulation.py)."
    )
    parser.add_argument(
        "--single-integrator",
        action="store_true",
        help="Use ideal body-twist single-integrator prediction instead of KCF",
    )
    parser.add_argument(
        "--relative-frame",
        action="store_true",
        help=(
            "Anchor the 84-D KCF and MPC reference to one fixed local frame "
            "per prediction horizon"
        ),
    )
    parser.add_argument(
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
        reference_mode="lissajous" if args.lissajous else None,
        prediction_model="single_integrator" if args.single_integrator else None,
        relative_frame=args.relative_frame,
    )
    try:
        run(source, cfg)
    finally:
        source.close()


if __name__ == "__main__":
    main()
