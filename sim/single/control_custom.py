##
#
#  Custom controller: your own logic -> body-frame [vx, vy, wz].
#
##

import numpy as np
from control_base import CommandSource, RobotState, main


class CustomCommand(CommandSource):
    """Your own command logic: return the body-frame [vx, vy, wz] for the given RobotState.

    Called every control_dt. The sim clips the result to cmd_scale.
    """

    def __init__(self, cmd_scale: np.ndarray):
        super().__init__(cmd_scale)
        # TODO: set up your controller here (gains, reference, planner, ...)

    def get(self, state: RobotState) -> np.ndarray:
        # TODO: replace with your controller; state.t is time since start, see RobotState
        return np.zeros(3, dtype=np.float32)


if __name__ == "__main__":
    main(CustomCommand, "G1 custom controller")
