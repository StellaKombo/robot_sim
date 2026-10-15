##
#
#  Joystick controller: Xbox controller sticks -> body-frame [vx, vy, wz].
#
##

import numpy as np

from control_base import CommandSource, RobotState, import_pygame, main


class JoystickCommand(CommandSource):
    """Xbox controller via pygame, scaled by cmd_scale; zeros while unplugged, reconnects on plug-in."""

    def __init__(self, cmd_scale: np.ndarray):
        
        super().__init__(cmd_scale)
        
        # pygame and the Xbox button/axis mapping, loaded only here
        from utils.joystick_utils import pygame_to_joystick_state
        self.pygame = import_pygame()
        self.to_state = pygame_to_joystick_state

        # use the first joystick if one is plugged in
        self.joy = None
        self.pygame.init()
        self.pygame.joystick.init()
        if self.pygame.joystick.get_count() > 0:
            self._connect(0)
        else:
            print("No joystick detected; sending zero command until one is plugged in.")

    def _connect(self, index: int):
        self.joy = self.pygame.joystick.Joystick(index)
        self.joy.init()
        print(f"Joystick connected: [{self.joy.get_name()}].")
        print("  Left stick: forward/back -> vx, left/right -> vy.  Right stick: left/right -> yaw rate.")

    def get(self, state: RobotState) -> np.ndarray:
        
        pg = self.pygame

        # handle plug and unplug events
        for event in pg.event.get():
            if event.type == pg.JOYDEVICEADDED and self.joy is None:
                self._connect(event.device_index)
            elif event.type == pg.JOYDEVICEREMOVED and self.joy is not None \
                    and event.instance_id == self.joy.get_instance_id():
                print("Joystick disconnected; sending zero command.")
                self.joy = None
        
        # no joystick: stand still
        if self.joy is None:
            return np.zeros(3, dtype=np.float32)
        
        # read the sticks; a read error counts as an unplug
        try:
            s = self.to_state(self.joy)
        except pg.error:
            print("Joystick read error; sending zero command.")
            self.joy = None
            return np.zeros(3, dtype=np.float32)
        
        # left stick: vx, vy; right stick x: wz; scaled to the policy's range
        return np.array([s.LS_Y, s.LS_X, s.RS_X], dtype=np.float32) * self.cmd_scale

    def close(self):
        self.pygame.quit()


if __name__ == "__main__":
    main(JoystickCommand, "G1 joystick controller")
