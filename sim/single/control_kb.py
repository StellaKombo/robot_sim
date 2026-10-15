##
#
#  Keyboard controller: arrow keys in a small pygame window -> body-frame [vx, vy, wz].
#
##

import numpy as np

from control_base import CommandSource, RobotState, import_pygame, main


class KeyboardCommand(CommandSource):
    """Arrow keys in a small pygame window; a held key gives speed x cmd_scale on its axis.

    Up/Down: forward/back (vx), Left/Right: turn (wz), Shift + Left/Right: step sideways (vy).
    +/- change speed in steps of 0.1x, between 0.1x and 1.0x, starting at 1.0x. Opposite keys
    cancel. The window must have focus; the sim's viewer takes it when it opens, so the window
    turns red until it is clicked again.
    """

    def __init__(self, cmd_scale: np.ndarray):
        
        super().__init__(cmd_scale)
        
        self.pygame = import_pygame()
        self.speed_tenths = 10   # speed multiplier in tenths, so stepping never drifts

        # the window that receives keys and shows the command
        self.pygame.init()
        self.screen = self.pygame.display.set_mode((420, 180))
        self.pygame.display.set_caption("robot_sim keyboard")
        self.font = self.pygame.font.Font(None, 28)
        
        # focus is unknown until the first get()
        self.focused = None
        print("Keyboard control: focus the 'robot_sim keyboard' window.")
        print("  Up/Down: forward/back (vx), Left/Right: turn (wz), Shift + Left/Right: step sideways (vy).")
        print("  +/-: speed up/down by 0.1x (0.1x to 1.0x), starting at 1.0x.")

    def get(self, state: RobotState) -> np.ndarray:
        
        pg = self.pygame
        
        # window close, and +/- presses (one speed step per press)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                raise KeyboardInterrupt   # closing the window stops the controller like Ctrl+C
            if event.type == pg.KEYDOWN:
                if event.key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                    self._set_speed(self.speed_tenths + 1)
                elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    self._set_speed(self.speed_tenths - 1)
        
        # report each focus change once
        focused = bool(pg.key.get_focused())
        if focused != self.focused:
            print("Keyboard window focused; keys are live." if focused else
                  "Keyboard window lost focus; click it to drive. Sending zero command.")
            self.focused = focused

        # SDL clears held keys on focus loss, so an unfocused window already reads as zero
        k = pg.key.get_pressed()
        
        # Left/Right turn, or step sideways while Shift is held
        shift = k[pg.K_LSHIFT] or k[pg.K_RSHIFT]
        left_right = k[pg.K_LEFT] - k[pg.K_RIGHT]
        axes = np.array([k[pg.K_UP] - k[pg.K_DOWN], left_right if shift else 0,
                         0 if shift else left_right], dtype=np.float32)
        
        # scale by speed and the policy's range; zero when unfocused
        speed = self.speed_tenths / 10.0
        cmd = axes * speed * self.cmd_scale if focused else np.zeros(3, dtype=np.float32)
        self._draw(cmd, focused)
        return cmd

    def _set_speed(self, tenths: int):
        # clamp to 0.1x-1.0x and announce changes
        tenths = min(max(tenths, 1), 10)
        if tenths != self.speed_tenths:
            self.speed_tenths = tenths
            print(f"Speed {tenths / 10:.1f}x")

    def _draw(self, cmd: np.ndarray, focused: bool):
        # red background and a hint when the window lacks focus
        self.screen.fill((30, 30, 30) if focused else (140, 25, 25))
        lines = ["Up/Down: vx  Left/Right: wz  +Shift: vy" if focused else "CLICK HERE TO DRIVE",
                 f"speed = {self.speed_tenths / 10:.1f}x   (+/- to change)",
                 f"vx = {cmd[0]:+.2f} m/s",
                 f"vy = {cmd[1]:+.2f} m/s",
                 f"wz = {cmd[2]:+.2f} rad/s"]
        
        # one text line every 32 px
        for i, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True, (230, 230, 230)), (16, 14 + 32 * i))
        self.pygame.display.flip()

    def close(self):
        self.pygame.quit()


if __name__ == "__main__":
    main(KeyboardCommand, "G1 keyboard controller")
