##
#
#  Twist and state messages, and ZeroMQ socket helpers
#
##

import numpy as np
import zmq


# both run on this PC; the sim binds both ports and the controller connects
HOST = "127.0.0.1"
CMD_PORT = 5555     # control_*.py -> simulation.py: body-frame [vx, vy, wz] (+ reference point)
STATE_PORT = 5556   # simulation.py -> control_*.py: sim time, qpos, qvel, previous policy action


############################################################################
# MESSAGE
############################################################################

def pack_twist(cmd, ref=None) -> bytes:
    """Body-frame [vx, vy, wz] plus an optional world-frame reference point [x, y] for display.

    Sent as 5 raw float64s; a missing reference is sent as NaNs.
    """
    ref = [np.nan, np.nan] if ref is None else ref
    return np.concatenate([np.reshape(cmd, 3), np.reshape(ref, 2)]).astype(np.float64).tobytes()


def unpack_twist(buf: bytes):
    """Inverse of `pack_twist`: (cmd as float32 like the policy's observation, ref or None)."""
    # reject anything that isn't exactly 5 floats
    a = np.frombuffer(buf, dtype=np.float64)
    if a.size != 5:
        raise ValueError(f"twist message has {a.size} floats, expected 5.")
    ref = None if np.isnan(a[3:5]).any() else a[3:5].copy()
    return a[:3].astype(np.float32), ref


def pack_state(t: float, qpos, qvel, prev_action, *, obstacle_position=None, obstacle_time=None) -> bytes:
    """Optional world XYZ and measurement time, in the same clock as robot time."""
    values = [[t], qpos, qvel, prev_action]
    if obstacle_position is not None:
        center = np.asarray(obstacle_position, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Obstacle position must contain three finite world coordinates')
        values.append(center)
        if obstacle_time is not None:
            if not np.isfinite(obstacle_time):
                raise ValueError('Obstacle measurement time must be finite')
            values.append([obstacle_time])
    elif obstacle_time is not None:
        raise ValueError('Obstacle time requires an obstacle position')
    return np.concatenate(values).astype(np.float64).tobytes()


def unpack_state(buf: bytes, nq: int, nv: int, nu: int, *, include_obstacle=False,
                 include_obstacle_time=False):
    """Keep legacy four/five-value interfaces; timestamps opt into six values."""
    # reject anything that doesn't match this robot's sizes
    a = np.frombuffer(buf, dtype=np.float64)
    size = 1 + nq + nv + nu
    if a.size not in (size, size + 3, size + 4):
        raise ValueError(f"state message has {a.size} floats, expected {size}, {size + 3} or {size + 4}.")
    obstacle = a[size:size + 3].copy() if a.size > size else None
    stamp = float(a[-1] if a.size == size + 4 else a[0]) if obstacle is not None else None
    if obstacle is not None and not np.isfinite(obstacle).all():
        raise ValueError('Obstacle position must contain three finite world coordinates')
    if stamp is not None and not np.isfinite(stamp):
        raise ValueError('Obstacle measurement time must be finite')
    state = (float(a[0]), a[1:1 + nq].copy(), a[1 + nq:1 + nq + nv].copy(),
             a[1 + nq + nv:size].copy())
    if include_obstacle_time:
        return (*state, obstacle, stamp)
    return (*state, obstacle) if include_obstacle else state


############################################################################
# SOCKETS
############################################################################

def bind_sub(ctx: zmq.Context, port: int) -> zmq.Socket:
    """Subscriber bound to localhost only that keeps only the newest message."""
    # keep only the newest message, accept every topic, listen on localhost
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.CONFLATE, 1)   # must be set before bind
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.bind(f"tcp://{HOST}:{port}")
    return sock


def bind_pub(ctx: zmq.Context, port: int) -> zmq.Socket:
    """Publisher bound to localhost only."""
    # drop unsent messages on close
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{HOST}:{port}")
    return sock


def connect_sub(ctx: zmq.Context, port: int) -> zmq.Socket:
    """Subscriber connected to the publisher bound on localhost at `port`; keeps only the newest message."""
    # keep only the newest message, accept every topic, reconnect on its own
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.CONFLATE, 1)   # must be set before connect
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(f"tcp://{HOST}:{port}")
    return sock


def connect_pub(ctx: zmq.Context, port: int) -> zmq.Socket:
    """Publisher connected to the subscriber bound on localhost at `port`; reconnects on its own."""
    # drop unsent messages on close, and connect to the sim's socket
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{HOST}:{port}")
    return sock
