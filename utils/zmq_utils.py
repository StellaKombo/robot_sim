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


def pack_state(t: float, qpos, qvel, prev_action, *, policy_clock=None, policy_h=None, policy_c=None) -> bytes:
    """Raw float64 feedback; jogging appends its ungated clock and pre-query h/c.

    The legacy walking packet is unchanged. Jogging receivers require all 258
    recurrent channels and reject a legacy packet rather than filling with zeros.
    """
    fields = [[t], qpos, qvel, prev_action]
    if any(value is not None for value in (policy_clock, policy_h, policy_c)):
        if len(prev_action) != 29 or len(qpos) != 36 or len(qvel) != 35:
            raise ValueError('Recurrent feedback requires the 29-DoF robot layout')
        for name, value, size in (('clock', policy_clock, 2), ('h', policy_h, 128), ('c', policy_c, 128)):
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f'Missing or invalid jogging policy {name}')
            fields.append(value)
    return np.concatenate(fields).astype(np.float64).tobytes()


def unpack_state(buf: bytes, nq: int, nv: int, nu: int, *, with_policy_state=False):
    """Inverse of pack_state; optionally return the three recurrent fields too."""
    # reject anything that doesn't match this robot's sizes
    a = np.frombuffer(buf, dtype=np.float64)
    base = 1 + nq + nv + nu
    extra = 258 if with_policy_state and nu == 29 else 0
    if a.size != base + extra or not np.isfinite(a).all():
        raise ValueError(f"state message has {a.size} floats, expected {base + extra} finite values.")
    physical = (float(a[0]), a[1:1+nq].copy(), a[1+nq:1+nq+nv].copy(), a[1+nq+nv:base].copy())
    if not with_policy_state:
        return physical
    return physical + ((a[base:base+2].copy(), a[base+2:base+130].copy(), a[base+130:].copy())
                       if extra else (None, None, None))


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
