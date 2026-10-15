##
#
#  Twist message and ZeroMQ socket helpers
#
##

import numpy as np
import zmq


# control_*.py -> simulation.py, body-frame [vx, vy, wz]; both run on this PC
HOST = "127.0.0.1"
CMD_PORT = 5555


############################################################################
# MESSAGE
############################################################################

def pack_twist(cmd) -> bytes:
    """Body-frame [vx, vy, wz] as 3 raw float64s."""
    return np.asarray(cmd, dtype=np.float64).reshape(3).tobytes()


def unpack_twist(buf: bytes) -> np.ndarray:
    """Inverse of `pack_twist`, as float32 like the policy's observation."""
    # reject anything that isn't exactly 3 floats
    a = np.frombuffer(buf, dtype=np.float64)
    if a.size != 3:
        raise ValueError(f"twist message has {a.size} floats, expected 3.")
    return a.astype(np.float32)


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


def connect_pub(ctx: zmq.Context, port: int) -> zmq.Socket:
    """Publisher connected to the subscriber bound on localhost at `port`; reconnects on its own."""
    # drop unsent messages on close, and connect to the sim's socket
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{HOST}:{port}")
    return sock
