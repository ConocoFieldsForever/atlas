"""Gym-style Python wrapper for the Atlas viewer's drone agent link.

The viewer (with the Agent Link enabled — camera tab checkbox or ``EFT_AGENT=7878``)
serves a lockstep drone simulation over newline-delimited JSON on 127.0.0.1. This
module wraps that protocol as:

- ``AgentLink``      — thin protocol client (any command, raw dicts).
- ``EftDroneEnv``    — a Gymnasium-compatible target-tracking environment
                       (falls back to a plain class when gymnasium isn't installed,
                       same ``reset``/``step`` signatures).

Task: keep the (simulated, in-game) walking target centered in the FPV camera at a
comfortable stand-off distance. Observation is a 16-float state vector (no pixels —
fast); the full raw obs dict is always available in ``info["raw"]``.

Smoke test (viewer running with a map open + agent link on):

    python tools/drone_env.py            # hover-ish random policy, prints obs rate

Protocol spec: docs/AGENT_LINK.md.
"""

from __future__ import annotations

import json
import math
import socket
import time
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_BASE = gym.Env
except ImportError:  # gymnasium optional — the plain class has the same API surface
    gym = None
    spaces = None
    _GYM_BASE = object


class AgentLink:
    """Line-oriented JSON client for the viewer's agent TCP server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7878, timeout: float = 30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rd = self.sock.makefile("r", encoding="utf-8")

    def call(self, cmd: str, **kw: Any) -> dict:
        req = {"cmd": cmd, **kw}
        self.sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        line = self._rd.readline()
        if not line:
            raise ConnectionError("agent link closed the connection")
        rep = json.loads(line)
        if not rep.get("ok", False):
            raise RuntimeError(f"agent link error for {cmd!r}: {rep.get('err')}")
        return rep

    def close(self) -> None:
        try:
            self.call("bye")
        except Exception:
            pass
        self.sock.close()


class EftDroneEnv(_GYM_BASE):
    """Target-tracking FPV drone task on a real extracted map.

    Action (Box, shape (4,), [-1, 1]): ``[roll, pitch, yaw, throttle]`` sticks;
    throttle is rescaled to [0, 1]. ``control="rates"`` (default, realistic acro)
    or ``"angle"`` (self-leveling, much easier to learn first).

    Observation (Box, shape (16,)):
        0:3   target position in the CAMERA frame / 10 m  (x right, y up, z back)
        3:6   target relative velocity in camera frame / 10 m/s
        6     distance / 10 m
        7:9   target NDC (u, v) — 0,0 when off-screen
        9     in_fov flag
        10:13 drone world velocity / 10 m/s
        13:16 drone body-up vector (attitude without a full quat)

    Reward: keep the target near image center (+1 max), stay near ``stand_off``
    metres, small action penalty; -25 and terminate on crash/out-of-bounds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7878,
        control: str = "rates",
        dt: float = 0.02,
        repeat: int = 1,
        episode_seconds: float = 60.0,
        stand_off: float = 8.0,
        target: Optional[dict] = None,
        drone: Optional[dict] = None,
        wind: tuple[float, float] = (0.0, 0.0),
        action_noise: float = 0.0,
        latency: int = 0,
        fov_deg: float = 90.0,
    ):
        self.link = AgentLink(host, port)
        self.control = control
        self.dt = dt
        self.repeat = max(1, int(repeat))
        self.max_steps = int(episode_seconds / (dt * self.repeat))
        self.stand_off = stand_off
        self.target_cfg = target or {"mode": "wander", "speed": 1.8, "radius": 30.0}
        self.drone_cfg = drone or {}
        self.wind = wind
        self.action_noise = action_noise
        self.latency = latency
        self.fov_deg = fov_deg
        self._steps = 0
        if spaces is not None:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

    # -- gym API -------------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        rep = self.link.call(
            "reset",
            seed=seed if seed is not None else int(time.time_ns() % (1 << 62)),
            dt=self.dt,
            control=self.control,
            target=self.target_cfg,
            drone=self.drone_cfg,
            wind=list(self.wind),
            action_noise=self.action_noise,
            latency=self.latency,
            fov_deg=self.fov_deg,
        )
        self._steps = 0
        return self._vec(rep), {"raw": rep}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(4)
        throttle = float((a[3] + 1.0) * 0.5)  # [-1,1] -> [0,1]
        rep = self.link.call(
            "step",
            action=[float(a[0]), float(a[1]), float(a[2]), throttle],
            repeat=self.repeat,
        )
        self._steps += 1
        obs = self._vec(rep)
        reward, terminated = self._reward(rep, a)
        truncated = self._steps >= self.max_steps
        return obs, reward, terminated, truncated, {"raw": rep}

    def close(self):
        self.link.close()

    # -- internals -----------------------------------------------------------------------

    @staticmethod
    def _vec(rep: dict) -> np.ndarray:
        d, t = rep["drone"], rep["target"]
        ndc = t["ndc"] or [0.0, 0.0]
        return np.array(
            [
                *(np.asarray(t["rel_cam"]) / 10.0),
                *(np.asarray(t["relvel_cam"]) / 10.0),
                t["dist"] / 10.0,
                ndc[0],
                ndc[1],
                1.0 if t["in_fov"] else 0.0,
                *(np.asarray(d["vel"]) / 10.0),
                *np.asarray(d["up"]),
            ],
            dtype=np.float32,
        )

    def _reward(self, rep: dict, a: np.ndarray) -> tuple[float, bool]:
        if rep["crashed"] or rep["oob"]:
            return -25.0, True
        t = rep["target"]
        r = 0.0
        if t["in_fov"]:
            u, v = t["ndc"]
            r += 1.0 - 0.5 * (abs(u) + abs(v))  # centered target
        else:
            r -= 0.5
        r -= 0.03 * abs(t["dist"] - self.stand_off)  # hold stand-off distance
        r -= 0.01 * float(np.square(a[:3]).sum())  # smooth flying
        if rep["collided"]:
            r -= 1.0
        return r, False


def _smoke():
    """Connect, reset, and fly ~10 s of gentle hover with a slow yaw sweep."""
    link = AgentLink()
    hello = link.call("hello")
    print(f"connected: map={hello['map']!r} grid_ready={hello['grid_ready']}")
    if not hello["grid_ready"]:
        print("grid not built yet — open a map and enable the agent link, then rerun")
        return
    hover = hello["params"]["hover_throttle"]
    rep = link.call("reset", seed=7, dt=0.02, control="angle",
                    target={"mode": "wander", "speed": 1.8, "radius": 25.0})
    print("reset ok; drone at", [round(x, 1) for x in rep["drone"]["pos"]])
    t0 = time.perf_counter()
    n = 500  # 10 s sim time
    for i in range(n):
        yaw = 0.25 * math.sin(i / 60.0)
        rep = link.call("step", action=[0.0, 0.05, yaw, hover], repeat=1)
    wall = time.perf_counter() - t0
    print(
        f"{n} steps ({n * 0.02:.0f} s sim) in {wall:.2f} s wall "
        f"({n / wall:.0f} steps/s, {n * 0.02 / wall:.1f}x realtime)"
    )
    t = rep["target"]
    print(
        f"final: dist={t['dist']:.1f} m  in_fov={t['in_fov']}  "
        f"collided={rep['collided']} crashed={rep['crashed']}"
    )
    link.close()


if __name__ == "__main__":
    _smoke()
