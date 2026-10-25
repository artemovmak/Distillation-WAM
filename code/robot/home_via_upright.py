"""Bring the UR3 to the cube_in_box_v2 demo start joints from a foreign configuration (elbow on the other side, base
far away), one segment per call, instead of run_deploy's single direct moveJ (wide sweep at table height).
Path: 1) fold up to the upright pose keeping the current base and wrist 3 -> 2) turn base and wrist 3 to the demo
values up there (0.29 m sweep at 0.69 m) -> 3) lower into the demo q_mean.
Each call re-reads the arm, checks it is where that segment starts (segments 2, 3) and checks the whole joint-space
path with UR3 forward kinematics (elbow, wrists, flange, tool tip 0.19 m all above Z_MIN) before moving.
MOVES THE ROBOT: a human at the E-stop has explicitly allowed this segment.
    python home_via_upright.py plan          read-only: current q, the three targets, path heights
    python home_via_upright.py 1|2|3         run that segment (slow moveJ)
    python home_via_upright.py wrist3        turn only wrist 3 to the demo value (the joint is multi-turn: +186 deg looks
                                             like -174 deg but the cables are wrapped one turn more), other joints fixed
    python home_via_upright.py direct        one moveJ straight to q_mean, only if the elbow is already on the demo
                                             side and the tool tip stays within DIRECT_RMAX (after a hand-jog closer)
"""
import math
import os
import sys

import numpy as np
import rtde_receive

IP = os.environ.get("UR_IP", "192.168.1.10")  # robot controller address
SPEED, ACC = 0.35, 0.5  # rad/s, rad/s^2: a 200 deg base turn takes ~11 s
TOL_DEG = 4.0
Z_MIN = 0.15  # m above the base plane for every checked point along a segment
DIRECT_RMAX = 0.55  # m, tool tip horizontal radius bound for the direct mode (q_mean itself sits at 0.52)
Q_START = [math.degrees(x) for x in (0.2852, -1.5432, 1.5669, 0.2900, 1.1389, -3.0319)]  # cube_in_box_v2 q_mean
UPRIGHT = [None, -90.0, 0.0, -90.0, 90.0, None]  # base and wrist 3 filled in per segment

# UR3 (CB3) DH; checked on 2026-10-23: FK + 0.15 m tool along flange z reproduces getActualTCPPose
DH_D = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
DH_A = [0, -0.24365, -0.21325, 0, 0, 0]
DH_AL = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
TIP = 0.19


def fk_points(q_deg):
    T, pts = np.eye(4), []
    for i, th in enumerate(np.radians(q_deg)):
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(DH_AL[i]), math.sin(DH_AL[i])
        T = T @ np.array([[ct, -st * ca, st * sa, DH_A[i] * ct], [st, ct * ca, -ct * sa, DH_A[i] * st],
                          [0, sa, ca, DH_D[i]], [0, 0, 0, 1]])
        if i >= 1:
            pts.append(T[:3, 3].copy())
    pts.append(T[:3, 3] + T[:3, 2] * TIP)
    return pts  # elbow, wrist1, wrist2, wrist3, flange, tool tip


def path_report(a, b):
    a, b = np.array(a), np.array(b)
    zmin, rmax = 9.0, 0.0
    for s in np.linspace(0, 1, 201):
        P = fk_points(a + s * (b - a))
        zmin = min(zmin, min(p[2] for p in P))
        rmax = max(rmax, math.hypot(P[-1][0], P[-1][1]))
    return zmin, rmax


def targets(q):
    up1 = [q[0], *UPRIGHT[1:5], q[5]]
    up2 = [Q_START[0], *UPRIGHT[1:5], Q_START[5]]
    return {1: up1, 2: up2, 3: Q_START}


def read():
    r = rtde_receive.RTDEReceiveInterface(IP)
    q = [math.degrees(x) for x in r.getActualQ()]
    state = dict(robot_mode=r.getRobotMode(), safety_mode=r.getSafetyMode(),
                 protective_stop=r.isProtectiveStopped(), e_stop=r.isEmergencyStopped())
    r.disconnect()
    return q, state


def connect_control():
    """Same start sequence as the deploy stack's UR driver: optional per-host rig lease (refuses if our own deploy
    holds the arm), dashboard stop of any old script, 125 Hz (CB3 cap), up to 3 attempts (the first start after
    someone else's session often fails with 'Failed to start control script'), then the yaml TCP and payload."""
    import time

    import dashboard_client
    import rtde_control
    if os.environ.get("RIG_LEASE_PATH"):  # optional: the deploy stack's per-host lease, if available
        sys.path.insert(0, os.environ["RIG_LEASE_PATH"])
        from phantom.drivers.real import rig_lease
        connect_control.lease = rig_lease.acquire(IP)  # held until the process exits
    db = dashboard_client.DashboardClient(IP)
    db.connect()
    try:
        db.stop()
    finally:
        db.disconnect()
    time.sleep(0.5)
    last = None
    for attempt in range(3):
        try:
            c = rtde_control.RTDEControlInterface(IP, frequency=125.0)
            break
        except Exception as e:
            last, c = e, None
            print(f"RTDE control start failed ({attempt + 1}/3): {e}")
            time.sleep(2.0)
    if c is None:
        raise RuntimeError(f"RTDE control would not start: {last}")
    c.setTcp([0.0, 0.0, 0.18, 0.0, 0.0, 0.0])  # tool offset of the deployment config
    c.setPayload(1.2, [0.0, 0.0, 0.05])  # payload of the deployment config
    return c


def err(q, w, joints=range(6)):
    return max(abs(q[j] - w[j]) for j in joints)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "plan"
    q, state = read()
    print("q now", [round(x, 1) for x in q], state)
    T = targets(q)
    starts = {1: q, 2: T[1], 3: T[2]}
    for s in (1, 2, 3):
        z, r = path_report(starts[s], T[s])
        print(f"  seg {s} target {[round(x, 1) for x in T[s]]} | dq {[round(b - a, 1) for a, b in zip(starts[s], T[s])]} "
              f"| lowest point {z:.3f} m, tool tip max radius {r:.2f} m (if run from its planned start)")
    print(f"  direct moveJ to start (run_deploy): lowest {path_report(q, Q_START)[0]:.3f} m, "
          f"tip max radius {path_report(q, Q_START)[1]:.2f} m, max |dq| {err(q, Q_START):.1f} deg")
    if arg == "plan":
        return
    assert arg in ("1", "2", "3", "direct", "wrist3"), "argument must be plan, 1, 2, 3, direct or wrist3"
    seg = arg if arg in ("direct", "wrist3") else int(arg)
    assert state["robot_mode"] == 7 and state["safety_mode"] == 1 and not state["protective_stop"] and not state["e_stop"], \
        f"robot not in normal running state: {state}"
    if seg == "wrist3":
        target = [*q[:5], Q_START[5]]
    elif seg == "direct":
        assert q[2] > 20, f"elbow {q[2]:.1f} deg is not on the demo side (+{Q_START[2]:.0f}) - use segments 1-3"
        z, r = path_report(q, Q_START)
        assert r <= DIRECT_RMAX, f"direct path sweeps the tool tip to {r:.2f} m > {DIRECT_RMAX} m - use segments 1-3"
        target = Q_START
    elif seg == 2:  # must be upright; base and wrist 3 are what this segment changes
        assert err(q, T[1], joints=(1, 2, 3, 4)) <= TOL_DEG, f"not upright: {[round(x, 1) for x in q]} - abort"
        target = T[2]
    elif seg == 3:
        assert err(q, T[2]) <= TOL_DEG, f"not at upright-turned {T[2]}: max |dq| {err(q, T[2]):.1f} deg - abort"
        target = T[3]
    else:
        target = T[1]
    z, r = path_report(q, target)
    # from a low pose (episode ended near the table) the path may start below Z_MIN but must never go lower than it starts
    z_start = min(p[2] for p in fk_points(q))
    z_ok = min(Z_MIN, z_start - 0.005)
    assert z >= z_ok, f"segment {seg} path dips to {z:.3f} m < {z_ok:.3f} m (start {z_start:.3f} m) - abort"
    print(f"segment {seg} -> {[round(x, 1) for x in target]}, lowest point {z:.3f} m, moveJ speed {SPEED} rad/s acc {ACC}")
    c = connect_control()
    try:
        ok = c.moveJ([math.radians(x) for x in target], SPEED, ACC)
    finally:
        c.stopScript()
        c.disconnect()
    q, state = read()
    print("moveJ returned", ok, "| q now", [round(x, 1) for x in q], state, f"| max |dq| to target {err(q, target):.1f} deg")


if __name__ == "__main__":
    main()
