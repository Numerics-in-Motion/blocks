# -*- coding: utf-8 -*-
"""Under this pulse, the bigger block stays standing.

    python rocking_block.py                # the registered run -> $ROCK_DATA (default ../data)

THE MODEL (Housner 1963)

A planar, homogeneous, rigid rectangular block on rigid horizontal ground. It rocks
about one bottom corner or the other; it does not slide, bounce, flex or wobble, and
the ground moves only horizontally. With half-diagonal R and slenderness
alpha = atan(b/h), and p = sqrt(3g / 4R),

    theta'' = -p^2 [ sin(alpha sgn(theta) - theta) + (ug''/g) cos(alpha sgn(theta) - theta) ].

It lifts off when |ug''| exceeds g tan(alpha). When it comes back to theta = 0 it
lands on the other corner: angular momentum about that corner is conserved, so the
angular velocity keeps its direction and its size is multiplied by

    c_v = 1 - (3/2) sin^2(alpha)            (kinetic energy by c_v^2).

THE PULSE

One cycle of ground acceleration a_p sin(2 pi t / T_p), 0 <= t <= T_p, then nothing;
the amplitude is A = a_p / (g tan alpha), so A = 1 is the lift-off threshold.

WHAT IS COMPUTED

Dimensionless x = theta/alpha, tau = p t, v = dx/dtau. The forced phase is integrated
event by event (impacts x = 0, overturning |x| = 1). After the pulse no work is done
on the block, so its fate is decided exactly by its energy
E/(gR) = alpha^2 v^2 / 2 + cos(alpha (1 - |x|)) - cos(alpha) against the barrier
1 - cos(alpha): moving outward it overturns if E >= E_b; moving inward it first
lands, keeping c_v^2 E. A block that reaches |x| = 1 during the pulse is counted as
overturned.

Each solve runs in three implementations (registered in the design review, locked
before the production run):
  P  production   DOP853, rtol 1e-10, atol 1e-12, max step min(0.002, tau_p/1000)
  T  tight        DOP853, rtol 1e-12, atol 1e-14, half the production step
  E  event audit  production integration; every event bracketed independently from
                  dense output (bracket <= 1e-10, residual <= 1e-12), and again (E2)
                  with half the event-search spacing
and every reported number carries u = |q_P - q_T| + |q_T - q_E| + |q_h - q_h/2| +
64 eps max(1, |q_T|).

Contact forces (per unit mass and g, impacts excluded):
  H = f + (3/4)[ s alpha^2 v^2 sin(phi) + alpha a cos(phi) ]
  N = 1 + (3/4)[ -alpha^2 v^2 cos(phi) + s alpha a sin(phi) ],   phi = alpha(1 - |x|).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import DOP853, solve_ivp
from scipy.optimize import brentq

G = 9.80665
ALPHA = 0.2
TP = 0.5
A_HEAD = 2.0
SIZES = (0.25, 0.5, 1.0, 2.0, 4.0)
HEAD_PAIR = (0.5, 2.0)
EXPECTED = ("O", "O", "S", "S", "S")
A_NEIGHBOURS = (1.9, 2.1)
NEIGHBOUR_CLEARANCE = 0.10
CV_FACTORS = (0.80, 0.90, 1.00)
SCAN_HUNDREDTHS = (100, 600)          # even hundredths: baseline 0.02; odd: audit points
BISECT_WIDTH = 1e-5
RECT_ALPHA, RECT_R, RECT_PTD, RECT_TOL = 0.02, 1.0, (0.5, 1.0, 2.0), 1e-3
SIMILARITY = (((0.5, 0.25), (1.0, 1.0)), ((0.5, 0.5), (1.0, 2.0)), ((0.5, 1.0), (1.0, 4.0)))
SIMILARITY_TOL = 1e-8
N_MIN, MU_MAX = 0.80, 0.30
IMPACT_RATIO_TOL = 1e-9
IMPACT_RESIDUAL_TOL = 1e-10
EVENT_BRACKET, ROOT_RESIDUAL = 1e-10, 1e-12
EPS = float(np.finfo(float).eps)
V_REST = 1e-12            # a landing slower than this leaves the block resting on the ground
MAX_IMPACTS = 100000
POST_E_FLOOR = 1e-6       # post-pulse forces are sampled until E/E_b falls below this
IMPLS = ("P", "T", "E", "E2")
IMPL = dict(P=dict(rtol=1e-10, atol=1e-12, step=1.0, dense_events=None),
            T=dict(rtol=1e-12, atol=1e-14, step=0.5, dense_events=None),
            E=dict(rtol=1e-10, atol=1e-12, step=1.0, dense_events=1.0),
            E2=dict(rtol=1e-10, atol=1e-12, step=1.0, dense_events=0.5))
OUT = os.environ.get("ROCK_DATA") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def cv_housner(alpha):
    return 1.0 - 1.5 * math.sin(alpha) ** 2


class Case:
    """One registered solve: block (alpha, R), pulse, amplitude, restitution factor."""

    def __init__(self, R, A, alpha=ALPHA, Tp=TP, pulse="sine", ptd=None, cv_factor=1.0):
        self.R, self.A, self.alpha, self.Tp, self.pulse, self.ptd = R, A, alpha, Tp, pulse, ptd
        self.cv_factor = cv_factor
        self.p = math.sqrt(3.0 * G / (4.0 * R))
        self.cv = cv_factor * cv_housner(alpha)
        self.thr = math.tan(alpha)
        if pulse == "sine":
            self.Om = (2.0 * math.pi / Tp) / self.p
            self.tau_p = self.p * Tp
            self.amp = A * math.tan(alpha)
        else:                                   # rectangular, A = a_p / (g alpha)
            self.Om = None
            self.tau_p = ptd
            self.amp = A * alpha
        self.hmax = min(0.002, self.tau_p / 1000.0)

    def key(self):
        return dict(R=self.R, A=self.A, alpha=self.alpha, Tp=self.Tp, pulse=self.pulse, ptd=self.ptd,
                    cv_factor=self.cv_factor)

    def f(self, tau):
        if tau < 0.0 or tau > self.tau_p:
            return 0.0
        if self.pulse == "sine":
            return self.amp * math.sin(self.Om * tau)
        return self.amp

    def accel(self, s, x, v, f):
        phi = self.alpha * (1.0 - s * x)
        return -(s * math.sin(phi) + f * math.cos(phi)) / self.alpha

    def forces(self, s, x, v, f):
        a = self.accel(s, x, v, f)
        phi = self.alpha * (1.0 - s * x)
        al = self.alpha
        H = f + 0.75 * (s * al * al * v * v * math.sin(phi) + al * a * math.cos(phi))
        N = 1.0 + 0.75 * (-al * al * v * v * math.cos(phi) + s * al * a * math.sin(phi))
        return H, N

    def energy(self, x, v):
        """E / E_b."""
        return (0.5 * self.alpha ** 2 * v * v + math.cos(self.alpha * (1.0 - abs(x))) - math.cos(self.alpha)) \
            / (1.0 - math.cos(self.alpha))

    def uplift_analytic(self):
        if self.pulse != "sine" or self.A <= 1.0:
            return None
        return math.asin(1.0 / self.A) / self.Om


def next_uplift(c, tau0, n=4000):
    """First tau >= tau0 inside the pulse at which |f| exceeds g tan(alpha), taken on the
    side where it does, and the corner the ground acceleration tips the block onto."""
    if tau0 >= c.tau_p:
        return None
    if abs(c.f(tau0)) > c.thr:
        return tau0, (-1 if c.f(tau0) > 0 else 1)
    grid = np.linspace(tau0, c.tau_p, n + 1)
    g = lambda t: abs(c.f(t)) - c.thr
    prev = g(grid[0])
    for k in range(1, n + 1):
        cur = g(grid[k])
        if cur > 0.0:
            r = brentq(g, grid[k - 1], grid[k], xtol=1e-15, rtol=4 * EPS, maxiter=500)
            while g(r) <= 0.0:
                r = float(np.nextafter(r, np.inf))
            return r, (-1 if c.f(r) > 0 else 1)
        prev = cur
    return None


def _rhs(c, s):
    def rhs(t, y):
        return [y[1], c.accel(s, y[0], y[1], c.f(t))]
    return rhs


def branch(c, s, tau0, y0, tau_end, impl):
    """Integrate one rocking branch from tau0; return (kind, tau, y, sol, residual, bracket)
    with kind 'impact', 'over' or 'end'."""
    cfg = IMPL[impl]
    hmax = c.hmax * cfg["step"]
    atol = (cfg["atol"], cfg["atol"])
    rhs = _rhs(c, s)
    if cfg["dense_events"] is None:
        def hit(t, y):
            return s * y[0]
        hit.terminal, hit.direction = True, -1

        def over(t, y):
            return abs(y[0]) - 1.0
        over.terminal, over.direction = True, 1
        sol = solve_ivp(rhs, (tau0, tau_end), y0, method="DOP853", rtol=cfg["rtol"], atol=atol,
                        max_step=hmax, events=(hit, over), dense_output=True)
        te = [(float(sol.t_events[k][0]), k) for k in (0, 1) if sol.t_events[k].size]
        if not te:
            return "end", tau_end, [float(sol.y[0][-1]), float(sol.y[1][-1])], sol, 0.0, 0.0
        t, k = min(te)
        y = [float(v) for v in (sol.y_events[k][0])]
        res = abs(y[0]) if k == 0 else abs(abs(y[0]) - 1.0)
        return ("impact" if k == 0 else "over"), t, y, sol, res, 0.0
    # the same DOP853 stepping as production, but no solver events: each step's dense
    # output is searched on a fixed grid (spacing = step fraction of the production
    # maximum step) and a sign change is bracketed by brentq
    solver = DOP853(rhs, tau0, y0, tau_end, rtol=cfg["rtol"], atol=atol, max_step=hmax)
    spacing = hmax * cfg["dense_events"]
    pieces = Pieces()
    k_next = 1
    while solver.status == "running":
        t_old = solver.t
        solver.step()
        dense = solver.dense_output()
        pieces.add(t_old, solver.t, dense)
        # grid points tau0 + k spacing inside (t_old, t]; the step end is always checked
        pts = []
        while tau0 + k_next * spacing <= solver.t:
            pts.append(tau0 + k_next * spacing)
            k_next += 1
        if not pts or pts[-1] < solver.t:
            pts.append(solver.t)
        prev_t = t_old
        prev_x = float(dense(t_old)[0])
        for tg in pts:
            xg = float(dense(tg)[0])
            imp, ov = s * xg < 0.0, abs(xg) >= 1.0
            if imp or ov:
                best = None
                for kind, g in ((("impact", lambda t: s * float(dense(t)[0])),) if imp else ()) +                         ((("over", lambda t: abs(float(dense(t)[0])) - 1.0),) if ov else ()):
                    if kind == "impact" and s * prev_x <= 0.0:
                        t = prev_t
                    else:
                        t = brentq(g, prev_t, tg, xtol=1e-13, rtol=4 * EPS, maxiter=500)
                    if best is None or t < best[1]:
                        best = (kind, t)
                kind, t = best
                y = [float(v) for v in dense(t)]
                res = abs(y[0]) if kind == "impact" else abs(abs(y[0]) - 1.0)
                pieces.cut(t)
                return kind, t, y, pieces, res, 1e-13 + 4 * EPS * abs(t)
            prev_t, prev_x = tg, xg
    y = [float(v) for v in solver.y]
    return "end", solver.t, y, pieces, 0.0, 0.0


class Pieces:
    """Dense output of a sequence of steps, callable like OdeSolution.sol."""

    def __init__(self):
        self.t0, self.t1, self.f = [], [], []

    def add(self, a, b, f):
        self.t0.append(a)
        self.t1.append(b)
        self.f.append(f)

    def cut(self, t):
        self.t1[-1] = t

    @property
    def sol(self):
        return self

    def __call__(self, t):
        scalar = np.ndim(t) == 0
        ts = np.atleast_1d(np.asarray(t, float))
        out = np.empty((2, ts.size))
        idx = np.searchsorted(np.asarray(self.t1), ts, side="left")
        idx = np.clip(idx, 0, len(self.f) - 1)
        for i in np.unique(idx):
            m = idx == i
            out[:, m] = np.asarray(self.f[i](ts[m])).reshape(2, -1)
        return out[:, 0] if scalar else out


def sample_segment(c, s, sol, t0, t1, h, acc):
    """Extremes of |x|, N and |H|/N on [t0, t1] sampled at spacing h."""
    if t1 <= t0:
        return
    n = max(1, int(math.ceil((t1 - t0) / h)))
    ts = np.linspace(t0, t1, n + 1)
    Y = sol.sol(ts)
    for t, x, v in zip(ts, Y[0], Y[1]):
        H, N = c.forces(s, x, v, c.f(t))
        acc["xmax"] = max(acc["xmax"], abs(x))
        acc["Nmin"] = min(acc["Nmin"], N)
        acc["mu"] = max(acc["mu"], abs(H) / N if N > 0 else math.inf)


def rest_segment(c, t0, t1, h, acc):
    if t1 <= t0:
        return
    n = max(1, int(math.ceil((t1 - t0) / h)))
    for t in np.linspace(t0, t1, n + 1):
        acc["mu"] = max(acc["mu"], abs(c.f(t)))


def simulate(c, impl, keep=False, post=False):
    """The forced phase event by event, then the exact post-pulse decision."""
    tau, x, v, s = 0.0, 0.0, 0.0, 0
    impacts, rests, uplifts = [], 0, []
    acc = {h: dict(xmax=0.0, Nmin=1.0, mu=0.0) for h in ("h", "h2")}
    segs = []
    outcome = how = None
    tau_over = None
    over_residual = None
    e_test = None
    while True:
        if s == 0:
            u = next_uplift(c, tau)
            if u is None:
                for tag, hh in (("h", c.hmax), ("h2", c.hmax / 2)):
                    rest_segment(c, tau, c.tau_p, hh, acc[tag])
                outcome, how = "S", "rest"
                e_end = 0.0
                x_end, v_end, s_end = 0.0, 0.0, 0
                break
            for tag, hh in (("h", c.hmax), ("h2", c.hmax / 2)):
                rest_segment(c, tau, u[0], hh, acc[tag])
            tau, s = u
            x, v = 0.0, 0.0
            uplifts.append(tau)
        kind, t, y, sol, res, br = branch(c, s, tau, [x, v], c.tau_p, impl)
        for tag, hh in (("h", c.hmax), ("h2", c.hmax / 2)):
            sample_segment(c, s, sol, tau, t, hh, acc[tag])
        if keep:
            segs.append(dict(s=s, t0=tau, t1=t, sol=sol))
        if kind == "over":
            outcome, how, tau_over = "O", "event", t
            over_residual = res
            x_end, v_end, s_end, e_end = y[0], y[1], s, c.energy(y[0], y[1])
            break
        if kind == "impact":
            v_minus = y[1]
            v_plus = c.cv * v_minus
            e_minus = 0.5 * c.alpha ** 2 * v_minus ** 2
            e_plus = 0.5 * c.alpha ** 2 * v_plus ** 2
            impacts.append(dict(tau=t, residual=res, bracket=br, v_minus=v_minus, v_plus=v_plus,
                                ratio_err=abs(abs(v_plus / v_minus) - c.cv) if v_minus else 0.0,
                                energy_err=abs(e_plus / e_minus - c.cv ** 2) if e_minus else 0.0))
            if len(impacts) > MAX_IMPACTS:
                outcome, how = "U", "impact limit"
                x_end = v_end = e_end = 0.0
                s_end = 0
                break
            tau = t
            if abs(v_plus) < V_REST:
                s, x, v = 0, 0.0, 0.0
                rests += 1
                continue
            s = 1 if v_plus > 0 else -1
            x, v = 0.0, v_plus
            continue
        # end of the pulse
        x_end, v_end, s_end = y[0], y[1], s
        e_end = c.energy(x_end, v_end)
        outward = s * v_end > 0.0
        e_test = e_end if outward else c.cv ** 2 * e_end
        outcome = "O" if e_test >= 1.0 else "S"
        how = "energy outward" if outward else "energy after landing"
        break
    out = dict(case=c.key(), impl=impl, outcome=outcome, how=how, tau_over=tau_over,
               impacts=len(impacts), rests=rests, uplifts=uplifts,
               x_end=x_end, v_end=v_end, s_end=s_end, e_end=e_end, e_test=e_test, over_residual=over_residual,
               xmax_h=acc["h"]["xmax"], xmax_h2=acc["h2"]["xmax"],
               Nmin_h=acc["h"]["Nmin"], Nmin_h2=acc["h2"]["Nmin"],
               mu_h=acc["h"]["mu"], mu_h2=acc["h2"]["mu"],
               impact_residual=max([i["residual"] for i in impacts], default=0.0),
               impact_bracket=max([i["bracket"] for i in impacts], default=0.0),
               ratio_err=max([i["ratio_err"] for i in impacts], default=0.0),
               energy_err=max([i["energy_err"] for i in impacts], default=0.0),
               chronological=all(b["tau"] > a["tau"] for a, b in zip(impacts, impacts[1:])),
               impact_taus=[i["tau"] for i in impacts])
    ua = c.uplift_analytic()
    out["uplift_err"] = abs(uplifts[0] - ua) if (uplifts and ua is not None) else None
    if keep:
        out["_segs"] = segs
    if how in ("energy outward", "energy after landing"):
        out.update(post_contact_exact(c, x_end, v_end, s_end))
    elif how == "rest":
        out.update(dict(post_x_Nmin_h=1.0, post_x_Nmin_h2=1.0, post_x_mu_h=0.0, post_x_mu_h2=0.0, post_x_segments=0))
    if post and how in ("energy outward", "energy after landing"):
        out.update(post_pulse(c, x_end, v_end, s_end, impl))
    elif post and how == "rest":
        out.update(dict(post_Nmin_h=1.0, post_Nmin_h2=1.0, post_mu_h=0.0, post_mu_h2=0.0, post_impacts=0))
    return out


def free_rock(c, x, v, s, tau0, impl, e_floor=POST_E_FLOOR, t_end=None, keep=False):
    """Unforced rocking after the pulse, impact by impact, until E/E_b < e_floor (or
    t_end, or overturning |x| = 1)."""
    tau = tau0
    segs, n_imp = [], 0
    audit = dict(residual=0.0, ratio_err=0.0, energy_err=0.0, chronological=True, over_residual=None)
    last_t = -math.inf
    acc = {h: dict(xmax=0.0, Nmin=1.0, mu=0.0) for h in ("h", "h2")}
    hmax = c.hmax
    fate = "rest"
    while True:
        if s == 0 or c.energy(x, v) < e_floor:
            break
        # a quarter-period bound: integrate in windows until an event
        horizon = tau + 50.0 if t_end is None else t_end
        kind, t, y, sol, res, br = branch(c, s, tau, [x, v], horizon, impl)
        for tag, hh in (("h", hmax), ("h2", hmax / 2)):
            sample_segment(c, s, sol, tau, t, hh, acc[tag])
        if keep:
            segs.append(dict(s=s, t0=tau, t1=t, sol=sol))
        if kind == "over":
            fate = "over"
            tau = t
            audit["over_residual"] = res
            break
        if kind == "end":
            tau, x, v = t, y[0], y[1]
            if t_end is not None:
                fate = "time"
                break
            continue
        n_imp += 1
        audit["residual"] = max(audit["residual"], res)
        audit["chronological"] = audit["chronological"] and t > last_t
        last_t = t
        v = c.cv * y[1]
        if y[1]:
            audit["ratio_err"] = max(audit["ratio_err"], abs(abs(v / y[1]) - c.cv))
            audit["energy_err"] = max(audit["energy_err"], abs((v * v) / (y[1] * y[1]) - c.cv ** 2))
        x, tau = 0.0, t
        if abs(v) < V_REST:
            break
        s = 1 if v > 0 else -1
    return dict(fate=fate, tau=tau, impacts=n_imp, acc=acc, segs=segs, audit=audit)


PHI_SAMPLES = 2000          # per post-pulse energy segment ("h"; "h2" doubles it)


def post_contact_exact(c, x, v, s):
    """Contact forces after the pulse, exactly on the free-rocking energy curve.

    With no ground acceleration the energy is constant between landings, so on a
    branch the speed is a function of the angle alone:
        alpha^2 v^2 = 2 [ e (1 - cos alpha) - (cos phi - cos alpha) ],  phi = alpha (1 - |x|),
    and the forces follow with f = 0:
        N = 1 - (3/4) [ alpha^2 v^2 cos phi + sin^2 phi ],
        |H| = (3/4) sin phi | alpha^2 v^2 - cos phi |.
    The motion is walked segment by segment -- the rest of the current swing, then each
    whole swing after a landing (energy x c_v^2) -- until E/E_b < POST_E_FLOOR or the
    block reaches |x| = 1. Each segment is sampled in phi at PHI_SAMPLES and twice that,
    endpoints included (N is least at x = 0, an endpoint)."""
    al = c.alpha
    ca = math.cos(al)
    e = c.energy(x, v)
    phi_now = al * (1.0 - abs(x))
    outward = s * v > 0.0

    def turn(e):
        return 0.0 if e >= 1.0 else math.acos(min(1.0, ca + e * (1.0 - ca)))
    segs = []
    if outward:
        if e >= 1.0:
            segs.append((e, 0.0, phi_now))
        else:
            segs.append((e, turn(e), al))
            e = c.cv ** 2 * e
    else:
        segs.append((e, phi_now, al))
        e = c.cv ** 2 * e
    while segs[-1][1] > 0.0 and e >= POST_E_FLOOR:
        segs.append((e, turn(e), al))
        e = c.cv ** 2 * e
    out = {}
    for tag, n in (("h", PHI_SAMPLES), ("h2", 2 * PHI_SAMPLES)):
        nmin, mu = 1.0, 0.0
        for ee, lo, hi in segs:
            phi = np.linspace(lo, hi, n + 1)
            a2v2 = np.maximum(0.0, 2.0 * (ee * (1.0 - ca) - (np.cos(phi) - ca)))
            N = 1.0 - 0.75 * (a2v2 * np.cos(phi) + np.sin(phi) ** 2)
            H = 0.75 * np.sin(phi) * np.abs(a2v2 - np.cos(phi))
            nmin = min(nmin, float(N.min()))
            mu = max(mu, float(np.max(H / N)))
        out["post_x_Nmin_" + tag] = nmin
        out["post_x_mu_" + tag] = mu
    out["post_x_segments"] = len(segs)
    out["post_x_fate"] = "over" if segs[-1][1] == 0.0 else "below floor"
    return out


def post_pulse(c, x, v, s, impl):
    r = free_rock(c, x, v, s, c.tau_p, impl)
    return dict(post_fate=r["fate"], post_impacts=r["impacts"], post_audit=r["audit"],
                post_Nmin_h=r["acc"]["h"]["Nmin"], post_Nmin_h2=r["acc"]["h2"]["Nmin"],
                post_mu_h=r["acc"]["h"]["mu"], post_mu_h2=r["acc"]["h2"]["mu"])


# ------------------------------------------------------------------------------
# jobs (top level so they can run in worker processes)

def job_cell(args):
    key, impl, post = args
    c = Case(**key)
    return simulate(c, impl, post=post)


def job_bisect(args):
    """Bisect an unlike-outcome adjacency (lo, hi) in amplitude for one implementation."""
    key, impl, lo, hi = args
    def outcome(A):
        k = dict(key, A=A)
        return simulate(Case(**k), impl)["outcome"]
    o_lo, o_hi = outcome(lo), outcome(hi)
    start = (lo, hi, o_lo, o_hi)
    if o_lo == o_hi:
        return dict(key=key, impl=impl, lo=lo, hi=hi, o_lo=o_lo, o_hi=o_hi, width=hi - lo, lost=True,
                    start=start)
    while hi - lo > BISECT_WIDTH:
        m = 0.5 * (lo + hi)
        om = outcome(m)
        if om == o_lo:
            lo = m
        else:
            hi = m
    return dict(key=key, impl=impl, lo=lo, hi=hi, o_lo=o_lo, o_hi=outcome(hi), width=hi - lo, lost=False,
                start=start)


def job_rect(args):
    ptd, impl = args
    key = dict(R=RECT_R, A=1.0, alpha=RECT_ALPHA, pulse="rect", ptd=ptd)
    # first overturning amplitude on a 0.01 scan, then bisection
    prev = 1.0
    A = 1.0
    found = None
    for k in range(101, 501):
        A = k / 100.0
        if simulate(Case(**dict(key, A=A)), impl)["outcome"] == "O":
            found = (prev, A)
            break
        prev = A
    if found is None:
        return dict(ptd=ptd, impl=impl, lo=None, hi=None)
    r = job_bisect((key, impl, found[0], found[1]))
    return dict(ptd=ptd, impl=impl, lo=r["lo"], hi=r["hi"], o_lo=r["o_lo"], o_hi=r["o_hi"],
                closed=1.0 / (1.0 - math.exp(-ptd)))


def job_similarity(args):
    (Tp1, R1), (Tp2, R2), impl = args
    out = []
    for Tp, R in ((Tp1, R1), (Tp2, R2)):
        c = Case(R=R, A=A_HEAD, Tp=Tp)
        r = simulate(c, impl, keep=True)
        out.append((c, r))
    (c1, r1), (c2, r2) = out
    tau_end = min(c1.tau_p, c2.tau_p)
    if r1["tau_over"] is not None:
        tau_end = min(tau_end, r1["tau_over"])
    if r2["tau_over"] is not None:
        tau_end = min(tau_end, r2["tau_over"])
    grid = np.linspace(0.0, tau_end, 4001)

    def xs(r):
        vals = np.zeros_like(grid)
        for sg in r["_segs"]:
            m = (grid >= sg["t0"]) & (grid <= sg["t1"])
            if m.any():
                vals[m] = sg["sol"].sol(grid[m])[0]
        return vals
    x1, x2 = xs(r1), xs(r2)
    return dict(pair=[[Tp1, R1], [Tp2, R2]], impl=impl, Om=[c1.Om, c2.Om],
                outcomes=[r1["outcome"], r2["outcome"]], impacts=[r1["impacts"], r2["impacts"]],
                xmax_h=[r1["xmax_h"], r2["xmax_h"]], xmax_h2=[r1["xmax_h2"], r2["xmax_h2"]],
                discrepancy=float(np.max(np.abs(x1 - x2))), x1=x1.tolist())


# ------------------------------------------------------------------------------
# display trajectories (production implementation only)

def trajectory(R, A=A_HEAD, t_show=30.0, fps=120, cv_factor=1.0):
    """theta(t) in radians for the animation, from the production solve: the forced phase,
    then free rocking (or, after overturning, the same equation carried on until the block
    lies on its side, |theta| = pi/2, or comes back to theta = 0)."""
    c = Case(R=R, A=A, cv_factor=cv_factor)
    r = simulate(c, "P", keep=True)
    segs = list(r["_segs"])
    tau_show = t_show * c.p
    after = None
    if r["outcome"] == "O":
        last = segs[-1] if segs else None
        tau0 = r["tau_over"] if r["how"] == "event" else c.tau_p
        s = r["s_end"]
        x0, v0 = r["x_end"], r["v_end"]
        # carry the same branch on (forced while the pulse lasts, free afterwards)
        rhs = _rhs(c, s)
        lie = (math.pi / 2) / c.alpha

        def down(t, y):
            return abs(y[0]) - lie
        down.terminal, down.direction = True, 1

        def back(t, y):
            return s * y[0]
        back.terminal, back.direction = True, -1
        sol = solve_ivp(rhs, (tau0, tau_show), [x0, v0], method="DOP853", rtol=1e-10,
                        atol=(1e-12, 1e-12), max_step=c.hmax * 10, events=(down, back), dense_output=True)
        if r["how"] == "energy outward" or r["how"] == "energy after landing":
            # the post-pulse motion up to |x| = 1 first
            fr = free_rock(c, x0, v0, s, c.tau_p, "P", e_floor=0.0, t_end=None, keep=True)
            segs.extend(fr["segs"])
            tau0 = fr["tau"]
            yb = fr["segs"][-1]["sol"].sol(tau0)
            s = fr["segs"][-1]["s"]
            rhs = _rhs(c, s)
            sol = solve_ivp(rhs, (tau0, tau_show), [float(yb[0]), float(yb[1])], method="DOP853",
                            rtol=1e-10, atol=(1e-12, 1e-12), max_step=c.hmax * 10, events=(down, back),
                            dense_output=True)
        t_end = sol.t[-1]
        after = "lies on its side" if sol.t_events[0].size else ("came back" if sol.t_events[1].size else "moving")
        segs.append(dict(s=s, t0=tau0, t1=t_end, sol=sol))
        tau_rest_from = t_end
        rest_angle = s * math.pi / 2 if after == "lies on its side" else None
    else:
        tau_rest_from = None
        rest_angle = 0.0
        if r["how"] != "rest":
            fr = free_rock(c, r["x_end"], r["v_end"], r["s_end"], c.tau_p, "P", e_floor=1e-12,
                           t_end=None, keep=True)
            segs.extend(fr["segs"])
            tau_rest_from = fr["tau"]
        else:
            tau_rest_from = c.tau_p
    ts = np.arange(0.0, t_show + 1e-12, 1.0 / fps)
    theta = np.zeros_like(ts)
    taus = ts * c.p
    for sg in segs:
        m = (taus >= sg["t0"]) & (taus <= sg["t1"])
        if m.any():
            theta[m] = sg["sol"].sol(taus[m])[0] * c.alpha
    if tau_rest_from is not None:
        m = taus > tau_rest_from
        if rest_angle is not None:
            theta[m] = rest_angle
    theta[taus < (r["uplifts"][0] if r["uplifts"] else math.inf)] = 0.0
    ag = np.array([c.f(t) for t in taus]) * G                    # m/s^2
    wp = 2 * math.pi / c.Tp
    ap = c.amp * G
    tt = np.clip(ts, 0.0, c.Tp)
    ug = ap / wp ** 2 * (wp * tt - np.sin(wp * tt))              # ground displacement, m
    return dict(R=R, A=A, b=R * math.sin(c.alpha), h=R * math.cos(c.alpha), outcome=r["outcome"],
                how=r["how"], impacts=r["impacts"], after=after, fps=fps, t=ts.tolist(),
                theta=theta.tolist(), ag=ag.tolist(), ug=ug.tolist(),
                tau_over_s=(r["tau_over"] / c.p if r["tau_over"] is not None else None),
                uplift_s=(r["uplifts"][0] / c.p if r["uplifts"] else None))


# ------------------------------------------------------------------------------

def u_of(qP, qT, qE, qh, qh2):
    return abs(qP - qT) + abs(qT - qE) + abs(qh - qh2) + 64 * EPS * max(1.0, abs(qT))


def main():
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)
    workers = max(1, (os.cpu_count() or 2) - 2)
    say("rocking block study: %d workers" % workers)
    import hashlib
    src = open(os.path.abspath(__file__), "rb").read().replace(b"\r\n", b"\n")
    doc = dict(source_sha256=hashlib.sha256(src).hexdigest(), model=dict(g=G, alpha=ALPHA, Tp=TP, A_head=A_HEAD, sizes=SIZES, head_pair=HEAD_PAIR,
                          cv_housner=cv_housner(ALPHA), v_rest=V_REST, post_e_floor=POST_E_FLOOR,
                          impl=IMPL, registration="design review, locked before the production run"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # 1. amplitude scan, every size, every implementation
        jobs = []
        for R in SIZES:
            for k in range(SCAN_HUNDREDTHS[0], SCAN_HUNDREDTHS[1] + 1):
                A = k / 100.0
                for impl in IMPLS:
                    jobs.append((dict(R=R, A=A), impl, abs(A - A_HEAD) < 1e-12))
        say("scan: %d solves" % len(jobs))
        scan = list(ex.map(job_cell, jobs, chunksize=8))
        say("  done %.0f s" % (time.time() - t_start))
        # 2. transitions: unlike neighbours on the 0.01 grid, per implementation
        bis_jobs = []
        adj = {}
        for R in SIZES:
            for impl in IMPLS:
                cells = sorted([r for r in scan if r["case"]["R"] == R and r["impl"] == impl],
                               key=lambda r: r["case"]["A"])
                for a, b in zip(cells, cells[1:]):
                    if a["outcome"] != b["outcome"]:
                        adj.setdefault((R, impl), []).append((a["case"]["A"], b["case"]["A"]))
                        bis_jobs.append((dict(R=R, A=a["case"]["A"]), impl, a["case"]["A"], b["case"]["A"]))
        say("transitions to bisect: %d" % len(bis_jobs))
        bis = list(ex.map(job_bisect, bis_jobs, chunksize=1))
        say("  done %.0f s" % (time.time() - t_start))
        # 3. restitution sensitivity
        rjobs = [(dict(R=R, A=A_HEAD, cv_factor=fct), impl, True)
                 for fct in CV_FACTORS for R in SIZES for impl in IMPLS]
        rest = list(ex.map(job_cell, rjobs, chunksize=2))
        # 4. rectangular-pulse control
        rect = list(ex.map(job_rect, [(ptd, impl) for ptd in RECT_PTD for impl in IMPLS]))
        # 5. similarity
        sim = list(ex.map(job_similarity, [(a, b, impl) for a, b in SIMILARITY for impl in IMPLS]))
        say("  controls done %.0f s" % (time.time() - t_start))
    doc["scan"] = scan
    doc["bisections"] = bis
    doc["restitution"] = rest
    doc["rect"] = rect
    doc["similarity"] = sim

    # transitions with their uncertainty, matched across implementations by grid adjacency
    trans = []
    for R in SIZES:
        pairs = sorted(set(p for impl in ("P", "T", "E") for p in adj.get((R, impl), [])))
        for lo, hi in pairs:
            got = {b["impl"]: b for b in bis if b["key"]["R"] == R and abs(b["start"][0] - lo) < 1e-12}
            if not all(i in got for i in ("P", "T", "E")):
                trans.append(dict(R=R, grid=[lo, hi], matched=False, brackets={k: [v["lo"], v["hi"]] for k, v in got.items()}))
                continue
            mid = {i: 0.5 * (got[i]["lo"] + got[i]["hi"]) for i in ("P", "T", "E")}
            w = max(got[i]["hi"] - got[i]["lo"] for i in ("P", "T", "E"))
            uA = 0.5 * w + abs(mid["P"] - mid["T"]) + abs(mid["T"] - mid["E"]) + 64 * EPS * max(1.0, abs(mid["T"]))
            trans.append(dict(R=R, grid=[lo, hi], matched=True,
                              brackets={i: [got[i]["lo"], got[i]["hi"], got[i]["o_lo"], got[i]["o_hi"]]
                                        for i in ("P", "T", "E")} |
                              ({"E2": [got["E2"]["lo"], got["E2"]["hi"], got["E2"]["o_lo"], got["E2"]["o_hi"]]}
                               if "E2" in got else {}),
                              u_A=uA, A_T=mid["T"], lost=any(got[i]["lost"] for i in got)))
    doc["transitions"] = trans

    # resolved outcomes on the scan
    cells = {}
    for r in scan:
        cells.setdefault((r["case"]["R"], r["case"]["A"]), {})[r["impl"]] = r
    table = []
    for (R, A), by in sorted(cells.items()):
        outs = {i: by[i]["outcome"] for i in IMPLS}
        agree = len(set(outs.values())) == 1
        near = [t for t in trans if t["R"] == R and t.get("matched") and abs(A - t["A_T"]) <= 3 * t["u_A"]]
        table.append(dict(R=R, A=A, outcome=(outs["P"] if agree and not near else "UNRESOLVED"), impl=outs))
    doc["table"] = table

    # display trajectories
    doc["display"] = {("R%.2f" % R): trajectory(R) for R in SIZES}
    doc["runtime_s"] = time.time() - t_start
    say("runtime %.0f s" % doc["runtime_s"])
    with open(os.path.join(OUT, "rocking_block.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, default=lambda o: None)
    with open(os.path.join(OUT, "study.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log) + "\n")
    return doc


if __name__ == "__main__":
    main()
