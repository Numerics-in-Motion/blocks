# -*- coding: utf-8 -*-
"""How far past a table edge can identical blocks be balanced?

THE QUESTION

Stack identical blocks at a table edge. Stacked one on one -- each block
resting on a single block below -- the limit is the textbook harmonic stack.
Allowed to put more than one block on a level, a block can act as a
counterweight. How much further out can the same blocks then be balanced?

THE MODEL

Paterson & Zwick, "Overhang", Amer. Math. Monthly 116 (2009) 19-44,
arXiv:0710.2357. 2-D blocks of length 1 and weight 1 lie in levels; level 0
rests on a table whose top is x <= 0. A block rests on the blocks of the level
below whose intervals meet its own. A stack is BALANCED when non-negative
vertical forces at the two ends of every contact interval hold every block in
force and moment balance (their Theorem 2.2): a linear programme. D(n) is the
largest overhang -- the right end of the block furthest out -- of a balanced
stack of n blocks. Blocks are rigid, identical, homogeneous, and the contact is
frictionless: every force is vertical.

WHAT IS COMPUTED

  enumerate      every contact topology of n = 1..6 blocks: a split of the
                 blocks into levels, and for every block the run of blocks
                 below it that it touches (closed intervals: a point counts).
                 An LP keeps each candidate only if it exists with EXACTLY its
                 own contacts -- every undeclared pair held apart by a positive
                 clearance; a candidate whose placements all touch more blocks
                 is covered by the richer topology, and one with no placement
                 at all is empty. Random and grid stacks are classified to
                 check that every actual contact pattern is in the list.
  formulation A  per topology and candidate furthest block, positions and end
                 forces (Paterson-Zwick's own variables) by SLSQP from several
                 starts and a warm restart.
  formulation B  the same maximum written separately: one resultant per
                 contact acting at a point inside the contact, torques r x F in
                 2-D with the block height carried through.
  unbalanceable  a topology neither formulation can balance is certified, by
                 branch and bound on one connected component, to have no
                 balanced placement at all.
  certify        branch and bound over positions with McCormick envelopes of
                 the force x position products: an upper bound on D(n) for
                 n = 1..4, within a stated gap, that does not depend on any
                 start.
  harmonic       the one-on-one stack balances at H_n/2 and not 1e-6 beyond.
  margin         the largest overhang when every resultant must stay a margin
                 inside its contact interval, for the one-on-one class and for
                 every topology -- the model's balance limit has zero margin.

WHAT IS NOT CLAIMED

Balanced, and nothing more: the limit stacks have zero margin. No friction, no
tipping or sliding dynamics (the moves the video draws are quasi-static
repositionings through balanced states), no rounded edges, uneven weights,
deformation or 3-D effects, nothing about real blocks, and nothing beyond
n = 6.
"""
from __future__ import annotations

import io
import itertools
import json
import math
import os
import sys
import time
import zlib
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog, minimize, nnls

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("OVERHANG_DATA") or os.path.join(ROOT, "data")

N_VALIDATE = (1, 2, 3, 4, 5, 6)
N_CLAIM = (3, 4)
# Registered with the design review: one-on-one H_n/2 and the
# published unrestricted optima, with the precision each was published to.
REGISTERED = {1: ("0.500000", 6), 2: ("0.750000", 6), 3: ("1.000000", 6),
              4: ("1.167893219", 9), 5: ("1.30455", 5), 6: ("1.43670", 5)}
STARTS = 24
POLISH = 3
SEED = 39
LP_TOLERANCES = (1e-7, 1e-10)
PUSH = 1e-6
CERT_EPS = 2e-3                 # the certificate's gap; it enters u
CERT_N = (1, 2, 3, 4)           # n = 1, 2 close the argument for stacks that fall apart
HEIGHTS = (0.1, 0.25, 1.0, 2.0)
MARGINS = (0.0, 1e-6, 1e-3, 0.005, 0.01, 0.02, 0.05)
GEOM_TOL = 1e-9


# ---------------------------------------------------------------- topologies
def compositions(n):
    """Blocks per level, bottom level first."""
    for k in range(1, n + 1):
        for cut in itertools.combinations(range(1, n), k - 1):
            yield tuple(b - a for a, b in zip((0,) + cut, cut + (n,)))


def relations(k, m):
    """Contact relations between an upper level of k blocks and a lower level of
    m blocks, both ordered left to right: upper block i rests on lower blocks
    s_i..e_i. Contact intervals are closed, so a unit block can touch THREE
    disjoint unit blocks -- the outer two at a single point, when they sit end to
    end under it -- and a lower block can carry three; the runs move right with
    i. Whether a relation can be realised is left to the LP in `realisable`."""
    spans = [(s, e) for s in range(m) for e in range(s, min(s + 3, m))]
    out = []

    def rec(i, prev, acc):
        if i == k:
            used = [0] * m
            for s, e in acc:
                for j in range(s, e + 1):
                    used[j] += 1
            if max(used) <= 3:
                out.append(tuple(acc))
            return
        for s, e in spans:
            if prev is None or (s >= prev[0] and e >= prev[1]):
                rec(i + 1, (s, e), acc + [(s, e)])
    rec(0, None, [])
    return out


class Topology:
    """A split into levels and who rests on whom."""

    def __init__(self, comp, rels):
        self.comp = tuple(comp)
        self.rels = tuple(tuple(r) for r in rels)
        self.n = sum(comp)
        self.first = [int(v) for v in np.cumsum((0,) + self.comp[:-1])]
        self.levels = [l for l, c in enumerate(self.comp) for _ in range(c)]
        self.contacts = [(self.first[0] + q, -1) for q in range(self.comp[0])]
        self.apart = []                       # (left block, right block) at adjacent levels
        for l in range(1, len(self.comp)):
            for q, (s, e) in enumerate(self.rels[l - 1]):
                i = self.first[l] + q
                for p in range(self.comp[l - 1]):
                    j = self.first[l - 1] + p
                    if s <= p <= e:
                        self.contacts.append((i, j))
                    elif p < s:
                        self.apart.append((j, i))
                    else:
                        self.apart.append((i, j))
        self.rightmost = [self.first[l] + c - 1 for l, c in enumerate(self.comp)]

    def key(self):
        return "%s|%s" % (",".join(map(str, self.comp)),
                          ";".join("".join("%d%d" % se for se in r) for r in self.rels))

    def geometry_rows(self):
        """Linear constraints on positions, G x >= h."""
        n, G, h = self.n, [], []

        def ge(coef, rhs):
            r = np.zeros(n)
            for k, v in coef:
                r[k] += v
            G.append(r)
            h.append(rhs)
        for l, c in enumerate(self.comp):
            for q in range(c - 1):
                ge([(self.first[l] + q + 1, 1), (self.first[l] + q, -1)], 1.0)
        for q in range(self.comp[0]):
            ge([(self.first[0] + q, -1)], 0.0)
        for i, j in self.contacts:
            if j >= 0:
                ge([(j, 1), (i, -1)], -1.0)
                ge([(i, 1), (j, -1)], -1.0)
        for a, b in self.apart:
            ge([(b, 1), (a, -1)], 1.0)
        return np.array(G).reshape(-1, n), np.array(h)


def topologies(n):
    for comp in compositions(n):
        per = [relations(comp[l], comp[l - 1]) for l in range(1, len(comp))]
        for rels in itertools.product(*per):
            yield Topology(comp, rels)


CLEARANCE_TOL = 1e-6


def clearance(t):
    """The largest common clearance delta by which every pair NOT declared in
    contact can be held apart while every declared contact stays closed. With
    closed contact intervals, two blocks at adjacent levels exactly end to end
    touch, so a topology whose undeclared pairs can only sit at clearance 0 does
    not exist with exactly its own contacts: every placement of it has more.
    -1 if even the closure is empty."""
    G, h = t.geometry_rows()
    n = t.n
    apart_rows = len(t.apart)
    D = np.zeros(G.shape[0])
    if apart_rows:
        D[-apart_rows:] = 1.0                    # the apart rows come last
    A = np.hstack([-G, D[:, None]])
    c = np.zeros(n + 1)
    c[-1] = -1.0
    res = linprog(c, A_ub=A, b_ub=-h, bounds=[(-n, n)] * n + [(0.0, 1.0)], method="highs",
                  options=dict(primal_feasibility_tolerance=1e-10,
                               dual_feasibility_tolerance=1e-10))
    if res.status != 0:
        return -1.0
    return float(-res.fun) if apart_rows else 1.0


def realisable(t):
    """Exactly its own contacts: undeclared pairs held apart by a positive clearance."""
    return clearance(t) > CLEARANCE_TOL


# ------------------------------------------------------------- balance (LP)
def contacts_of(levels, xs, tol=GEOM_TOL):
    """Contact intervals of a placed stack: (upper, lower or -1, a, b)."""
    out = []
    for i, (li, xi) in enumerate(zip(levels, xs)):
        if li == 0:
            if xi <= tol:
                out.append((i, -1, xi, min(xi + 1.0, 0.0)))
            continue
        for j, (lj, xj) in enumerate(zip(levels, xs)):
            if lj == li - 1 and max(xi, xj) <= min(xi, xj) + 1.0 + tol:
                out.append((i, j, max(xi, xj), min(xi, xj) + 1.0))
    return out


def _system(levels, xs, cs):
    n, m = len(levels), 2 * len(cs)
    A = np.zeros((2 * n, m))
    b = np.zeros(2 * n)
    for c, (i, j, a, bb) in enumerate(cs):
        for k, s in ((i, 1.0), (j, -1.0)):
            if k < 0:
                continue
            A[k, 2 * c] += s
            A[k, 2 * c + 1] += s
            A[n + k, 2 * c] += s * a
            A[n + k, 2 * c + 1] += s * bb
    for k in range(n):
        b[k] = 1.0
        b[n + k] = xs[k] + 0.5
    return A, b


def balance(levels, xs, tol=1e-9):
    """Paterson-Zwick Theorem 2.2: is there a balancing set of end forces?"""
    cs = contacts_of(levels, xs)
    if any(levels[k] > 0 and not any(i == k for i, *_ in cs) for k in range(len(levels))) \
            or any(l == 0 and x > GEOM_TOL for l, x in zip(levels, xs)):
        return dict(ok=False, residual=None, contacts=cs)
    A, b = _system(levels, xs, cs)
    res = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=[(0, None)] * A.shape[1],
                  method="highs", options=dict(primal_feasibility_tolerance=tol,
                                               dual_feasibility_tolerance=tol))
    if res.status != 0:
        return dict(ok=False, residual=None, contacts=cs)
    f = res.x
    return dict(ok=True, residual=float(np.max(np.abs(A @ f - b))),
                negative=float(max(0.0, -f.min())), forces=f.tolist(), contacts=cs)


def display_forces(levels, xs, weight=1e5, tol=1e-9):
    """One balancing set -- the minimum-norm non-negative one -- as a resultant
    per contact. Unique for statically determinate stacks; for the others it is
    one of many. The equations are met exactly, not by penalty: a weighted NNLS
    picks the contacts that carry force, the minimum-norm solution on those is
    solved exactly, and an equality-constrained QP is the fallback. Returns
    (resultants, residual, most negative force); callers check both against tol."""
    cs = contacts_of(levels, xs)
    A, b = _system(levels, xs, cs)
    m = A.shape[1]
    M = np.vstack([weight * A, np.eye(m)])
    rhs = np.concatenate([weight * b, np.zeros(m)])
    f0, _ = nnls(M, rhs, maxiter=50 * m)
    f = np.zeros(m)
    S = f0 > 1e-12
    if S.any():
        f[S] = np.linalg.pinv(A[:, S]) @ b
    if np.max(np.abs(A @ f - b)) > tol or f.min() < -tol:
        r = minimize(lambda v: 0.5 * v @ v, f0, jac=lambda v: v, bounds=[(0, None)] * m,
                     constraints=[dict(type="eq", fun=lambda v: A @ v - b, jac=lambda v: A)],
                     method="SLSQP", options=dict(maxiter=500, ftol=1e-15))
        f = r.x
    resid = float(np.max(np.abs(A @ f - b)))
    neg = float(max(0.0, -f.min()))
    f = np.maximum(f, 0.0)
    out = []
    for c, (i, j, a, bb) in enumerate(cs):
        R = f[2 * c] + f[2 * c + 1]
        p = (a * f[2 * c] + bb * f[2 * c + 1]) / R if R > 1e-12 else 0.5 * (a + bb)
        out.append(dict(upper=i, lower=j, a=a, b=bb, R=float(R), at=float(p)))
    return out, resid, neg


def balance_resultant(levels, xs, height=1.0, margin=0.0, tol=1e-9):
    """Formulation B for a placed stack, written apart from `balance`: a resultant
    R >= 0 and its moment M per contact, R (a + margin) <= M <= R (b - margin),
    and torques r x F in 2-D about the origin with every block `height` tall.
    Contact is frictionless, so the horizontal components are zero."""
    n = len(levels)
    pairs = []
    for i in range(n):
        if levels[i] == 0:
            if xs[i] <= GEOM_TOL:
                pairs.append((i, -1, xs[i], min(xs[i] + 1.0, 0.0)))
        else:
            for j in range(n):
                if levels[j] == levels[i] - 1:
                    lo, hi = max(xs[i], xs[j]), min(xs[i], xs[j]) + 1.0
                    if lo <= hi + GEOM_TOL:
                        pairs.append((i, j, lo, hi))
    nv = 2 * len(pairs)
    Aeq, beq, Aub, bub = [], [], [], []
    for k in range(n):
        fy, tz = np.zeros(nv), np.zeros(nv)
        yk = levels[k] * height
        for c, (i, j, lo, hi) in enumerate(pairs):
            s = 1.0 if i == k else (-1.0 if j == k else 0.0)
            if not s:
                continue
            y_contact = (levels[i] * height)                # base of the upper block
            hx = 0.0                                          # frictionless
            fy[2 * c] += s
            tz[2 * c + 1] += s                                # x * Fy
            tz[2 * c] += -s * y_contact * hx                  # - y * Fx
        weight_torque = (xs[k] + 0.5) * (-1.0) - (yk + 0.5 * height) * 0.0
        Aeq.append(fy)
        beq.append(1.0)
        Aeq.append(tz)
        beq.append(-weight_torque)
    for c, (i, j, lo, hi) in enumerate(pairs):
        r = np.zeros(nv); r[2 * c] = lo + margin; r[2 * c + 1] = -1.0
        Aub.append(r); bub.append(0.0)
        r = np.zeros(nv); r[2 * c + 1] = 1.0; r[2 * c] = -(hi - margin)
        Aub.append(r); bub.append(0.0)
    bounds = []
    for _ in pairs:
        bounds += [(0, None), (None, None)]
    if not pairs:
        return False
    res = linprog(np.zeros(nv), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bounds, method="highs",
                  options=dict(primal_feasibility_tolerance=tol,
                               dual_feasibility_tolerance=tol))
    return res.status == 0


def edge_margin(levels, xs, height=1.0):
    """The largest distance every resultant can keep from the ends of its contact
    interval (bisection on formulation B). Zero at a balance limit."""
    if not balance_resultant(levels, xs, height):
        return None
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if balance_resultant(levels, xs, height, margin=mid):
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- harmonic
def harmonic_fraction(n):
    return sum(Fraction(1, 2 * k) for k in range(1, n + 1))


def harmonic_stack(n):
    """One block per level; block l from the bottom ends sum_{k=n-l}^{n} 1/2k past the edge."""
    xs = [float(sum(Fraction(1, 2 * k) for k in range(n - l, n + 1))) - 1.0
          for l in range(n)]
    return list(range(n)), xs


def harmonic_check(n):
    levels, xs = harmonic_stack(n)
    pushed_top = list(xs)
    pushed_top[-1] += PUSH
    pushed_all = [x + PUSH for x in xs]
    return dict(n=n, xs=xs, overhang=xs[-1] + 1.0, exact=str(harmonic_fraction(n)),
                balanced=balance(levels, xs)["ok"],
                balanced_b=balance_resultant(levels, xs),
                top_pushed=balance(levels, pushed_top)["ok"],
                all_pushed=balance(levels, pushed_all)["ok"])


# ------------------------------------------------- formulation A (end forces)
def _setup_a(t, margin=0.0):
    n, nc = t.n, len(t.contacts)
    nv = n + 4 * nc
    G0, h0 = t.geometry_rows()
    rows, rhs = [np.hstack([G0, np.zeros((G0.shape[0], 4 * nc))])], [h0]

    def var(c, k):
        return n + 4 * c + k                      # a, b, f0, f1
    extra, erhs = [], []
    for c, (i, j) in enumerate(t.contacts):
        r = np.zeros(nv); r[var(c, 0)] = 1; r[i] = -1; extra.append(r); erhs.append(margin)
        r = np.zeros(nv); r[i] = 1; r[var(c, 1)] = -1; extra.append(r); erhs.append(margin - 1.0)
        if j >= 0:
            r = np.zeros(nv); r[var(c, 0)] = 1; r[j] = -1; extra.append(r); erhs.append(margin)
            r = np.zeros(nv); r[j] = 1; r[var(c, 1)] = -1; extra.append(r); erhs.append(margin - 1.0)
        else:
            r = np.zeros(nv); r[var(c, 1)] = -1; extra.append(r); erhs.append(margin)
        r = np.zeros(nv); r[var(c, 1)] = 1; r[var(c, 0)] = -1; extra.append(r); erhs.append(0.0)
    G = np.vstack(rows + [np.array(extra)])
    h = np.concatenate(rhs + [np.array(erhs)])
    F = np.zeros((n, nv))
    for c, (i, j) in enumerate(t.contacts):
        for k, s in ((i, 1.0), (j, -1.0)):
            if k >= 0:
                F[k, var(c, 2)] += s
                F[k, var(c, 3)] += s

    def moment(z):
        out = -(z[:n] + 0.5)
        for c, (i, j) in enumerate(t.contacts):
            a, b, f0, f1 = z[var(c, 0):var(c, 0) + 4]
            v = a * f0 + b * f1
            out[i] += v
            if j >= 0:
                out[j] -= v
        return out

    def moment_jac(z):
        J = np.zeros((n, nv))
        J[np.arange(n), np.arange(n)] = -1.0
        for c, (i, j) in enumerate(t.contacts):
            a, b, f0, f1 = z[var(c, 0):var(c, 0) + 4]
            for k, s in ((i, 1.0), (j, -1.0)):
                if k >= 0:
                    J[k, var(c, 0)] += s * f0
                    J[k, var(c, 1)] += s * f1
                    J[k, var(c, 2)] += s * a
                    J[k, var(c, 3)] += s * b
        return J
    bounds = [(-n, n - 1)] * n
    for _ in t.contacts:
        bounds += [(-n, n), (-n, n), (0, n), (0, n)]
    return nv, G, h, F, moment, moment_jac, bounds


def _start_positions(t, rng):
    G, h = t.geometry_rows()
    res = linprog(rng.normal(size=t.n), A_ub=-G, b_ub=-(h + 0.02 * rng.random(len(h))),
                  bounds=[(-t.n, t.n - 1)] * t.n, method="highs")
    if res.status != 0:
        res = linprog(rng.normal(size=t.n), A_ub=-G, b_ub=-h,
                      bounds=[(-t.n, t.n - 1)] * t.n, method="highs")
    return res.x


def _seed(seed, t, target):
    return [seed, t.n, zlib.crc32(t.key().encode()), target]


def optimise_a(t, target, starts=STARTS, seed=SEED, ftol=1e-12, margin=0.0):
    """With margin > 0 every resultant keeps that distance from the ends of its
    contact interval (the table edge included)."""
    nv, G, h, F, moment, moment_jac, bounds = _setup_a(t, margin)
    n = t.n
    rng = np.random.default_rng(_seed(seed, t, target))
    cons = [dict(type="ineq", fun=lambda z: G @ z - h, jac=lambda z: G),
            dict(type="eq", fun=lambda z: F @ z - 1.0, jac=lambda z: F),
            dict(type="eq", fun=moment, jac=moment_jac)]
    obj = np.zeros(nv)
    obj[target] = -1.0
    best = None
    for s in range(starts):
        x0 = _start_positions(t, rng)
        z0 = np.zeros(nv)
        z0[:n] = x0
        for c, (i, j) in enumerate(t.contacts):
            lo = max(x0[i], x0[j]) if j >= 0 else x0[i]
            hi = min(x0[i], x0[j]) + 1 if j >= 0 else min(x0[i] + 1, 0.0)
            z0[n + 4 * c:n + 4 * c + 4] = [lo, max(lo, hi), 0.5 * rng.random(), 0.5 * rng.random()]
        best = _keep(best, _solve_a(z0, obj, bounds, cons, ftol, G, h, F, moment, n,
                                    len(t.contacts), target))
    for _ in range(POLISH if best else 0):
        again = _solve_a(np.array(best["z"]), obj, bounds, cons, ftol, G, h, F, moment,
                         n, len(t.contacts), target)
        if not again or again["value"] <= best["value"] + 1e-15:
            break
        best = again
    return best


def _keep(best, new):
    return new if new and (best is None or new["value"] > best["value"]) else best


def _solve_a(z0, obj, bounds, cons, ftol, G, h, F, moment, n, nc, target):
    r = minimize(lambda z: obj @ z, z0, jac=lambda z: obj, bounds=bounds,
                 constraints=cons, method="SLSQP", options=dict(maxiter=500, ftol=ftol))
    z = r.x
    viol = max(float(np.max(np.maximum(h - G @ z, 0))),
               float(np.max(np.abs(F @ z - 1.0))),
               float(np.max(np.abs(moment(z)))),
               float(max(0.0, -min(z[n + 4 * c + k] for c in range(nc) for k in (2, 3)))))
    if viol > 1e-7:
        return None
    return dict(value=float(z[target] + 1.0), xs=z[:n].tolist(), violation=viol,
                z=z.tolist())


# --------------------------------------------- formulation B (resultants)
def optimise_b(t, target, starts=STARTS, seed=SEED + 1, height=1.0):
    """Written apart from formulation A: one resultant R >= 0 per contact acting
    at a point p inside the contact interval (max of the left ends <= p <= min
    of the right ends, the table top included), and every block's torque
    balance as r x F in 2-D with the block `height` carried through. Contact is
    frictionless, so every horizontal component is zero."""
    n, pairs = t.n, t.contacts
    nc = len(pairs)
    nv = n + 2 * nc
    G0, h0 = t.geometry_rows()
    rows, rhs = [np.hstack([G0, np.zeros((G0.shape[0], 2 * nc))])], list(h0)
    extra = []

    def P(c):
        return n + nc + c

    def Rv(c):
        return n + c
    for c, (i, j) in enumerate(pairs):
        for k in ([i, j] if j >= 0 else [i]):
            r = np.zeros(nv); r[P(c)] = 1.0; r[k] = -1.0; extra.append(r); rhs.append(0.0)
            r = np.zeros(nv); r[k] = 1.0; r[P(c)] = -1.0; extra.append(r); rhs.append(-1.0)
        if j < 0:
            r = np.zeros(nv); r[P(c)] = -1.0; extra.append(r); rhs.append(0.0)
    G = np.vstack(rows + [np.array(extra)])
    h = np.array(rhs)

    def torque(rx, ry, fx, fy):
        return rx * fy - ry * fx

    def eq(z):
        fy = np.full(n, -1.0)                              # every block's weight
        tz = np.array([torque(z[k] + 0.5, (t.levels[k] + 0.5) * height, 0.0, -1.0)
                       for k in range(n)])
        for c, (i, j) in enumerate(pairs):
            y = t.levels[i] * height                       # the upper block's base
            for k, s_ in ((i, 1.0), (j, -1.0)):
                if k < 0:
                    continue
                fy[k] += s_ * z[Rv(c)]
                tz[k] += torque(z[P(c)], y, 0.0, s_ * z[Rv(c)])
        return np.concatenate([fy, tz])

    def eq_jac(z):
        J = np.zeros((2 * n, nv))
        for k in range(n):
            J[n + k, k] = -1.0
        for c, (i, j) in enumerate(pairs):
            for k, s_ in ((i, 1.0), (j, -1.0)):
                if k < 0:
                    continue
                J[k, Rv(c)] += s_
                J[n + k, Rv(c)] += s_ * z[P(c)]
                J[n + k, P(c)] += s_ * z[Rv(c)]
        return J

    rng = np.random.default_rng(_seed(seed, t, target))
    bounds = [(-n, n - 1)] * n + [(0, n)] * nc + [(-n, n)] * nc
    grad = -np.eye(nv)[target]
    cons = [dict(type="ineq", fun=lambda z: G @ z - h, jac=lambda z: G),
            dict(type="eq", fun=eq, jac=eq_jac)]

    def solve(z0):
        r = minimize(lambda z: -z[target], z0, jac=lambda z: grad, bounds=bounds,
                     constraints=cons, method="SLSQP", options=dict(maxiter=500, ftol=1e-12))
        z = r.x
        viol = max(float(np.max(np.maximum(h - G @ z, 0))), float(np.max(np.abs(eq(z)))),
                   float(max(0.0, -z[n:n + nc].min())))
        if viol > 1e-7:
            return None
        return dict(value=float(z[target] + 1.0), xs=z[:n].tolist(), violation=viol,
                    z=z.tolist())
    best = None
    for s in range(starts):
        x0 = _start_positions(t, rng)
        z0 = np.concatenate([x0, 0.5 + rng.random(nc), np.zeros(nc)])
        for c, (i, j) in enumerate(pairs):
            lo = max(x0[i], x0[j]) if j >= 0 else x0[i]
            hi = min(x0[i], x0[j]) + 1 if j >= 0 else min(x0[i] + 1, 0.0)
            z0[P(c)] = lo + rng.random() * max(hi - lo, 0.0)
        best = _keep(best, solve(z0))
    for _ in range(POLISH if best else 0):
        again = solve(np.array(best["z"]))
        if not again or again["value"] <= best["value"] + 1e-15:
            break
        best = again
    return best


def _topology_job(args):
    comp, rels, which = args
    t = Topology(comp, rels)
    gap = clearance(t)
    out = dict(key=t.key(), comp=list(comp), clearance=gap, closure=gap >= 0,
               realisable=gap > CLEARANCE_TOL, a={}, b={})
    if not out["realisable"]:
        return out
    for target in t.rightmost:
        if "a" in which:
            out["a"][str(target)] = optimise_a(t, target)
        if "b" in which:
            out["b"][str(target)] = optimise_b(t, target)
    return out


# ------------------------------------------------------ branch and bound
def _relax(levels, lo, hi, target, n, bound, tol):
    """Upper bound of x_target + 1 over a box of positions, for connected balanced
    stacks whose overhang exceeds `bound`; -inf if the box holds none."""
    N = len(levels)
    above = [sum(1 for l in levels if l >= levels[k]) for k in range(N)]
    cand = []
    for i in range(N):
        if levels[i] == 0:
            cand.append((i, -1, "sure"))
            continue
        any_c = False
        for j in range(N):
            if levels[j] != levels[i] - 1:
                continue
            if lo[i] > hi[j] + 1 or lo[j] > hi[i] + 1:
                continue
            sure = hi[i] <= lo[j] + 1 and hi[j] <= lo[i] + 1
            cand.append((i, j, "sure" if sure else "maybe"))
            any_c = True
        if not any_c:
            return -np.inf
    rows = []
    for a_ in range(N):
        for b_ in range(a_ + 1, N):
            if levels[a_] == levels[b_]:
                rows.append(([(a_, 1), (b_, -1)], -1.0))        # left to right
            rows.append(([(a_, 1), (b_, -1)], N - 1.0))         # connected: span <= N
            rows.append(([(b_, 1), (a_, -1)], N - 1.0))
    rows.append(([(target, -1)], -bound + 1.0))                  # x_t + 1 >= bound
    return _relaxation_lp(levels, lo, hi, cand, rows, target, tol)


def _relaxation_lp(levels, lo, hi, cand, rows, target, tol):
    """The McCormick relaxation of the balance equations over a box of positions,
    with extra linear rows (sum coef x <= rhs). Returns the bound on x_target + 1,
    0.0 for a feasibility question (target None), -inf if infeasible, +inf if the
    LP could not decide (never prunes)."""
    N = len(levels)
    above = [sum(1 for l in levels if l >= levels[k]) for k in range(N)]
    nc = len(cand)
    nv = N + 6 * nc

    def cv(c, k):
        return N + 6 * c + k                   # fl, fr, wl, wr, pl, pr
    bounds = [(lo[k], hi[k]) for k in range(N)]
    A, b, E, e = [], [], [], []

    def row(pairs, rhs, store=A, sb=b):
        r = np.zeros(nv)
        for k, v in pairs:
            r[k] += v
        store.append(r)
        sb.append(rhs)
    for c, (i, j, kind) in enumerate(cand):
        F = float(above[i])
        if j < 0:
            pl = (lo[i], hi[i])
            pr = (min(lo[i] + 1, 0.0), min(hi[i] + 1, 0.0))
        else:
            pl = (max(lo[i], lo[j]), max(hi[i], hi[j]))
            pr = (min(lo[i], lo[j]) + 1, min(hi[i], hi[j]) + 1)
        bounds += [(0, F), (0, F), (None, None), (None, None), pl, pr]
        row([(i, 1), (cv(c, 4), -1)], 0.0)                     # pl >= x_i
        row([(cv(c, 5), 1), (i, -1)], 1.0)                     # pr <= x_i + 1
        if j >= 0:
            row([(j, 1), (cv(c, 4), -1)], 0.0)
            row([(cv(c, 5), 1), (j, -1)], 1.0)
        if kind == "sure":
            row([(cv(c, 4), 1), (cv(c, 5), -1)], 0.0)          # pl <= pr
        for fk, wk, pk, (pL, pU) in ((0, 2, 4, pl), (1, 3, 5, pr)):
            f, w, p = cv(c, fk), cv(c, wk), cv(c, pk)
            row([(f, pL), (w, -1)], 0.0)
            row([(f, pU), (p, F), (w, -1)], F * pU)
            row([(w, 1), (f, -pU)], 0.0)
            row([(w, 1), (f, -pL), (p, -F)], -F * pL)
    for k in range(N):
        fr, mr = [], []
        for c, (i, j, kind) in enumerate(cand):
            s = 1.0 if i == k else (-1.0 if j == k else 0.0)
            if s:
                fr += [(cv(c, 0), s), (cv(c, 1), s)]
                mr += [(cv(c, 2), s), (cv(c, 3), s)]
        row(fr, 1.0, E, e)
        row(mr + [(k, -1.0)], 0.5, E, e)
    for pairs, rhs in rows:
        row(pairs, rhs)
    cobj = np.zeros(nv)
    if target is not None:
        cobj[target] = -1.0
    res = linprog(cobj, A_ub=np.array(A).reshape(-1, nv), b_ub=np.array(b),
                  A_eq=np.array(E), b_eq=np.array(e), bounds=bounds, method="highs",
                  options=dict(primal_feasibility_tolerance=tol,
                               dual_feasibility_tolerance=tol))
    if res.status == 2:
        return -np.inf
    if res.status != 0:
        return np.inf                                            # cannot prune
    return 0.0 if target is None else -res.fun + 1.0


def components(t):
    """Blocks joined by block-on-block contacts."""
    parent = list(range(t.n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i, j in t.contacts:
        if j >= 0:
            parent[find(i)] = find(j)
    groups = {}
    for k in range(t.n):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values())


def unbalanceable(t, tol=1e-9, max_boxes=200000):
    """Certify that NO placement of topology t is balanced. It is enough that one
    connected component cannot balance on its own (its blocks touch nothing
    outside it). A balanced component stays balanced when shifted left, and is
    unchanged by any shift that keeps all its table contacts whole, so its
    rightmost bottom block can be put at x = -1; a connected component then spans
    at most its size. Every box of positions whose relaxation is infeasible is
    discarded; the certificate holds if every box is."""
    out = []
    for comp in components(t):
        idx = {k: q for q, k in enumerate(comp)}
        levels = [t.levels[k] for k in comp]
        N = len(comp)
        cand = [(idx[i], -1 if j < 0 else idx[j], "sure") for i, j in t.contacts if i in idx]
        rows = []
        for q in range(N):
            for q2 in range(q + 1, N):
                if levels[q] == levels[q2]:
                    rows.append(([(q, 1), (q2, -1)], -1.0))
        for i, j in t.contacts:
            if i in idx and j >= 0:
                rows.append(([(idx[i], 1), (idx[j], -1)], 1.0))
                rows.append(([(idx[j], 1), (idx[i], -1)], 1.0))
        for a_, b_ in t.apart:
            if a_ in idx and b_ in idx:
                rows.append(([(idx[a_], 1), (idx[b_], -1)], -1.0))
        anchor = max(q for q in range(N) if levels[q] == 0)   # bottoms run left to right
        lo = [-1.0 - N] * N
        hi = [0.0 if levels[q] == 0 else N - 1.0 for q in range(N)]
        lo[anchor] = hi[anchor] = -1.0
        root = _relaxation_lp(levels, lo, hi, cand, rows, None, tol)
        stack = [(lo, hi)] if root > -np.inf else []
        boxes, ok = 0, True
        while stack:
            lo_, hi_ = stack.pop()
            boxes += 1
            w = np.array(hi_) - np.array(lo_)
            k = int(np.argmax(w))
            if w[k] < 1e-7 or boxes > max_boxes:
                ok = False
                break
            mid = 0.5 * (lo_[k] + hi_[k])
            for a_, b_ in ((lo_[k], mid), (mid, hi_[k])):
                l2, h2 = list(lo_), list(hi_)
                l2[k], h2[k] = a_, b_
                if _relaxation_lp(levels, l2, h2, cand, rows, None, tol) > -np.inf:
                    stack.append((l2, h2))
        out.append(dict(blocks=comp, certified=ok, boxes=boxes))
        if ok:
            return dict(key=t.key(), unbalanceable=True, component=comp, boxes=boxes,
                        components=out)
    return dict(key=t.key(), unbalanceable=False, components=out)


def certify(n, bound, eps=CERT_EPS, tol=1e-9, max_boxes=400000):
    """True if no connected balanced stack of n blocks extends past bound + eps.
    (A stack that is not connected is a smaller stack with the same overhang,
    certified at its own n.) Boxes: every x in [-n, n-1]; a connected stack
    spans at most n and touches the table at x <= 0."""
    t0 = time.time()
    boxes = 0
    thr = bound + eps
    for comp in compositions(n):
        levels = [l for l, c in enumerate(comp) for _ in range(c)]
        N = len(levels)
        firsts = np.cumsum((0,) + comp[:-1])
        targets = [int(firsts[l] + c - 1) for l, c in enumerate(comp)]
        lo0 = [-float(n)] * N
        hi0 = [0.0 if l == 0 else n - 1.0 for l in levels]
        for target in targets:
            u0 = _relax(levels, lo0, hi0, target, n, thr, tol)
            stack = [(u0, lo0, hi0)] if u0 > thr else []
            while stack:
                ub, lo, hi = stack.pop()
                boxes += 1
                if boxes > max_boxes:
                    return dict(n=n, certified=False, reason="box limit", boxes=boxes)
                widths = np.array(hi) - np.array(lo)
                k = int(np.argmax(widths))
                if widths[k] < 1e-7:
                    return dict(n=n, certified=False, reason="unresolved box",
                                box=[lo, hi], ub=ub, boxes=boxes)
                mid = 0.5 * (lo[k] + hi[k])
                for a_, b_ in ((lo[k], mid), (mid, hi[k])):
                    l2, h2 = list(lo), list(hi)
                    l2[k], h2[k] = a_, b_
                    u2 = _relax(levels, l2, h2, target, n, thr, tol)
                    if u2 > thr:
                        stack.append((u2, l2, h2))
    return dict(n=n, certified=True, bound=bound, eps=eps, tol=tol, boxes=boxes,
                seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------- the study
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_topologies(n, which=("a", "b"), processes=None):
    jobs = [(t.comp, t.rels, which) for t in topologies(n)]
    if processes == 1 or len(jobs) < 8:
        return [_topology_job(j) for j in jobs]
    from multiprocessing import Pool
    with Pool(processes) as pool:
        return pool.map(_topology_job, jobs, chunksize=1)


def summarise(n, rows):
    def best(formulation, only_one_on_one=False):
        top = None
        for r in rows:
            if only_one_on_one and any(c != 1 for c in r["comp"]):
                continue
            for target, v in r[formulation].items():
                if v and (top is None or v["value"] > top["value"]):
                    top = dict({k: w for k, w in v.items() if k != "z"},
                               key=r["key"], target=int(target))
        return top
    # A topology neither formulation could solve for any principal block must be
    # certified to have no balanced placement at all; one solved for some
    # principal blocks but not others is an optimiser failure and is listed.
    none, partial = {}, []
    for r in rows:
        if not r["realisable"]:
            continue
        vals = list(r["a"].values()) + list(r["b"].values())
        if not any(vals):
            none[r["key"]] = unbalanceable(Topology(*_parse_key(r["key"])))
        elif not all(vals):
            partial.append(r["key"])
    return dict(n=n, compositions=2 ** (n - 1), topologies=len(rows),
                no_solution=none, partial=partial,
                unrealisable=[r["key"] for r in rows if not r["realisable"]],
                closure_only=[r["key"] for r in rows if r["closure"] and not r["realisable"]],
                empty=[r["key"] for r in rows if not r["closure"]],
                realisable=sum(r["realisable"] for r in rows),
                solved_a=sum(1 for r in rows for v in r["a"].values() if v),
                solved_b=sum(1 for r in rows for v in r["b"].values() if v),
                runs=sum(len(r["a"]) for r in rows),
                a=best("a"), b=best("b"), one_on_one_a=best("a", True),
                one_on_one_b=best("b", True),
                per_topology={r["key"]: {t: (v["value"] if v else None)
                                         for t, v in r["a"].items()} for r in rows})


def verify_optimum(n, best):
    """The optimiser's stack, checked by the LPs (both formulations) at two
    tolerances, and one push of 1e-6 outward of every block."""
    t = Topology(*_parse_key(best["key"]))
    xs = best["xs"]
    out = dict(n=n, key=best["key"])
    for tol in LP_TOLERANCES:
        r = balance(t.levels, xs, tol=tol)
        out["balanced_tol_%g" % tol] = r["ok"]
        out["residual_tol_%g" % tol] = r.get("residual")
    out["balanced_b"] = balance_resultant(t.levels, xs)
    out["pushed"] = balance(t.levels, [x + PUSH for x in xs])["ok"]
    return out


def _parse_key(key):
    comp_s, rel_s = key.split("|")
    comp = tuple(int(c) for c in comp_s.split(","))
    rels = []
    if rel_s:
        for r in rel_s.split(";"):
            rels.append(tuple((int(r[k]), int(r[k + 1])) for k in range(0, len(r), 2)))
    return comp, tuple(rels)


def height_invariance(n, best):
    t = Topology(*_parse_key(best["key"]))
    per = {}
    for h in HEIGHTS:
        b = optimise_b(t, best["target"], height=h)
        per[str(h)] = dict(value=b["value"] if b else None,
                           balanced_at_a_optimum=balance_resultant(t.levels, best["xs"], height=h),
                           margin_at_a_optimum=edge_margin(t.levels, best["xs"], height=h))
    vals = [v["value"] for v in per.values() if v["value"] is not None]
    return dict(n=n, key=best["key"], heights=per,
                spread=(max(vals) - min(vals)) if len(vals) == len(HEIGHTS) else None)


def _margin_job(args):
    comp, rels, margin = args
    t = Topology(comp, rels)
    if not realisable(t):
        return None
    best = None
    for target in t.rightmost:
        v = optimise_a(t, target, margin=margin)
        if v and (best is None or v["value"] > best["value"]):
            best = dict({k: w for k, w in v.items() if k != "z"}, key=t.key(), target=target)
    return best


def margin_curve(n, processes=None):
    """The largest overhang when every resultant must stay `margin` inside its
    contact interval: the one-on-one class against every topology."""
    rows = []
    for m in MARGINS:
        jobs = [(t.comp, t.rels, m) for t in topologies(n)]
        if processes == 1:
            res = [_margin_job(j) for j in jobs]
        else:
            from multiprocessing import Pool
            with Pool(processes) as pool:
                res = pool.map(_margin_job, jobs, chunksize=1)
        allv = [r for r in res if r]
        one = [r for r in allv if all(c == 1 for c in _parse_key(r["key"])[0])]
        top = max(allv, key=lambda r: r["value"]) if allv else None
        top1 = max(one, key=lambda r: r["value"]) if one else None
        by_comp = {}
        for r in allv:
            c = _parse_key(r["key"])[0]
            k = ",".join(map(str, c))
            by_comp[k] = max(by_comp.get(k, -np.inf), r["value"])
        rows.append(dict(margin=m, unrestricted=top["value"] if top else None,
                         unrestricted_key=top["key"] if top else None,
                         one_on_one=top1["value"] if top1 else None,
                         by_composition=by_comp,
                         gain=(top["value"] / top1["value"] - 1) if top and top1 else None))
    return dict(n=n, rows=rows)


def shrunk_harmonic(n, eps):
    """The one-on-one stack with every offset (1/2 - eps)/k: overhang
    (1 - 2 eps) H_n / 2. Balanced with every resultant eps inside its contact."""
    xs = [sum((0.5 - eps) / k for k in range(n - l, n + 1)) - 1.0 for l in range(n)]
    return list(range(n)), xs


def three_block_margin_proof(margin_rows):
    """The argument that ANY positive contact margin eps (0 < eps < 1/2) removes
    the three-block gain, and its numerical check at every sampled margin.

    Three blocks have four level splits. [3]: every block on the table, overhang
    at most 1/2 - eps. [1,2]: each upper block rests on the bottom block alone,
    so its centre lies inside its contact, eps from the ends; two disjoint unit
    blocks on one unit block then need x_left + 1/2 >= x_b + eps and
    x_right + 1/2 <= x_b + 1 - eps with x_right >= x_left + 1, i.e.
    1 + 2 eps <= 1: no balanced placement. [2,1]: the right table block R carries
    its own weight at x_R + 1/2 and at most the whole top block's weight f <= 1
    at a point >= x_R + eps; its table resultant is <= -eps, so
    x_R + 1 <= 1 - eps - (1/2 + f eps)/(1 + f) <= 3/4 - 3 eps/2 (the fraction
    falls with f); the top block, if it rests on both table blocks, ends left of
    R's end, and on one of them it is a two-block one-on-one stack. [1,1,1]: the
    shrunk harmonic stack is balanced with margin eps and ends at
    (1 - 2 eps) 11/12, which exceeds 3/4 - 3 eps/2 for every eps < 1/2. So the
    largest three-block overhang at any margin 0 < eps < 1/2 belongs to the
    one-on-one class: gain zero."""
    rows = []
    for r in margin_rows:
        eps = r["margin"]
        if eps <= 0:
            continue
        lv, xs = shrunk_harmonic(3, eps)
        rows.append(dict(
            margin=eps,
            shrunk_harmonic=xs[-1] + 1.0,
            shrunk_harmonic_balanced=balance_resultant(lv, xs, margin=eps * (1 - 1e-9)),
            bound_2_1=0.75 - 1.5 * eps,
            found_2_1=r["by_composition"].get("2,1"),
            found_1_2=r["by_composition"].get("1,2"),
            found_3=r["by_composition"].get("3"),
            found_one_on_one=r["one_on_one"]))
    ok = all(q["shrunk_harmonic_balanced"]
             and (q["found_1_2"] is None)
             and (q["found_2_1"] is None or q["found_2_1"] <= q["bound_2_1"] + 1e-9)
             and (q["found_3"] is None or q["found_3"] <= 0.5 - q["margin"] + 1e-9)
             and q["shrunk_harmonic"] > q["bound_2_1"]
             and abs(q["found_one_on_one"] - q["shrunk_harmonic"]) < 1e-7
             for q in rows)
    return dict(rows=rows, consistent=ok, argument=three_block_margin_proof.__doc__)


def counterweight_three(xb=-0.5):
    """The three-block counterweight family: bottom block at xb, one block over
    each half of it. Balanced only with both upper blocks exactly half over."""
    return [0, 1, 1], [xb, xb - 0.5, xb + 0.5]


def classify(comp, xs, closed=False):
    """The topology key of a placed stack (blocks in level order, left to right);
    None if it is not a stack (a block with nothing under it, a bottom block off
    the table, blocks overlapping on a level). Open: contacts are strict
    overlaps, and a touching pair is skipped as belonging to two topologies.
    Closed: touching counts as contact, as it may carry force -- the topology
    with every touch in it must be enumerated too."""
    first = [int(v) for v in np.cumsum((0,) + tuple(comp[:-1]))]
    if any(xs[first[0] + q] > 0 or (not closed and xs[first[0] + q] == 0)
           for q in range(comp[0])):
        return None
    for l, c in enumerate(comp):
        for q in range(c - 1):
            gap = xs[first[l] + q + 1] - xs[first[l] + q]
            if gap < 1.0 or (not closed and gap == 1.0):
                return None
    rels = []
    for l in range(1, len(comp)):
        rl = []
        for q in range(comp[l]):
            xi = xs[first[l] + q]
            on = []
            for p_ in range(comp[l - 1]):
                d = abs(xi - xs[first[l - 1] + p_])
                if not closed and abs(d - 1.0) < 1e-12:
                    return None
                if d < 1.0 or (closed and d <= 1.0):
                    on.append(p_)
            if not on or on != list(range(on[0], on[-1] + 1)):
                return None
            rl.append((on[0], on[-1]))
        rels.append(tuple(rl))
    return Topology(comp, rels).key()


def coverage(n, samples=4000, seed=SEED, closed=False):
    """Random stacks, classified from their geometry, must all be enumerated.
    Closed: positions on a half-block grid, so blocks touch end to end and a
    block can sit over three."""
    rng = np.random.default_rng([seed, n, 7 if not closed else 8])
    keys = {t.key() for t in topologies(n) if realisable(t)}
    comps = list(compositions(n))
    sampled = missing = 0
    examples = []
    tries = 0
    while sampled < samples and tries < 200 * samples:
        tries += 1
        comp = comps[rng.integers(len(comps))]
        xs = []
        for c in comp:
            if closed:
                x = -0.5 * rng.integers(0, 2 * n + 1)
            else:
                x = -rng.random() * n
            for _ in range(c):
                xs.append(x)
                x += 1.0 + (0.5 * rng.integers(0, 2) if closed else rng.exponential(0.4))
        k = classify(comp, xs, closed=closed)
        if k is None:
            continue
        sampled += 1
        if k not in keys:
            missing += 1
            examples.append(k)
    return dict(n=n, sampled=sampled, missing=missing, examples=examples[:5])


def other_end_max(levels, xs, upper, lower):
    """Over every balancing force set, the largest force at the RIGHT end of the
    contact (upper on lower): zero means the whole contact force is forced to the
    left end of the interval."""
    cs = contacts_of(levels, xs)
    A, b = _system(levels, xs, cs)
    c = [k for k, (i, j, *_r) in enumerate(cs) if i == upper and j == lower][0]
    obj = np.zeros(A.shape[1])
    obj[2 * c + 1] = -1.0
    res = linprog(obj, A_eq=A, b_eq=b, bounds=[(0, None)] * A.shape[1], method="highs",
                  options=dict(primal_feasibility_tolerance=1e-10,
                               dual_feasibility_tolerance=1e-10))
    return dict(upper=upper, lower=lower, left_end=cs[c][2], lower_left_end=xs[lower],
                total=float(res.x[2 * c] + res.x[2 * c + 1]),
                other_end_max=float(-res.fun))


def forces_record(summary):
    states = {}
    four = summary["4"]["a"]
    t4 = Topology(*_parse_key(four["key"]))
    for name, (lv, xs) in dict(harmonic_4=harmonic_stack(4), harmonic_3=harmonic_stack(3),
                               unrestricted_4=(t4.levels, four["xs"]),
                               unrestricted_3=counterweight_three()).items():
        _f, resid, neg = display_forces(lv, xs)
        states[name] = dict(residual=resid, negative=neg)
    # [1,2,1]: 0 bottom, 1 rear, 2 front, 3 top
    assert t4.comp == (1, 2, 1)
    path = dict(rear_on_bottom=other_end_max(t4.levels, four["xs"], 1, 0),
                top_on_front=other_end_max(t4.levels, four["xs"], 3, 2))
    return dict(states=states, load_path=path,
                harmonic_top_left_end=harmonic_stack(4)[1][-1])


def main(processes=None):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    doc = dict(model=dict(source="Paterson & Zwick, Overhang, Amer. Math. Monthly 116 (2009); arXiv:0710.2357",
                          block_length=1, block_weight=1, table="x <= 0",
                          friction="none: all forces vertical",
                          starts=STARTS, seed=SEED, lp_tolerances=LP_TOLERANCES,
                          push=PUSH, certificate_eps=CERT_EPS, margins=MARGINS,
                          heights=HEIGHTS),
               registered={str(k): v[0] for k, v in REGISTERED.items()},
               harmonic=[harmonic_check(n) for n in N_VALIDATE])
    summary, verify, heights, certs = {}, {}, {}, {}
    for n in N_VALIDATE:
        rows = run_topologies(n, processes=processes)
        s = summarise(n, rows)
        summary[str(n)] = s
        verify[str(n)] = verify_optimum(n, s["a"])
        print("n=%d topologies %d realisable %d  A %.6f  B %.6f  one-on-one %.6f  %.0f s"
              % (n, s["topologies"], s["realisable"], s["a"]["value"], s["b"]["value"],
                 s["one_on_one_a"]["value"], time.time() - t0), flush=True)
    for n in N_CLAIM:
        heights[str(n)] = height_invariance(n, summary[str(n)]["a"])
    for n in CERT_N:
        certs[str(n)] = {}
        for tol in LP_TOLERANCES:
            bound = summary[str(n)]["a"]["value"]
            certs[str(n)]["%g" % tol] = certify(n, bound, tol=tol)
            print("certify n=%d tol %g: %s" % (n, tol, certs[str(n)]["%g" % tol]), flush=True)
    doc.update(enumeration=summary, verify=verify, height_invariance=heights,
               certificate=certs, forces=forces_record(summary),
               coverage={str(n): coverage(n) for n in N_VALIDATE},
               coverage_closed={str(n): coverage(n, closed=True) for n in N_VALIDATE})
    doc["margin"] = [margin_curve(n, processes) for n in N_CLAIM]
    doc["three_block_margin_proof"] = three_block_margin_proof(
        [m for m in doc["margin"] if m["n"] == 3][0]["rows"])
    doc["unbalanceable_controls"] = {
        k: unbalanceable(Topology(*_parse_key(k)), max_boxes=3000)["unbalanceable"]
        for k in ("1,2|0000", "1,2,1|0000;01", "1,1,1,1|00;00;00", "2,2|0101")}
    for m in doc["margin"]:
        for r in m["rows"]:
            print("margin n=%d  %.3f  unrestricted %s  one-on-one %s  gain %s"
                  % (m["n"], r["margin"], r["unrestricted"], r["one_on_one"], r["gain"]), flush=True)
    doc["seconds"] = round(time.time() - t0, 1)
    with io.open(os.path.join(OUT, "block_overhang.json"), "w", encoding="utf-8",
                 newline="\n") as fh:
        json.dump(clean(doc), fh, indent=1)
    print("wrote %s  %.1f s" % (os.path.join(OUT, "block_overhang.json"), time.time() - t0))
    return doc


if __name__ == "__main__":
    main()
