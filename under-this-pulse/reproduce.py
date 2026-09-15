"""Under this pulse, the bigger block stays standing.

    python reproduce.py            # the whole registered run and every gate
    python reproduce.py --quick    # the five sizes at A = 2 and one control (a minute or two)

WHAT IS CLAIMED

In Housner's ideal rigid rocking model -- a planar rectangular block of slenderness
0.2 rad on rigid ground, no sliding or bouncing, impact factor 1 - 1.5 sin^2(alpha) --
under one sine pulse of ground acceleration peaking at 0.405 g and lasting 0.5 s, the
0.98 m tall block overturned and the geometrically similar 3.92 m tall block did not.
Across the five registered sizes (0.49 to 7.84 m tall) the two smallest overturned.

WHAT IS NOT

Anything about a real object or a recorded ground motion. Sliding, bouncing, soil,
vertical shaking and 3-D motion are outside the model. The drawn motion of an
overturned block stops at first side contact; that impact is not modelled. How long
a standing block takes to come to rest is not computed.

WHAT THIS DOES

Checks both modules against the SHA-256 frozen with the reference, re-runs the
registered study (every size on the amplitude scan in four implementations, the
bisected transitions, the restitution sensitivity, the rectangular-pulse control
and the similarity control), re-applies every gate, and fails if a gated outcome
or number has moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "solver"))
RUN = tempfile.mkdtemp(prefix="rocking-")
os.environ["ROCK_DATA"] = RUN

import rocking_block as RB                 # noqa: E402
import rocking_block_gates as G            # noqa: E402

REF = os.path.join(HERE, "reference", "reference_canonical.json")
E_TOL = 1e-8          # E/E_b at the end of the pulse, across platforms
A_TOL = 2e-5          # a transition amplitude: two bisection widths


def sha(path):
    """SHA-256 of the file with line endings normalised to LF, so the check means
    the same on a Linux clone, a Windows checkout that converts to CRLF, and the
    tree it was frozen from."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def check_sources(ref):
    bad = [k for k, want in ref["source_sha256"].items()
           if sha(os.path.join(HERE, "solver", k + ".py")) != want]
    if bad:
        raise SystemExit("these do not match the reference: %s" % ", ".join(bad))
    print("   %d modules match the reference by SHA-256" % len(ref["source_sha256"]))


def quick(ref):
    worst = 0.0
    ok = True
    for R in RB.SIZES:
        want = ref["headline_A2"]["%.2f" % R]
        outs = []
        for impl in RB.IMPLS:
            r = RB.simulate(RB.Case(R=R, A=RB.A_HEAD), impl)
            outs.append(r["outcome"])
            if impl == "P":
                worst = max(worst, abs(r["e_end"] - want["e_end"]))
                e = r["e_end"]
        same = len(set(outs)) == 1 and outs[0] == want["outcome"]
        ok = ok and same
        print("   R %.2f m (%.2f m tall): %s in all four implementations, E/E_b at pulse end %.6f"
              % (R, 2 * R * RB.math.cos(RB.ALPHA), "OVERTURNED" if outs[0] == "O" else "STANDING", e))
    rc = RB.job_rect((1.0, "P"))
    rel = max(abs(rc["lo"] - rc["closed"]), abs(rc["hi"] - rc["closed"])) / rc["closed"]
    print("   rectangular pulse, p t_d = 1: threshold in [%.6f, %.6f], Housner %.6f (%.1e relative)"
          % (rc["lo"], rc["hi"], rc["closed"], rel))
    print("   E/E_b reproduces the reference to %.1e" % worst)
    return 0 if (ok and worst <= E_TOL and rel <= RB.RECT_TOL) else 1


def full(ref):
    RB.main()
    res = G.run(write=True)
    print()
    for g in res["gates"]:
        print("   %-12s %s" % (g["name"], "PASS" if g["passes"] else "FAIL"))
    doc = G.load()
    moved = []
    for A, want in ref["outcomes_A"].items():
        got = [G.table_outcome(doc, R, float(A)) for R in RB.SIZES]
        if got != want:
            moved.append(("outcomes at A %s" % A, got, want))
    got_t = sorted((t["R"], t["A_T"]) for t in doc["transitions"])
    want_t = sorted((t["R"], t["A_T"]) for t in ref["transitions"])
    if len(got_t) != len(want_t) or any(g[0] != w[0] or abs(g[1] - w[1]) > A_TOL for g, w in zip(got_t, want_t)):
        moved.append(("transitions", got_t, want_t))
    by = G.group(doc["scan"])
    for R in RB.SIZES:
        e = by[(R, RB.A_HEAD, 1.0, "sine", RB.TP)]["P"]["e_end"]
        if abs(e - ref["headline_A2"]["%.2f" % R]["e_end"]) > E_TOL:
            moved.append(("E/E_b at R %.2f" % R, e, ref["headline_A2"]["%.2f" % R]["e_end"]))
    print()
    for R, A_T in got_t:
        print("   R %.2f m: STANDING below A = %.5f, OVERTURNED above it on the scan" % (R, A_T))
    if moved:
        raise SystemExit("the answer moved against the frozen reference: %r" % moved)
    print("   the gated outcomes, transitions and energies reproduce the reference")
    print()
    print("   claim: %s" % res["claim"])
    return 0 if res["passed"] == res["total"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print("UNDER THIS PULSE, THE BIGGER BLOCK STAYS STANDING")
    ref = json.load(open(REF, encoding="utf-8"))
    check_sources(ref)
    return quick(ref) if a.quick else full(ref)


if __name__ == "__main__":
    raise SystemExit(main())
