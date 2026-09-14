"""How far past a table edge can identical blocks go?

    python reproduce.py            # the whole study and every gate (minutes)
    python reproduce.py --quick    # three and four blocks only (about a minute)

WHAT IS CLAIMED

In an ideal 2-D model of identical, rigid, homogeneous, frictionless blocks,
counterweights raise the maximum balanced overhang: for four blocks the balance
limit is 1.1679 block lengths, against 1.0417 under the one-on-one rule
(+12.1%); for three blocks it is 1.0000 against 11/12 (+9.1%), and any positive
contact margin removes that three-block gain. Both are balance limits with zero
margin.

WHAT IS NOT

Anything beyond balance: a margin, friction, sliding or tipping. Rounded edges, uneven weights,
deformation, anything in 3-D. Real blocks, or more than six blocks.

WHAT THIS DOES

Checks both modules against the SHA-256 frozen with the reference, re-runs the
study (every candidate contact pattern of one to six blocks, each solved by two
formulations or certified to have no balanced placement; the upper-bound
certificates; the height and contact-margin sweeps), re-applies every gate,
and fails if a gated number has moved.
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
RUN = tempfile.mkdtemp(prefix="overhang-")
os.environ["OVERHANG_DATA"] = RUN

import block_overhang as BO                # noqa: E402
import block_overhang_gates as G           # noqa: E402

REF = os.path.join(HERE, "reference", "reference_canonical.json")
TOL = 1e-8


def sha(path):
    """SHA-256 of the file with line endings normalised to LF, so the check
    means the same on a Linux clone, a Windows checkout that converts to
    CRLF, and the tree it was frozen from."""
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
    for n in (3, 4):
        s = BO.summarise(n, BO.run_topologies(n))
        c = BO.certify(n, s["a"]["value"])
        h = float(G.H(n))
        print("   %d blocks: one on one %.6f, unrestricted %.6f (formulation B %.6f), "
              "+%.2f%%, certified to +%.3g: %s"
              % (n, s["one_on_one_a"]["value"], s["a"]["value"], s["b"]["value"],
                 100 * (s["a"]["value"] / h - 1), BO.CERT_EPS, c["certified"]))
        worst = max(worst, abs(s["a"]["value"] - ref["maxima"][str(n)]["a"]))
        if not c["certified"]:
            return 1
    print("   reproduce the reference to %.1e" % worst)
    return 0 if worst <= TOL else 1


def full(ref):
    BO.main()
    res = G.run(write=True)
    print()
    for g in res["gates"]:
        print("   %-12s %s" % (g["name"], "PASS" if g["passes"] else "FAIL"))
    doc = G.load()
    worst = 0.0
    for n, m in ref["maxima"].items():
        s = doc["enumeration"][n]
        worst = max(worst, abs(s["a"]["value"] - m["a"]), abs(s["b"]["value"] - m["b"]),
                    abs(s["one_on_one_a"]["value"] - m["one_on_one"]))
        print("   %s blocks: %3d topologies, one on one %.6f, unrestricted %.6f"
              % (n, s["topologies"], s["one_on_one_a"]["value"], s["a"]["value"]))
    print()
    print("   the gated numbers reproduce the reference to %.1e" % worst)
    if worst > TOL:
        raise SystemExit("the answer moved against the frozen reference")
    print()
    print("   claim: %s" % res["claim"])
    return 0 if res["passed"] == res["total"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print("HOW FAR PAST THE EDGE?")
    ref = json.load(open(REF, encoding="utf-8"))
    check_sources(ref)
    return quick(ref) if a.quick else full(ref)


if __name__ == "__main__":
    raise SystemExit(main())
