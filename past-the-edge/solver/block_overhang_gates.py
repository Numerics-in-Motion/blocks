# -*- coding: utf-8 -*-
"""How far past a table edge -- the gates.

    python src/block_overhang_gates.py    # reads block_overhang.json, writes gates.json

The ship/kill line is the design review's, registered
before any number was computed:

    SHIP only if every combinatorial contact topology for n = 1..6 is
    enumerated; the independent implementation reproduces all six values to
    their published precision; n = 3 and n = 4 have analytic or certified global
    upper bounds rather than multi-start evidence alone; and both exact gains
    exceed max(5 %, 3u), where u includes LP-tolerance variation, primal/dual
    residuals, independent-formulation disagreement and the outer optimiser's
    certified gap. Otherwise KILL; do not narrow the range after seeing the
    results.

  enumeration_gate   every candidate contact relation of n = 1..6 built; those
                     that exist with exactly their own contacts solved by both
                     formulations or certified to have no balanced placement;
                     random and grid stacks all land in one of them
  reproduction_gate  both formulations reproduce the registered values to the
                     precision they were registered with, and the one-on-one
                     class reproduces H_n/2
  harmonic_gate      the harmonic stacks balance, and do not 1e-6 further out
  verify_gate        every optimum is balanced by both LPs at both tolerances,
                     with equality residual <= 1e-9, and not 1e-6 further out
  certificate_gate   branch and bound certifies D(n) <= value + gap for
                     n = 1..4 at both LP tolerances (1, 2 close the argument
                     for stacks that fall apart into smaller ones)
  height_gate        the block height changes nothing (frictionless statics)
  gain_gate          n = 3 and 4: gain > max(5 %, 3u)
  margin_gate        the three-block caveat is on screen and spoken exactly when
                     the contact-margin sweep removes the three-block gain
  forces_gate        the drawn force sets meet the equations; the spoken load
                     path (rear block and top block pressing on inner ends) is
                     forced at the limit, not a choice of drawing
  wording_gate       the design review's forbidden words appear nowhere shipped
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import block_overhang as BO            # noqa: E402

GAIN_FLOOR = 0.05
RESIDUAL_LINE = 1e-9
HEIGHT_LINE = 1e-9
FORCED_LINE = 1e-7

TITLE = "HOW FAR PAST THE EDGE?"
ON_SCREEN = dict(
    setup="4 IDENTICAL BLOCKS AT A TABLE EDGE",
    one_template="ONE-ON-ONE BALANCE LIMIT: {v:.2f} BLOCK LENGTHS",
    free_template="UNRESTRICTED BALANCE LIMIT: {v:.2f} BLOCK LENGTHS",
    gain_template="MAXIMUM BALANCED OVERHANG: +{pct:d}%",
    moves="REPOSITIONED THROUGH BALANCED STATES",
    none="NO BALANCING FORCE SET EXISTS",
    forces="ONE BALANCING SET OF CONTACT FORCES",
    model="IDEAL 2-D MODEL: RIGID, IDENTICAL, HOMOGENEOUS, FRICTIONLESS BLOCKS",
    limits="BALANCE LIMITS: ZERO CONTACT MARGIN",
    three_template="3 BLOCKS: +{pct:d}% ONLY AT ZERO CONTACT MARGIN",
)
SILENT = dict(
    gain_template="+{pct:d}%",
    result_template="4 blocks: {one:.2f} to {free:.2f} block lengths, +{pct:d}%",
    three_template="3 blocks: +{pct:d}% only at zero contact margin",
    limits=["ideal 2-D model: rigid, identical, homogeneous, frictionless blocks",
            "balance limits: zero contact margin"],
)
NARRATION = dict(
    a1="Four identical blocks at a table edge.",
    b1="Under the one-on-one rule, the maximum balanced overhang is one point zero four block lengths.",
    b2="At that limit, the top block is entirely past the table.",
    b3="Beyond this one-on-one limit, no balancing force set exists.",
    c1="Now allow blocks side by side, so weight can sit behind the edge.",
    c2="For the same four blocks, the unrestricted balance limit is one point one seven.",
    d1="The rear block presses down on the bottom block's inner end, while the top block presses down on the front block's inner end.",
    e1="Twelve percent farther, in an ideal model of rigid, identical, homogeneous, frictionless blocks.",
    f1="With three blocks, the nine-percent gain requires exact placement: any positive contact margin removes it.",
    z1="Both shown values are balance limits with zero contact margin.",
)
# the spoken numbers, checked against the data
SPOKEN = dict(b1=1.04, c2=1.17, e1=12, f1=9)
FORBIDDEN = [r"\bstab", r"\bsafe", r"will not fall", r"won.?t fall", r"real[- ]world",
             r"friction (does not|doesn.?t) matter", r"\bwood", r"\balways\b",
             r"\bnever\b", r"best for every", r"\bbest\b", r"\bmasonry\b", r"\bcorbel",
             r"\bholds? down\b", r"\breach", r"\bprove", r"\bproof\b", r"\bguarantee",
             r"\bcollaps", r"\btopple", r"\btip(s|ping)? over\b"]


def load(path=None):
    return json.load(io.open(path or os.path.join(BO.OUT, "block_overhang.json"),
                             encoding="utf-8"))


def H(n):
    return Fraction(sum(Fraction(1, 2 * k) for k in range(1, n + 1)))


def enumeration_gate(doc):
    rows, bad = {}, []
    for n in BO.N_VALIDATE:
        s = doc["enumeration"][str(n)]
        unsolved = [k for k, v in s["per_topology"].items()
                    if not any(x is not None for x in v.values())]
        unrealisable = s["topologies"] - s["realisable"]      # the LP proves these empty
        uncertified = [k for k in unsolved
                       if k not in s["unrealisable"]
                       and not s["no_solution"].get(k, {}).get("unbalanceable")]
        rows[n] = dict(topologies=s["topologies"], realisable=s["realisable"],
                       runs=s["runs"], solved_a=s["solved_a"], solved_b=s["solved_b"],
                       unrealisable=unrealisable, closure_only=len(s["closure_only"]),
                       empty=len(s["empty"]),
                       no_balanced_placement=len(s["no_solution"]),
                       unsolved_not_certified=uncertified, partial=s["partial"])
        if uncertified or s["partial"]:
            bad.append(n)
    cov = list(doc["coverage"].values()) + list(doc["coverage_closed"].values())
    missing = sum(c["missing"] for c in cov)
    sampled = sum(c["sampled"] for c in cov)
    controls = doc["unbalanceable_controls"]            # balanceable: must not certify
    return dict(name="enumeration", per_n=rows, coverage_sampled=sampled,
                coverage_missing=missing, certificate_controls=controls, bad=bad,
                passes=bool(not bad and missing == 0 and sampled > 0
                            and not any(controls.values())))


def reproduction_gate(doc):
    rows, bad = {}, []
    for n in BO.N_VALIDATE:
        s = doc["enumeration"][str(n)]
        txt, dec = BO.REGISTERED[n]
        tol = 0.5 * 10 ** (-dec) + 1e-12
        ref = float(txt)
        h = float(H(n))
        row = dict(registered=txt, a=s["a"]["value"], b=s["b"]["value"],
                   one_on_one=float(h), one_a=s["one_on_one_a"]["value"],
                   one_b=s["one_on_one_b"]["value"], tolerance=tol)
        row["ok"] = bool(abs(row["a"] - ref) <= tol and abs(row["b"] - ref) <= tol
                         and abs(row["one_a"] - h) <= 5e-10 and abs(row["one_b"] - h) <= 5e-10)
        rows[n] = row
        if not row["ok"]:
            bad.append(n)
    four = doc["enumeration"]["4"]["a"]["value"]
    closed = (15 - 4 * math.sqrt(2)) / 8
    return dict(name="reproduction", per_n=rows, four_closed_form=closed,
                four_minus_closed_form=four - closed, bad=bad,
                passes=bool(not bad and abs(four - closed) < 1e-9))


def harmonic_gate(doc):
    bad = [h["n"] for h in doc["harmonic"]
           if not (h["balanced"] and h["balanced_b"]) or h["top_pushed"] or h["all_pushed"]]
    return dict(name="harmonic", bad=bad, passes=not bad)


def verify_gate(doc):
    bad = []
    worst = 0.0
    for n, v in doc["verify"].items():
        res = [v["residual_tol_%g" % t] for t in BO.LP_TOLERANCES]
        worst = max([worst] + [r for r in res if r is not None])
        if not all(v["balanced_tol_%g" % t] for t in BO.LP_TOLERANCES) or not v["balanced_b"] \
                or v["pushed"] or any(r is None or r > RESIDUAL_LINE for r in res):
            bad.append(n)
    return dict(name="verify", worst_residual=worst, bad=bad, passes=not bad)


def certificate_gate(doc):
    rows, bad = {}, []
    for n in BO.CERT_N:
        c = doc["certificate"][str(n)]
        ok = all(c["%g" % t]["certified"] for t in BO.LP_TOLERANCES)
        bounds = [c["%g" % t].get("bound") for t in BO.LP_TOLERANCES]
        rows[n] = dict(certified=ok, bound=bounds[0], gap=BO.CERT_EPS,
                       boxes=[c["%g" % t].get("boxes") for t in BO.LP_TOLERANCES])
        if not ok:
            bad.append(n)
    return dict(name="certificate", per_n=rows, bad=bad, passes=not bad)


def height_gate(doc):
    rows, bad = {}, []
    for n, h in doc["height_invariance"].items():
        balanced = all(v["balanced_at_a_optimum"] for v in h["heights"].values())
        rows[n] = dict(spread=h["spread"], balanced_every_height=balanced)
        if h["spread"] is None or h["spread"] > HEIGHT_LINE or not balanced:
            bad.append(n)
    return dict(name="height", per_n=rows, bad=bad, passes=not bad)


def uncertainty(doc, n):
    s = doc["enumeration"][str(n)]
    v = doc["verify"][str(n)]
    c = doc["certificate"][str(n)]
    bounds = [c["%g" % t]["bound"] for t in BO.LP_TOLERANCES]
    parts = dict(
        lp_tolerance=abs(bounds[0] - bounds[1]) + BO.PUSH,
        residuals=max([s["a"]["violation"], s["b"]["violation"]]
                      + [v["residual_tol_%g" % t] for t in BO.LP_TOLERANCES]),
        formulations=abs(s["a"]["value"] - s["b"]["value"]),
        certificate_gap=BO.CERT_EPS)
    return sum(parts.values()), parts


def gain_gate(doc):
    rows, bad = {}, []
    for n in BO.N_CLAIM:
        h = float(H(n))
        d = doc["enumeration"][str(n)]["a"]["value"]
        u, parts = uncertainty(doc, n)
        gain = d / h - 1.0
        line = max(GAIN_FLOOR, 3.0 * u / h)
        rows[n] = dict(one_on_one=h, unrestricted=d, gain=gain,
                       gain_upper=(d + BO.CERT_EPS) / h - 1.0, u=u, parts=parts,
                       line=line, ok=bool(gain > line))
        if not rows[n]["ok"]:
            bad.append(n)
    return dict(name="gain", per_n=rows, bad=bad, passes=not bad)


def margin_gate(doc, on, narration):
    """"Any positive contact margin removes it" is universal, so a sampled sweep
    cannot carry it alone: the words need the three-block argument in
    `three_block_margin_proof` AND its numerical check at every sampled margin,
    down to 1e-6."""
    m = {r["n"]: r["rows"] for r in doc["margin"]}
    positive = [r for r in m[3] if r["margin"] > 0]
    three_vanishes = all(r["gain"] <= 1e-9 for r in positive)
    proof = doc["three_block_margin_proof"]["consistent"]
    four_holds = all(r["gain"] >= m[4][0]["gain"] - 1e-6 for r in m[4])
    says_three = ("ZERO CONTACT MARGIN" in on["three"]
                  and "any positive contact margin removes it" in narration["f1"])
    return dict(name="margin", three_gain=[(r["margin"], r["gain"]) for r in positive],
                three_vanishes=three_vanishes, proof_consistent=proof,
                four_gain_holds_to=max(r["margin"] for r in m[4]),
                four_gain_holds=four_holds, caveat_on_screen_and_spoken=says_three,
                passes=bool(three_vanishes and proof and says_three and four_holds))


def forces_gate(doc):
    f = doc["forces"]
    bad = [k for k, v in f["states"].items()
           if v["residual"] > RESIDUAL_LINE or v["negative"] > RESIDUAL_LINE]
    forced = f["load_path"]
    ok_path = all(v["other_end_max"] <= FORCED_LINE for v in forced.values())
    return dict(name="forces", states=f["states"], load_path=forced,
                top_block_left_end=f["harmonic_top_left_end"],
                bad=bad, passes=bool(not bad and ok_path
                                     and f["harmonic_top_left_end"] > 0))


def spoken_gate(values):
    """The numbers in the words are the numbers in the data, as rounded."""
    one, free, g4, g3 = values
    checks = dict(b1=(round(one, 2), SPOKEN["b1"]), c2=(round(free, 2), SPOKEN["c2"]),
                  e1=(int(round(100 * g4)), SPOKEN["e1"]), f1=(int(round(100 * g3)), SPOKEN["f1"]))
    bad = {k: v for k, v in checks.items() if abs(v[0] - v[1]) > 1e-9}
    return dict(name="spoken", checks=checks, bad=bad, passes=not bad)


def hits(text):
    return [(m.group(0), text[max(0, m.start() - 30):m.end() + 30])
            for pat in FORBIDDEN for m in re.finditer(pat, text, re.I)]


def wording_gate(texts):
    found = [h for t in texts for h in hits(t)]
    return dict(name="wording", hits=found, passes=not found)


def run(write=True):
    doc = load()
    one = float(H(4))
    free = doc["enumeration"]["4"]["a"]["value"]
    g4 = free / one - 1.0
    g3 = doc["enumeration"]["3"]["a"]["value"] / float(H(3)) - 1.0
    on = {k: v for k, v in ON_SCREEN.items() if not k.endswith("_template")}
    on["one"] = ON_SCREEN["one_template"].format(v=one)
    on["free"] = ON_SCREEN["free_template"].format(v=free)
    on["gain"] = ON_SCREEN["gain_template"].format(pct=int(round(100 * g4)))
    on["three"] = ON_SCREEN["three_template"].format(pct=int(round(100 * g3)))
    silent = dict(gain=SILENT["gain_template"].format(pct=int(round(100 * g4))),
                  result=SILENT["result_template"].format(one=one, free=free,
                                                          pct=int(round(100 * g4))),
                  three=SILENT["three_template"].format(pct=int(round(100 * g3))),
                  limits=SILENT["limits"])
    claim = ("In an ideal 2-D model of identical, rigid, homogeneous, frictionless "
             "blocks, counterweights raise the maximum balanced overhang: for four "
             "blocks the balance limit is %.4f block lengths, against %.4f under the "
             "one-on-one rule (+%.1f%%); for three blocks it is %.4f against 11/12 "
             "(+%.1f%%), and any positive contact margin removes that three-block gain. "
             "Both are balance limits with zero margin."
             % (free, one, 100 * g4, doc["enumeration"]["3"]["a"]["value"], 100 * g3))
    texts = [TITLE, claim] + list(on.values()) + list(NARRATION.values()) + \
        [silent["gain"], silent["result"], silent["three"]] + silent["limits"]
    gates = [enumeration_gate(doc), reproduction_gate(doc), harmonic_gate(doc),
             verify_gate(doc), certificate_gate(doc), height_gate(doc), gain_gate(doc),
             margin_gate(doc, on, NARRATION), forces_gate(doc),
             spoken_gate((one, free, g4, g3)), wording_gate(texts)]
    out = dict(title=TITLE, claim=claim, on_screen=on, silent=silent,
               narration=NARRATION, values=dict(one_on_one_4=one, unrestricted_4=free,
                                                gain_4=g4, gain_3=g3,
                                                one_on_one_3=float(H(3)),
                                                unrestricted_3=doc["enumeration"]["3"]["a"]["value"]),
               stacks=dict(four=doc["enumeration"]["4"]["a"],
                           three=doc["enumeration"]["3"]["a"]),
               gates=gates, passed=sum(bool(g["passes"]) for g in gates), total=len(gates))
    if write:
        with io.open(os.path.join(BO.OUT, "gates.json"), "w", encoding="utf-8",
                     newline="\n") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    return out


def main():
    out = run()
    for g in out["gates"]:
        extra = {k: v for k, v in g.items() if k not in ("name", "passes")}
        print("%-12s %s  %s" % (g["name"], "PASS" if g["passes"] else "FAIL",
                                json.dumps(extra, default=str)[:260]))
    print("%d / %d" % (out["passed"], out["total"]))
    print("claim:", out["claim"])
    return 0 if out["passed"] == out["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
