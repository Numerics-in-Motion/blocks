# -*- coding: utf-8 -*-
"""The registered ship/kill line (design review, locked before the production run).

    python rocking_block_gates.py          # reads $ROCK_DATA/rocking_block.json, writes gates.json

Items 1-8 are physics and numerics; item 9 is wording. Failure of items 1-8 kills the
video. Every gate reads every registered cell, not only the headline.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocking_block as RB                                   # noqa: E402

DATA = RB.OUT
EPS = RB.EPS


def load(path=None):
    with open(path or os.path.join(DATA, "rocking_block.json"), encoding="utf-8") as fh:
        return json.load(fh)


def u_of(qP, qT, qE, qh, qh2):
    return abs(qP - qT) + abs(qT - qE) + abs(qh - qh2) + 64 * EPS * max(1.0, abs(qT))


def all_solves(doc):
    """Every forced-phase solve that carries impact records: scan, restitution."""
    return list(doc["scan"]) + list(doc["restitution"])


def group(solves):
    out = {}
    for r in solves:
        c = r["case"]
        out.setdefault((c["R"], c["A"], c["cv_factor"], c["pulse"], c["Tp"]), {})[r["impl"]] = r
    return out


def table_outcome(doc, R, A):
    for row in doc["table"]:
        if abs(row["R"] - R) < 1e-12 and abs(row["A"] - A) < 1e-12:
            return row["outcome"]
    return None


# 1. anchors ---------------------------------------------------------------------

def anchor_gate(doc):
    ups = [r["uplift_err"] for r in all_solves(doc) if r["uplift_err"] is not None]
    uplift = dict(max_err=max(ups), n=len(ups), passes=max(ups) <= RB.EVENT_BRACKET)
    solves = all_solves(doc)
    ratio = max(max(r["ratio_err"], r["energy_err"],
                    (r.get("post_audit") or {}).get("ratio_err", 0.0),
                    (r.get("post_audit") or {}).get("energy_err", 0.0)) for r in solves)
    impact = dict(max_ratio_or_energy_err=ratio, passes=ratio <= RB.IMPACT_RATIO_TOL)
    rect = []
    for r in doc["rect"]:
        ok = r["lo"] is not None and r["o_lo"] == "S" and r["o_hi"] == "O"
        rel = max(abs(r["lo"] - r["closed"]), abs(r["hi"] - r["closed"])) / r["closed"] if ok else math.inf
        rect.append(dict(ptd=r["ptd"], impl=r["impl"], bracket=[r["lo"], r["hi"]], closed=r["closed"],
                         rel_err=rel, passes=bool(ok and rel <= RB.RECT_TOL)))
    sim = similarity_gate(doc)
    passes = uplift["passes"] and impact["passes"] and all(x["passes"] for x in rect) and sim["passes"]
    return dict(name="anchors", uplift=uplift, impact=impact, rect=rect, similarity=sim["pairs"],
                passes=bool(passes))


# 2. audits ------------------------------------------------------------------------

def audit_gate(doc):
    fails = []
    diag = dict(e_e2_max=(0.0, None))
    for key, by in group(all_solves(doc)).items():
        missing = [i for i in RB.IMPLS if i not in by]
        if missing:
            fails.append((key, "missing %s" % missing))
            continue
        for i in ("E", "E2"):
            r = by[i]
            if r["impact_residual"] > RB.ROOT_RESIDUAL:
                fails.append((key, i, "impact root residual", r["impact_residual"]))
            if r["over_residual"] is not None and r["over_residual"] > RB.ROOT_RESIDUAL:
                fails.append((key, i, "overturn root residual", r["over_residual"]))
            if r["impact_bracket"] > RB.EVENT_BRACKET:
                fails.append((key, i, "bracket", r["impact_bracket"]))
            pa = r.get("post_audit") or {}
            if pa.get("residual", 0.0) > RB.ROOT_RESIDUAL:
                fails.append((key, i, "post-pulse root residual", pa["residual"]))
            if pa.get("over_residual") is not None and pa["over_residual"] > RB.ROOT_RESIDUAL:
                fails.append((key, i, "post-pulse overturn residual", pa["over_residual"]))
        e, e2 = by["E"], by["E2"]
        label = table_outcome(doc, key[0], key[1]) if key[2] == 1.0 else None
        if e["outcome"] != e2["outcome"] and label == "UNRESOLVED":
            continue                      # an unresolved cell may split; it is labelled, not used
        if e["outcome"] != e2["outcome"] or e["impacts"] != e2["impacts"]:
            fails.append((key, "E vs E2 outcome/impacts", e["outcome"], e2["outcome"], e["impacts"], e2["impacts"]))
        elif e["impact_taus"]:
            # DIAGNOSTIC, not a gate (result review): the registration bounds each search's
            # bracket and residual, and asks for the half-spacing repeat; it gives no
            # tolerance to the difference between the two repeats' event times.
            d = max(abs(a - b) for a, b in zip(e["impact_taus"], e2["impact_taus"]))
            if d > diag["e_e2_max"][0]:
                diag["e_e2_max"] = (d, key)
    bis = [b for b in doc["bisections"]]
    for b in bis:
        if b["lost"] or b["width"] > RB.BISECT_WIDTH or b["o_lo"] == b["o_hi"]:
            fails.append(("bisection", b["key"]["R"], b["impl"], b["start"], b["lo"], b["hi"]))
    unresolved = [(row["R"], row["A"], row["impl"]) for row in doc["table"] if row["outcome"] == "UNRESOLVED"]
    unmatched = [t for t in doc["transitions"] if not t["matched"]]
    # the rest cutoff V_REST: where it fired, impact counts depend on it (the result review
    # checked 1e-10 / 1e-12 / 1e-14: no outcome changed, the counts did)
    rest = [r for r in all_solves(doc) if r["rests"] > 0]
    rest_cells = sorted(set((r["case"]["R"], r["case"]["A"], r["case"]["cv_factor"]) for r in rest))
    diagnostics = dict(
        e_vs_e2_event_time_max=diag["e_e2_max"][0], e_vs_e2_event_time_cell=diag["e_e2_max"][1],
        v_rest_triggers=sum(r["rests"] for r in rest), v_rest_records=len(rest), v_rest_cells=rest_cells,
        impact_counts_cutoff_dependent_in=rest_cells)
    return dict(name="audits", failures=fails[:50], n_failures=len(fails), n_solves=len(all_solves(doc)),
                n_bisections=len(bis), unresolved_cells=unresolved, unmatched_transitions=unmatched,
                diagnostics=diagnostics, passes=not fails)


# 3. impacts -----------------------------------------------------------------------

def impact_gate(doc):
    worst = dict(residual=0.0, ratio=0.0, chronological=True)
    for r in all_solves(doc):
        pa = r.get("post_audit") or {}
        worst["residual"] = max(worst["residual"], r["impact_residual"], pa.get("residual", 0.0))
        worst["ratio"] = max(worst["ratio"], r["ratio_err"], r["energy_err"], pa.get("ratio_err", 0.0),
                             pa.get("energy_err", 0.0))
        worst["chronological"] = worst["chronological"] and r["chronological"] and pa.get("chronological", True)
    passes = worst["residual"] <= RB.IMPACT_RESIDUAL_TOL and worst["ratio"] <= RB.IMPACT_RATIO_TOL \
        and worst["chronological"]
    return dict(name="impacts", worst=worst, passes=bool(passes))


# 4. headline ------------------------------------------------------------------------

def headline_gate(doc):
    rows = {}
    ok = True
    for A in (RB.A_HEAD,) + RB.A_NEIGHBOURS:
        got = tuple(table_outcome(doc, R, A) for R in RB.SIZES)
        rows["%.2f" % A] = got
        ok = ok and got == RB.EXPECTED
    clear = []
    for R in RB.SIZES:
        ts = [t for t in doc["transitions"] if abs(t["R"] - R) < 1e-12]
        best = None
        for t in ts:
            if t["matched"]:
                d, need = abs(t["A_T"] - RB.A_HEAD), RB.NEIGHBOUR_CLEARANCE + 3 * t["u_A"]
            else:
                d, need = min(abs(g - RB.A_HEAD) for g in t["grid"]), RB.NEIGHBOUR_CLEARANCE
            if best is None or d < best[0]:
                best = (d, need, t["grid"], t.get("A_T"), t.get("u_A"))
        clear.append(dict(R=R, nearest=(None if best is None else dict(distance=best[0], needed=best[1],
                                                                      grid=best[2], A_T=best[3], u_A=best[4])),
                          passes=best is None or best[0] >= best[1]))
    passes = ok and all(c["passes"] for c in clear)
    return dict(name="headline", outcomes=rows, expected=RB.EXPECTED, clearance=clear, passes=bool(passes))


# 5. restitution ---------------------------------------------------------------------

def restitution_gate(doc):
    rows = []
    g = group(doc["restitution"])
    ok = True
    for f in RB.CV_FACTORS:
        row = dict(cv_factor=f)
        for R in RB.SIZES:
            by = g[(R, RB.A_HEAD, f, "sine", RB.TP)]
            outs = sorted(set(by[i]["outcome"] for i in RB.IMPLS))
            row["R%.2f" % R] = outs[0] if len(outs) == 1 else "UNRESOLVED"
        rows.append(row)
        ok = ok and row["R%.2f" % RB.HEAD_PAIR[0]] == "O" and row["R%.2f" % RB.HEAD_PAIR[1]] == "S"
    return dict(name="restitution", rows=rows, passes=bool(ok))


# 6. contact -------------------------------------------------------------------------

def contact_of(by):
    """N_min and max |H|/N over the forced phase and the post-pulse motion, with u."""
    def q(r, which, tag):
        # the forced phase, the exact post-pulse energy curve, and (claimed cells) the
        # integrated post-pulse motion as well -- the worse of them
        vals = [r["%s_%s" % (which, tag)]] + [r[k] for k in ("post_x_%s_%s" % (which, tag),
                                                               "post_%s_%s" % (which, tag)) if r.get(k) is not None]
        return min(vals) if which == "Nmin" else max(vals)
    out = {}
    for which in ("Nmin", "mu"):
        P, T, E = (q(by[i], which, "h") for i in ("P", "T", "E"))
        h, h2 = q(by["P"], which, "h"), q(by["P"], which, "h2")
        out[which] = P
        out["u_" + which] = u_of(P, T, E, h, h2)
    out["applicable"] = bool(out["Nmin"] - 3 * out["u_Nmin"] >= RB.N_MIN and out["mu"] + 3 * out["u_mu"] <= RB.MU_MAX)
    return out


def contact_gate(doc):
    g = group(doc["scan"])
    claimed = []
    for R in RB.SIZES:
        c = contact_of(g[(R, RB.A_HEAD, 1.0, "sine", RB.TP)])
        c["R"] = R
        claimed.append(c)
    labels = {}
    for key, by in g.items():
        if all(i in by for i in ("P", "T", "E")):
            labels["R%.2f_A%.2f" % (key[0], key[1])] = contact_of(by)["applicable"]
    for key, by in group(doc["restitution"]).items():
        labels["R%.2f_A%.2f_cv%.2f" % (key[0], key[1], key[2])] = contact_of(by)["applicable"]
    n_out = sum(1 for v in labels.values() if not v)
    # the cell mask (result review): every scan cell outside the contact bounds is an ideal
    # no-slip / no-flight constraint result, and supports no real-object statement
    mask = {}
    for R in RB.SIZES:
        bad = [k / 100.0 for k in range(RB.SCAN_HUNDREDTHS[0], RB.SCAN_HUNDREDTHS[1] + 1)
               if not labels["R%.2f_A%.2f" % (R, k / 100.0)]]
        runs = []
        for A in bad:
            if runs and abs(A - runs[-1][1] - 0.01) < 1e-9:
                runs[-1][1] = A
            else:
                runs.append([A, A])
        mask["R%.2f" % R] = runs
    return dict(name="contact", claimed=claimed, reported_cells=len(labels),
                reported_outside_bounds=n_out,
                ideal_constraint_result_cells=dict(label="ideal no-slip/no-flight constraint result",
                                                   bounds=dict(N_min=RB.N_MIN, mu_max=RB.MU_MAX),
                                                   amplitude_ranges=mask,
                                                   restitution_cells=[k for k, v in labels.items()
                                                                      if "_cv" in k and not v]),
                passes=all(c["applicable"] for c in claimed))


# 7. similarity ------------------------------------------------------------------------

def similarity_gate(doc):
    pairs = []
    for a, b in RB.SIMILARITY:
        rows = {r["impl"]: r for r in doc["similarity"]
                if r["pair"][0] == list(a) and r["pair"][1] == list(b)}
        P, T, E = rows["P"], rows["T"], rows["E"]
        same = all(rows[i]["outcomes"][0] == rows[i]["outcomes"][1] and
                   rows[i]["impacts"][0] == rows[i]["impacts"][1] for i in rows)
        dx_PT = max(abs(p - t) for p, t in zip(P["x1"], T["x1"]))
        dx_TE = max(abs(t - e) for t, e in zip(T["x1"], E["x1"]))
        u_x = dx_PT + dx_TE + abs(P["xmax_h"][0] - P["xmax_h2"][0]) + 64 * EPS
        disc = max(rows[i]["discrepancy"] for i in rows)
        tol = max(RB.SIMILARITY_TOL, 3 * u_x)
        pairs.append(dict(pair=[a, b], outcomes=P["outcomes"], impacts=P["impacts"], discrepancy=disc,
                          u_x=u_x, tolerance=tol, passes=bool(same and disc <= tol)))
    return dict(name="similarity", pairs=pairs, passes=all(p["passes"] for p in pairs))


# 8. animation -------------------------------------------------------------------------

def animation_gate(doc):
    rows = []
    ok = True
    for R in RB.SIZES:
        d = doc["display"]["R%.2f" % R]
        want = table_outcome(doc, R, RB.A_HEAD)
        pre = [th for t, th in zip(d["t"], d["theta"]) if d["uplift_s"] is not None and t < d["uplift_s"]]
        good = d["outcome"] == want and all(th == 0.0 for th in pre)
        if want == "O":
            good = good and d["after"] == "lies on its side" and abs(abs(d["theta"][-1]) - math.pi / 2) < 1e-12
        else:
            good = good and abs(d["theta"][-1]) < 1e-9
        rows.append(dict(R=R, outcome=d["outcome"], table=want, after=d["after"], passes=bool(good)))
        ok = ok and good
    return dict(name="animation", rows=rows, passes=bool(ok))


# 9. wording ---------------------------------------------------------------------------

TITLE = "UNDER THIS PULSE, THE BIGGER BLOCK STAYS STANDING"
TITLE_LINES = ("UNDER THIS PULSE,", "THE BIGGER BLOCK STAYS STANDING")
ON_SCREEN = dict(
    same="SAME SHAPE. SAME PULSE. ONLY THE SIZE CHANGES.",
    pulse_template="ONE SINE PULSE: PEAK {apg:.2f} g, {Tp:.1f} s",
    height_template="{h:.2f} m TALL",
    over="OVERTURNED",
    standing="STILL STANDING",
    standing_row="STANDING",
    height_row_format="{h:.2f} m",
    snapshots="SOLVER SNAPSHOTS — SIMULATED TIME",
    first_contact="FIRST SIDE CONTACT",
    side="POST-THRESHOLD MOTION ENDS AT FIRST SIDE CONTACT; THAT IMPACT IS NOT MODELLED",
    side_1="POST-THRESHOLD MOTION ENDS AT FIRST SIDE CONTACT;",      # the same words, in two lines
    side_2="THAT IMPACT IS NOT MODELLED",
    ideal_short="IDEAL RIGID-BLOCK MODEL",
    clock_format="t = {t:.2f} s",
    ground="GROUND ACCELERATION",
    energy_head="ROCKING ENERGY AT PULSE END",
    tip_line="TIP-OVER ENERGY BARRIER",
    energy_template="{e:.2f} × BARRIER",
    why="IN THIS RIGID ROCKING MODEL, A BIGGER BLOCK ROCKS MORE SLOWLY",
    five="ACROSS FIVE SIZES, THE TWO SMALLEST OVERTURNED",
    model="IDEAL MODEL: RIGID BLOCK, NO SLIDING OR BOUNCING, RIGID GROUND",
    pulse_note="ONE IDEALISED PULSE, NOT A RECORDED GROUND MOTION",
    limits="NOT A RATING FOR ANY REAL STATUE, TOMBSTONE OR ROCK",
)
SILENT = dict(
    result_template="{hs:.2f} m block overturned; {hb:.2f} m block standing",
    limits=["ideal rigid block, no sliding or bouncing",
            "not a recorded ground motion; not a rating for any real object",
            "drawn to first side contact; that impact is not modelled"],
)
NARRATION = dict(
    a1="Two blocks with the same shape. The big one is four times as tall and four times as wide.",
    b1="The ground gives both the same single pulse, half a second long.",
    b2="After the pulse, the small one tips over. The big one keeps rocking without overturning.",
    c1="Again, at selected simulated times.",
    d1="At the end of the pulse, the small block's rocking energy was one point seven times the tip-over energy barrier. The big one's was less than half.",
    d2="In this rigid rocking model, a bigger block rocks more slowly.",
    e1="Across five sizes, the two smallest overturned.",
    z1="One idealised pulse, and a rigid block that cannot slide or bounce: not a rating for any real statue.",
)
SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0, "first side contact"]   # simulated seconds
SPOKEN = dict(d1_small=1.7, d1_big_below=0.5)
FORBIDDEN = [r"\bsafe", r"earthquake", r"\bsurviv", r"\bproof\b", r"\bdesign", r"minimum overturning",
             r"\bspectrum\b", r"\bsettl", r"stronger", r"\balways\b", r"\bnever\b", r"\bguarantee",
             r"\bprove", r"ruled out", r"\bwill (tip|fall|stand)", r"\bany (block|statue|tombstone)\b(?! )",
             r"tombstones? (would|will)", r"statues? (would|will)", r"\bmaterial property"]


def hits(text):
    found = []
    for pat in FORBIDDEN:
        for m in re.finditer(pat, text, re.I):
            found.append((m.group(0), text[max(0, m.start() - 30):m.end() + 30]))
    return found


def headline_values(doc):
    g = group(doc["scan"])
    small, big = (g[(R, RB.A_HEAD, 1.0, "sine", RB.TP)] for R in RB.HEAD_PAIR)
    def with_u(by, q):
        P, T, E = (by[i][q] for i in ("P", "T", "E"))
        return P, u_of(P, T, E, P, P)
    es, ues = with_u(small, "e_end")
    eb, ueb = with_u(big, "e_end")
    return dict(e_small=es, u_e_small=ues, e_big=eb, u_e_big=ueb,
                h_small=2 * RB.HEAD_PAIR[0] * math.cos(RB.ALPHA), h_big=2 * RB.HEAD_PAIR[1] * math.cos(RB.ALPHA),
                apg=RB.A_HEAD * math.tan(RB.ALPHA), Tp=RB.TP, size_ratio=RB.HEAD_PAIR[1] / RB.HEAD_PAIR[0],
                outcomes=[table_outcome(doc, R, RB.A_HEAD) for R in RB.SIZES], sizes=list(RB.SIZES),
                heights=[2 * R * math.cos(RB.ALPHA) for R in RB.SIZES])


def run(write=True, texts=None):
    doc = load()
    gates = [anchor_gate(doc), audit_gate(doc), impact_gate(doc), headline_gate(doc), restitution_gate(doc),
             contact_gate(doc), similarity_gate(doc), animation_gate(doc)]
    v = headline_values(doc)
    on = {k: val for k, val in ON_SCREEN.items() if not k.endswith("_template")}
    on["pulse"] = ON_SCREEN["pulse_template"].format(apg=v["apg"], Tp=v["Tp"])
    on["height_small"] = ON_SCREEN["height_template"].format(h=v["h_small"])
    on["height_big"] = ON_SCREEN["height_template"].format(h=v["h_big"])
    on["energy_small"] = ON_SCREEN["energy_template"].format(e=v["e_small"])
    on["energy_big"] = ON_SCREEN["energy_template"].format(e=v["e_big"])
    silent = dict(result=SILENT["result_template"].format(hs=v["h_small"], hb=v["h_big"]), limits=SILENT["limits"])
    # the spoken numbers are the computed ones, rounded as said (displayed precision >= u)
    spoken = dict(d1_small=(round(v["e_small"], 1), SPOKEN["d1_small"]),
                  d1_big_below=(v["e_big"] < SPOKEN["d1_big_below"], True),
                  display_precision=(max(v["u_e_small"], v["u_e_big"]) < 0.005, True))
    spoken_ok = abs(spoken["d1_small"][0] - spoken["d1_small"][1]) < 1e-9 and spoken["d1_big_below"][0]         and spoken["display_precision"][0]
    v["snapshot_times"] = SNAPSHOT_TIMES
    texts = (texts if texts is not None else
             [TITLE] + list(TITLE_LINES) + list(on.values()) + list(NARRATION.values()) + [silent["result"]]
             + silent["limits"])
    found = [h for t in texts for h in hits(t)]
    gates.append(dict(name="spoken", checks=spoken, passes=bool(spoken_ok)))
    gates.append(dict(name="wording", hits=found, n_texts=len(texts), passes=not found))
    claim = ("In Housner's ideal rigid rocking model (planar rectangular block, slenderness 0.2 rad, no "
             "sliding or bouncing, rigid ground, impact factor 1 - 1.5 sin^2 alpha), under one sine pulse of "
             "peak %.3f g lasting %.1f s, the %.2f m tall block overturned and the geometrically similar %.2f m "
             "tall block did not; across the five registered sizes (R 0.25-4 m) the two smallest overturned." %
             (v["apg"], v["Tp"], v["h_small"], v["h_big"]))
    out = dict(title=TITLE, title_lines=TITLE_LINES, claim=claim, on_screen=on, silent=silent,
               narration=NARRATION, values=v, source_sha256=doc.get("source_sha256"),
               runtime_s=doc.get("runtime_s"), gates=gates,
               passed=sum(bool(g["passes"]) for g in gates), total=len(gates))
    if write:
        with open(os.path.join(DATA, "gates.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1, default=str)
    return out


if __name__ == "__main__":
    res = run()
    for g in res["gates"]:
        print("%-12s %s" % (g["name"], "PASS" if g["passes"] else "FAIL"))
    print("%d / %d" % (res["passed"], res["total"]))
