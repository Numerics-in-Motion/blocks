# Blocks

Stacks of identical rigid blocks, solved for balance. One repository for the
rig, and one folder per study inside it.

```
past-the-edge/     how far past a table edge identical blocks can be balanced
```

## Why one repository and not one per study

Earlier videos on this channel each had their own repository, and the same
solver was copied from one to the next. **Every copy is a place that rots on
its own.** Here each study folder ships only what is new to it, and its
`reproduce.py` checks its modules against a frozen SHA-256 before running
anything. When a later study shares code, that code moves to the root once.

## Running one

```
cd past-the-edge
python reproduce.py           # the whole study and every gate (minutes)
python reproduce.py --quick   # three and four blocks only
```

`requirements.txt` is numpy and scipy. Nothing else is needed.

## The studies

| folder | asks | answers with |
|---|---|---|
| [`past-the-edge`](past-the-edge/) | stacked one on one, or side by side — how far past the edge can the same blocks be balanced? | the maximum balanced overhang, in block lengths |

## What the rig is

The balance model of Paterson and Zwick ("Overhang", *American Mathematical
Monthly* 116, 2009; arXiv:0710.2357): 2-D blocks of length 1 and weight 1 in
levels, a table top at x ≤ 0, and a stack is balanced when non-negative
vertical forces at the ends of every contact interval hold every block in force
and moment balance — a linear programme.

**Verification, not validation.** Every check inside a study reads the same
idealisation its solver reads: rigid, identical, homogeneous, frictionless
blocks with sharp edges. Validating any of it would need real blocks and real
measurements, and there are none.
