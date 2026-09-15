# Blocks

Rigid blocks, solved in ideal models: stacks solved for balance, and single blocks
solved for rocking. One repository for the rig, and one folder per study inside it.

```
past-the-edge/     how far past a table edge identical blocks can be balanced
under-this-pulse/  two blocks of one shape, one ground pulse: which one overturns
```

## Why one repository and not one per study

Earlier videos on this channel each had their own repository, and the same
solver was copied from one to the next. **Every copy is a place that rots on
its own.** Here each study folder ships only what is new to it, and its
`reproduce.py` checks its modules against a frozen SHA-256 before running
anything. When a later study shares code, that code moves to the root once.
The two studies here share none.

## Running one

```
cd past-the-edge
python reproduce.py           # the whole study and every gate (minutes)
python reproduce.py --quick   # three and four blocks only

cd under-this-pulse
python reproduce.py           # the whole registered run and every gate
python reproduce.py --quick   # the five sizes at A = 2 and one control
```

`requirements.txt` is numpy and scipy. Nothing else is needed.

## The studies

| folder | asks | answers with |
|---|---|---|
| [`past-the-edge`](past-the-edge/) | stacked one on one, or side by side — how far past the edge can the same blocks be balanced? | the maximum balanced overhang, in block lengths |
| [`under-this-pulse`](under-this-pulse/) | the same shape at five sizes, the same sine pulse of ground acceleration — which blocks overturn? | overturned or standing, with the energy each block has left when the pulse ends |

## What the rig is

- **Balance** (`past-the-edge`): the model of Paterson and Zwick ("Overhang",
  *American Mathematical Monthly* 116, 2009; arXiv:0710.2357): 2-D blocks of
  length 1 and weight 1 in levels, a table top at x ≤ 0, and a stack is balanced
  when non-negative vertical forces at the ends of every contact interval hold
  every block in force and moment balance — a linear programme.
- **Rocking** (`under-this-pulse`): Housner's model (*Bulletin of the
  Seismological Society of America* 53, 1963): a planar rigid rectangle rocking
  about its bottom corners on rigid ground that moves horizontally, with no
  sliding or bouncing, and a landing that keeps angular momentum about the new
  corner.

**Verification, not validation.** Every check inside a study reads the same
idealisation its solver reads: rigid, homogeneous blocks with sharp edges, and
in each study the further assumptions its README lists. Validating any of it
would need real blocks and real measurements, and there are none.
