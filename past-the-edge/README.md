# Past the edge

Four identical blocks at a table edge. Stacked one on one — each block resting
on a single block below — they can be balanced at most **1.0417** block lengths
past the edge. Allowed to sit side by side, the same four blocks can be
balanced **1.1679** block lengths out: **+12.1 %**.

```
python reproduce.py            # the whole study and every gate (minutes)
python reproduce.py --quick    # three and four blocks only (about a minute)
```

## What is claimed

In an ideal 2-D model of identical, rigid, homogeneous, frictionless blocks,
counterweights raise the maximum balanced overhang: for four blocks the balance
limit is 1.1679 block lengths, against 1.0417 under the one-on-one rule
(+12.1 %); for three blocks it is 1.0000 against 11/12 (+9.1 %), and any
positive contact margin removes that three-block gain. Both are balance limits
with zero margin.

## The model

Paterson and Zwick, "Overhang", *American Mathematical Monthly* 116 (2009),
arXiv:0710.2357, Theorem 2.2. Blocks of length 1 and weight 1 lie in levels;
the bottom level rests on a table whose top is x ≤ 0. A block rests on the
blocks of the level below whose intervals meet its own. A stack is **balanced**
when non-negative vertical forces at the two ends of every contact interval
hold every block in force and moment balance. Whether a given stack is balanced
is a linear programme; the largest overhang is not, because the moments multiply
forces by positions.

## What is computed

| step | what it does |
|---|---|
| enumeration | every candidate contact pattern of 1 to 6 blocks — the split into levels and which blocks each block touches, with closed contact intervals, so a single point counts (1, 2, 6, 22, 91, 406: 528). A linear programme keeps the 479 that exist with exactly their own contacts (every other adjacent pair held apart by a positive clearance); 24 exist only with extra contacts and are covered by the richer pattern, 25 have no placement. 24,000 random stacks and 24,000 stacks on a half-block grid, where blocks touch end to end, all fall into one of the 479 |
| formulation A | per topology and per candidate furthest block: positions and end forces, maximised by SLSQP from 24 starts |
| formulation B | the same maximum, written separately: one resultant per contact at a point inside the contact, torques as r × F with the block height carried through |
| no balanced placement | of the 479, both formulations solve 373, and the other 106, which no start could balance, are certified to have no balanced placement, by branch and bound on one connected component. Four balanceable controls are not certified |
| certificate | a branch-and-bound upper bound for 1 to 4 blocks: no balanced stack exceeds the value found by more than 0.002 block lengths, at two LP tolerances |
| harmonic | the one-on-one stacks balance at exactly half the harmonic number, and not 10⁻⁶ further |
| height | block height changes nothing, at heights 0.1 to 2 block lengths |
| contact margin | every contact force held a margin inside its contact, from 10⁻⁶ to 0.05 block lengths |

## The numbers

| blocks | one on one | unrestricted | published |
|---:|---:|---:|---:|
| 1 | 0.500000 | 0.500000 | 0.5 |
| 2 | 0.750000 | 0.750000 | 0.75 |
| 3 | 0.916667 | 1.000000 | 1 |
| 4 | 1.041667 | 1.167893 | (15 − 4√2)/8 = 1.16789 |
| 5 | 1.141667 | 1.304555 | 1.30455 |
| 6 | 1.225000 | 1.436700 | 1.4367 |

Both formulations agree to better than 10⁻¹¹ on every row.

The four-block gain survives a contact margin: with every contact force held
10⁻⁶ to 0.05 block lengths inside its contact, both limits shrink by the same
factor and the gain stays +12.1 %. The three-block gain does not survive any
margin at all. With every resultant a margin ε > 0 inside its contact:

- the counterweighted layout (one block, two on top) has no balanced placement:
  each upper block rests on the bottom block alone, so two disjoint unit blocks
  on one unit block would need 1 + 2ε ≤ 1;
- two blocks on the table and one on top end at most 3/4 − 3ε/2 past the edge
  (the right-hand table block carries its own weight and at most the top
  block's, and its table resultant sits ε behind the edge);
- three blocks on the table end at most 1/2 − ε;
- the one-on-one stack with every offset (1/2 − ε)/k is balanced and ends at
  (1 − 2ε) 11/12, which is larger than all of those for every ε < 1/2.

So at any positive margin the largest three-block overhang belongs to the
one-on-one class. The run checks this argument at every sampled margin: the
counterweighted layout has no balanced placement at any of them, and the largest
two-and-one overhang found
equals 3/4 − 3ε/2 to 10⁻¹².

## What it does not say

It is about balance and nothing more: every limit here has zero margin, and a
stack at its limit has nothing to spare. The model has no friction, no sliding
and no tipping — the video repositions blocks through balanced states and says
so. It assumes rigid, identical, homogeneous blocks with sharp edges, and says
nothing about real blocks or about more than six blocks.

## Files

```
reproduce.py                        this check
solver/block_overhang.py            the study
solver/block_overhang_gates.py      the registered gates
data/                               the run and the gate record
reference/reference_canonical.json  the frozen reference
```
