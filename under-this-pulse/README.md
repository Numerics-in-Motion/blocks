# Under this pulse

Two rigid blocks of the same shape stand on the same ground. One is 0.98 m tall, the
other four times as tall and four times as wide, 3.92 m. The ground gives both the same
single sine pulse: 0.405 g at its peak, half a second long. **The small block
overturns. The big one keeps rocking without overturning.**

```
python reproduce.py            # the whole registered run and every gate
python reproduce.py --quick    # the five sizes at A = 2 and one control (a minute or two)
```

The full run took 626 s on 20 worker processes; expect about half an hour on eight
cores.

## What is claimed

In Housner's ideal rigid rocking model (a planar rectangular block of slenderness
α = 0.2 rad on rigid ground, no sliding or bouncing, impact factor
c<sub>v</sub> = 1 − 1.5 sin²α), under one sine pulse of ground acceleration peaking at
0.405 g and lasting 0.5 s, the 0.98 m tall block overturned and the geometrically
similar 3.92 m tall block did not. Across the five registered sizes (0.49 to 7.84 m
tall) the two smallest overturned.

## The model

G. W. Housner, *Bulletin of the Seismological Society of America* 53(2), 403–417
(1963). A homogeneous rigid
rectangle of half-diagonal R rocks about one bottom corner or the other:

    θ'' = −p² [ sin(α sgn θ − θ) + (ü_g / g) cos(α sgn θ − θ) ],     p = √(3g / 4R).

It lifts off when |ü<sub>g</sub>| exceeds g tan α. Landing on the other corner conserves
angular momentum about that corner, so the angular velocity keeps its direction and is
multiplied by c<sub>v</sub> (kinetic energy by c<sub>v</sub>²). Only R changes between
the blocks; mass and density cancel.

The pulse is ü<sub>g</sub> = a<sub>p</sub> sin(2πt/T<sub>p</sub>) for one cycle, with
T<sub>p</sub> = 0.5 s and A = a<sub>p</sub>/(g tan α) = 2.

## What is computed

| step | what it does |
|---|---|
| forced phase | event by event in x = θ/α, τ = pt: landings at x = 0, overturning at \|x\| = 1 |
| after the pulse | no work is done, so the fate is decided exactly by energy: E/E<sub>b</sub> = (α²v²/2 + cos(α(1−\|x\|)) − cos α)/(1 − cos α); moving outward the block overturns if E ≥ E<sub>b</sub>, moving inward it lands first and keeps c<sub>v</sub>²E |
| four implementations | P (DOP853, rtol 10⁻¹⁰), T (rtol 10⁻¹², half the step), E and E2 (the production stepping, every event bracketed independently from dense output to 10⁻¹³, at two search spacings) |
| amplitude scan | every size, A = 1.00 to 6.00 in steps of 0.01 (10,020 solves); every unlike neighbour bisected to 10⁻⁵ in each implementation |
| restitution | c<sub>v</sub> × 0.8, 0.9 and 1.0 at A = 2 |
| rectangular pulse | at α = 0.02 the first overturning amplitude reproduces Housner's a<sub>p</sub>/(gα) = 1/(1 − e<sup>−pt<sub>d</sub></sup>) to 1.6 × 10⁻⁴ |
| similarity | the same α, A and ω<sub>p</sub>/p at two sizes give the same dimensionless motion |
| contact | the normal and friction forces the ideal model requires, over the pulse and the motion after it (the latter exactly on the energy curve) |

## The numbers

At A = 2 (peak 0.405 g, 0.5 s):

| R | tall | outcome | E/E<sub>b</sub> at pulse end | least N/mg | largest \|H\|/N |
|---:|---:|---|---:|---:|---:|
| 0.25 m | 0.49 m | overturned | 2.677 | 0.865 | 0.246 |
| 0.50 m | 0.98 m | overturned | 1.711 | 0.885 | 0.249 |
| 1.00 m | 1.96 m | standing | 0.925 | 0.901 | 0.250 |
| 2.00 m | 3.92 m | standing | 0.477 | 0.909 | 0.251 |
| 4.00 m | 7.84 m | standing | 0.242 | 0.911 | 0.251 |

The same five outcomes hold at A = 1.90 and 2.10, and with c<sub>v</sub> reduced by 10 %
and by 20 %. On the scan every size stands below one amplitude and overturns above it:
A = 1.26189 (R 0.25 m), 1.45844 (0.5 m), 2.16378 (1 m), 5.15493 (2 m); the 4 m block did
not overturn anywhere on A ≤ 6. No overturned–standing–overturned sequence appeared on
the scan; that says nothing about amplitudes that were not run.

`data/scan_cells.csv` lists every cell with its four implementations. From A ≈ 3.1
upward the model needs a friction ratio above 0.30 at the corner, and those cells are
labelled **ideal no-slip/no-flight constraint result**: they hold inside the model and
support no statement about a real block.

## What it does not say

It is one idealised pulse, not a recorded ground motion, and a rigid block that cannot
slide or bounce on rigid ground. Sliding, bouncing, soil, vertical shaking and 3-D
motion are outside the model; the impact factor is Housner's, not a measured
property. It is not a rating for any real statue, tombstone or rock, and it does not
say that bigger blocks in general fare better.

After an overturned block passes its tipping angle, the video carries the same equation
on to first side contact and stops there; that impact is not modelled. How long the big block takes to come to
rest is not computed.

## Numerical notes

- P and T place every landing within 5.2 × 10⁻¹² of each other. Just above lift-off
  the block chatters through hundreds of landings in the pulse, and there the two
  event-audit repeats E and E2 drift apart as landings accumulate, by at most
  1.6 × 10⁻¹⁰ (R 0.25 m, A 1.03). Outcomes agree in every cell.
- A landing slower than 10⁻¹² (dimensionless) leaves the block at rest. That cut-off
  fired 76 times, in 14 cells at A = 1.01–1.03; there the impact counts depend on it
  (the outcomes did not change with it at 10⁻¹⁰, 10⁻¹² and 10⁻¹⁴).

## Files

```
reproduce.py                        this check
solver/rocking_block.py             the study
solver/rocking_block_gates.py       the registered gates
data/scan_cells.csv                 every amplitude-scan cell
data/transitions.csv                the bisected transitions
data/restitution_cells.csv          the restitution sensitivity
data/gates.json                     the gate record
reference/reference_canonical.json  the frozen reference
```
