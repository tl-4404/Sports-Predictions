# July 11, 2026 MLB pitcher-prop model grade

This grades the latest updated rerun, not the superseded overnight board.
Official results were pulled from MLB StatsAPI after all July 11 games became final.

## Executive summary

- Official ranked recommendations: 8 wins, 3 losses (72.7%).
- Ranked pitching-outs recommendations: 5 wins, 2 losses (71.4%).
- Ranked strikeout recommendations: 3 wins, 1 loss (75.0%).
- Expected hits from the stated midpoint probabilities: 6.84; actual hits: 8.
- Unique suggested two-leg constructions: 0 wins, 4 losses.
- Main reason all cards lost: Tyler Mahle under 16.5 outs was the common anchor and lost.

## Official ranked recommendations

| Category | Prop | Result | Final | Outcome |
|---|---|---:|---:|---|
| Outs | Tyler Mahle under 16.5 outs | 21 outs | 7.0 IP, 104 pitches | LOSS |
| Outs | Griffin Jax under 15.5 outs | 15 outs | 5.0 IP, 69 pitches | WIN |
| Outs | Logan Gilbert over 17.5 outs | 20 outs | 6.2 IP, 96 pitches | WIN |
| Outs | Cristopher Sanchez under 18.5 outs | 21 outs | 7.0 IP, 103 pitches | LOSS |
| Outs | Javier Assad over 14.5 outs | 15 outs | 5.0 IP, 82 pitches | WIN |
| Outs | Ryan Johnson under 15.5 outs | 15 outs | 5.0 IP, 75 pitches | WIN |
| Outs | Freddy Peralta under 17.5 outs | 13 outs | 4.1 IP, 92 pitches | WIN |
| Ks | Cam Schlittler over 5.5 K | 6 K | 6.2 IP, 99 pitches | WIN |
| Ks | Walker Buehler over 3.5 K | 1 K | 2.0 IP, 65 pitches | LOSS |
| Ks | Eury Perez under 6.5 K | 6 K | 6.0 IP, 99 pitches | WIN |
| Ks | Ryan Johnson under 4.5 K | 1 K | 5.0 IP, 75 pitches | WIN |

## Suggested card constructions

| Construction | Leg results | Card result |
|---|---|---|
| Mahle under 16.5 outs + Buehler over 3.5 K | LOSS + LOSS | LOSS |
| Mahle under 16.5 outs + Eury Perez under 6.5 K | LOSS + WIN | LOSS |
| Mahle under 16.5 outs + Griffin Jax under 15.5 outs | LOSS + WIN | LOSS |
| Mahle under 16.5 outs + Logan Gilbert over 17.5 outs | LOSS + WIN | LOSS |

The later mixed and highest-hit-rate examples repeated the Mahle/Buehler and Mahle/Jax combinations, so they are not counted as additional unique cards.

## Additional published K directions that were not official ranked plays

| Prop | Pipeline label | Final K | Would have graded |
|---|---|---:|---|
| Yoshinobu Yamamoto over 5.5 K | Lean only | 6 | WIN |
| Tyler Mahle under 4.5 K | Freeze | 4 | WIN |
| Tanner Bibee under 4.5 K | Freeze | 3 | WIN |
| Freddy Peralta under 5.5 K | Freeze | 6 | LOSS |
| Cristopher Sanchez under 6.5 K | Pass/freeze | 7 | LOSS |
| Logan Gilbert under 6.5 K | Pass | 5 | WIN |
| Nick Lodolo over 4.5 K | Pass | 4 | LOSS |
| Kumar Rocker under 4.5 K | Pass/small lower lean | 6 | LOSS |

The pass/freeze group therefore contained 3 potential winners and 4 potential losers. Abstaining avoided more losing directions than winning directions.

## Lines with no committed direction

| Line | Final result | Note |
|---|---:|---|
| Joe Ryan 7.5 K | 6 K | Under would have won, but the latest rerun called the line efficient and did not commit a side. |
| Gage Jump 5.5 K | 7 K | Over would have won, but the latest rerun treated it as approximately 50/50. |

## Role and probable-pitcher controls

- Bryan Hudson was correctly removed as a normal starter play. He recorded 3 outs and 0 strikeouts on only 7 pitches.
- Miles Mikolas did not start; PJ Poulin started instead. Any Mikolas line should have been void or removed.
- Walker Buehler's failure was not merely a narrow miss: he recorded 1 strikeout and 6 outs, allowing 4 earned runs before an early hook.

## Main model errors

1. Tyler Mahle leash was materially underestimated. The model projected roughly 14.0-14.8 outs, but he threw 104 pitches and completed 7 innings. The bullpen concern was real, and the Giants extended him well beyond his post-return 70-85 pitch pattern.
2. Cristopher Sanchez leash/efficiency was underestimated. He completed 7 innings and 21 outs despite allowing 10 hits, showing that the current model overweights traffic as an automatic early-hook signal.
3. Walker Buehler's downside tail was underestimated. The projection centered around 4.1-4.5 strikeouts, but an early performance-based hook produced only 1 strikeout.
4. Card construction overconcentrated one uncertain assumption. Mahle appeared in every suggested combination, converting a good 8-3 individual slate into a 0-4 card slate.

## Grades

- Individual line selection: B+ (8-3, 72.7%).
- Strikeout recommendations: B+ (3-1), although the wider K directional board remained mediocre.
- Pitching-outs recommendations: B (5-2), with both misses caused by underestimated leash.
- Risk-control/pass discipline: B+; the role kill on Hudson was correct and pass/freeze decisions avoided four losing K directions.
- Card construction: F; every unique suggested two-leg card lost.
- Overall user-facing pipeline grade: C+.

The C+ reflects the gap between good single-prop classification and poor practical card construction. A bettor following the ranked lines selectively could have done well; a bettor following the proposed combinations would have lost every card.
