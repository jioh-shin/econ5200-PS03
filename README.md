# Evaluating Difference-in-Differences with Staggered Treatment Adoption

## Objective

I examined how staggered treatment adoption, heterogeneous treatment effects, and clustering can affect the reliability of Difference-in-Differences (DiD) estimates.

## Methodology

- Replicated the Card and Krueger minimum wage study using a two-by-two DiD approach.
- Simulated a panel of 40 counties across 8 regional boards over 12 years, with treatment introduced at different times.
- Applied TWFE regression and Goodman-Bacon decomposition to examine how already-treated groups can affect estimates.
- Estimated treatment effects using never-treated or not-yet-treated counties as controls.
- Constructed an event study to examine pre-treatment trends and changes in treatment effects over time.
- Compared default, county-clustered, and board-clustered standard errors.
- Implemented a wild cluster bootstrap using all 256 possible Rademacher weight combinations.

## Key Findings

- The Card and Krueger replication produced a DiD estimate of 2.75 FTE.
- The simulated true ATT was 10.361, while the TWFE estimate was -0.644, even though all true treatment effects were positive.
- The corrected DiD estimate was 10.433, much closer to the true ATT.
- The event study showed pre-treatment estimates close to zero and increasing treatment effects after adoption.
- The conventional board-clustered p-value was 0.8702, compared with a wild cluster bootstrap p-value of 0.9531. Neither was statistically significant at the 5% level.

Overall, I found that TWFE can give misleading results when treatment effects vary over time. Using appropriate control groups produced an estimate closer to the true effect, while clustering at the treatment-assignment level was important for statistical inference.
