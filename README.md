# Phantom Consensus

## Team Information
- **Team Name**: SASCode
- **Year**: 2028
- **All-Female Team**: No

## Architecture Overview

#### Describe your approach here. Keep it short and clear.

- **Data Cleaning:** We used Pandas vectorized operations to normalize IDs (lowercase, strip whitespace). Missing values and outliers (e.g., negative influence, strings like "high") were sanitized using `pd.to_numeric` with `coerce`, filled with defaults, and clamped to valid bounds using `.clip()`. Ghost records were pruned via strict referential integrity checks against valid ID sets.
- **Alliances & Asymmetric Trust:** We calculated a `relationship_score = trust * (1 - betrayal_prob)`. To thwart "False Friends" and asymmetric trust, we require a strict BIDIRECTIONAL relationship score > 50. Both representatives must mutually trust each other with low betrayal probability to form a valid alliance.
- **Proposal Prioritization:** We built an ML Ensemble Architecture (Stakeholder, Budget, and NLP engines). For objections, we calculated `objection_weight = sum(severity * objector_influence)` to derive a `controversy` metric. The Budget Engine normalized `priority`, and proposals were ranked by a Soft Voting Meta-Aggregator that balances Political viability, Economic priority, and NLP sentiment risk.
- **Stable Agreement Strategy:** We aggressively filtered "Trojan Horses" (reps with an average betrayal hazard > 0.5 across all relations) and "Faction Infiltrators". To avoid "Poison Pills," we implemented a strict "Supporter Coherence" check: chosen supporters must be safe reps who have absolutely zero objections to any of the selected proposals.
**Note:** Please do not change the format or spelling of anything in this README. The fields are extracted using a script, so any changes to the structure or formatting may break the extraction process.
