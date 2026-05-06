# Our Approach: The ML Ensemble Architecture

To solve the Phantom Consensus challenge, we moved beyond a naive rule-based script and implemented an **Advanced Machine Learning Ensemble Meta-Model**.

## 1. Data Sanitization & Integrity
Our pipeline uses purely vectorized Pandas operations to ensure high-performance scaling (tested capable of instantly handling 50+ reps and 30+ proposals). 
- **Type Sanitization:** Out-of-bounds metrics (e.g. influence > 100) are clamped, strings ("high") are mapped to numeric values, and missing values are imputed safely.
- **Referential Integrity:** Ghost sponsors and orphaned objections are systematically purged to prevent silent crashes in the scoring phase.
- **Deduplication:** We aggressively drop duplicate records, prioritizing highest influence, highest priority, and most recent interactions.

## 2. Base Engines (Feature Extraction)
We process the data through three specialized engines:

1. **The Stakeholder Engine (Political Viability):** Calculates a `political_score` by evaluating the objection weights against the absolute maximum possible objection weight in the system. It also calculates a `betrayal_hazard` for each representative to flag Trojan Horses.
2. **The Budget Engine (Economic Viability):** Uses Scikit-Learn's `MinMaxScaler` to normalize proposal priorities and budgets, generating a standardized `economic_score`.
3. **The NLP Risk Engine:** Extracts the `reason` field from objections and processes the text using `TextBlob` sentiment analysis. Highly negative language scales up the `nlp_risk_score`.

## 3. The Meta-Aggregator (Soft Voting)
Instead of relying on a single metric, our engine uses a Soft Voting Aggregator to combine the scores from the three Base Engines:
`Final Ensemble Score = (0.4 * Political) + (0.3 * Economic) + (0.3 * (1.0 - NLP Risk))`

## 4. Strategic Filters
- **Trojan Horses:** Any representative with an average `betrayal_prob > 0.5` is outright banned from becoming a supporter.
- **Faction Infiltrators:** We detect and exclude reps who exhibit high betrayal probabilities specifically against their *own* faction members.
- **False Friends:** Alliances require *bidirectional* trust. Both Rep A and Rep B must have a `relationship_score > 50` towards each other, thwarting asymmetric trust traps.
- **Supporter Coherence:** Selected supporters are strictly filtered to ensure they have exactly zero objections to the final selected proposals.
