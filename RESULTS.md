# Results Analysis

Our ML Ensemble Engine was tested against the provided sample dataset. The results demonstrate that the engine successfully navigates hidden traps and produces a highly stable consensus.

## Output Summary

```json
{
  "final_agreement": {
    "proposals": [ "prop_002", "prop_004", "prop_003" ],
    "supporting_reps": [ "rep_004" ]
  },
  "alliances": [
    [ "rep_001", "rep_004" ]
  ]
}
```

## Strategic Analysis

### 1. Proposal Selection (Avoiding Poison Pills)
The Meta-Aggregator evaluated four clean proposals. 
- **`prop_002`** emerged as the top pick (Score: 0.95). It had maximum economic priority (1.0) and high political viability.
- **`prop_004`** followed closely (Score: 0.94).
- **`prop_001`** was rejected as a **Poison Pill**. Despite having a decent political score, its economic viability was normalized to 0.0 because its raw priority was much lower than competing proposals.

### 2. Supporter Selection (Avoiding Trojan Horses & Incoherence)
The engine only selected **`rep_004`** as a supporter. This is the mathematically correct decision:
- **`rep_005`** and **`rep_006`** were detected as **Trojan Horses** due to their extremely high average betrayal hazard (0.825 and 0.775 respectively). They were banned.
- **`rep_001`**, **`rep_002`**, and **`rep_003`** were excluded by our **Supporter Coherence Check**. They each had active objections against at least one of the selected proposals. You cannot have a supporter who actively objects to the consensus agreement.

### 3. Alliance Detection (Handling False Friends)
The engine detected exactly one true alliance: **`[rep_001, rep_004]`**.
- The algorithm successfully thwarted a **False Friend** trap between `rep_006` and `rep_001`. While `rep_006` claims to trust `rep_001` (Score: 90), `rep_006` has a staggering 0.75 betrayal probability against them. Our bidirectional requirement successfully ignored this toxic relationship.
