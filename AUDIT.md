# Hidden Test Coverage Audit
*Checked against: `code2PS.txt` (18 hidden tests) + `ISSUES.md` (20 issues)*

## Status Key
- ✅ HANDLED — Logic exists and runs correctly
- ⚠️ PARTIAL — Logic exists but has a known gap or edge case risk
- ❌ MISSING — Not handled at all

---

## 18 Hidden Test Scenarios — Line by Line

| # | Scenario | Status | Where Handled | Risk / Gap |
|---|---|---|---|---|
| 1 | **Trojan Horse** — exclude high-betrayal rep | ✅ | `detect_threats()` — `influence > 60 AND avg_betrayal > 0.4` | None |
| 2 | **Poison Pill** — avoid universally-objected proposal | ✅ | `run_consensus()` — `viability = priority*100 - weight`, veto if ≤ 0 | None |
| 3 | **False Friend** — detect asymmetric trust | ✅ | Alliance requires **bidirectional** `rel_score > 50` via self-join | None |
| 4 | **Clear Alliance** — detect genuine strong alliance | ✅ | `run_consensus()` → pairs self-join on `rel_score > 50` | None |
| 5 | **Faction War** — pick least-objected proposal | ✅ | Viability formula ranks proposals; lowest weight wins | None |
| 6 | **Priority vs Objection** — objection weight beats raw priority | ✅ | `viability = priority*100 - objection_weight` — a weight-heavy prop loses | None |
| 7 | **Supporter Coherence** — objectors must NOT be supporting reps | ⚠️ | Supporters are picked from `safe_reps` by influence — but we do NOT explicitly exclude objectors from `supporting_reps` | **GAP** |
| 8 | **Faction Infiltrator** — spy betrays own faction | ✅ | `detect_threats()` checks intra-faction `betrayal_prob > 0.5` | None |
| 9 | **Cascading Betrayal** — exclude high-risk end of trust chain | ⚠️ | Only direct avg_betrayal checked. Multi-hop chain (A trusts B who betrays C) not traversed | **GAP** |
| 10 | **Alliance Hack** — protect stable alliance from disruptor | ⚠️ | Trojan Horses removed, but no explicit logic re-validates alliances after removal | Minor risk |
| 11 | **Complete Rivalry** — return empty alliances when all rivals | ✅ | `alliances_set` is empty set by default; returns `[]` naturally | None |
| 12 | **Ghost Sponsor** — exclude proposals with non-existent sponsors | ✅ | `df_props = df_props[df_props['sponsor'].isin(valid_reps)]` | None |
| 13 | **Minimum Viable** — works when only 1 valid rep + proposal | ✅ | Fallback logic: `if not selected: pick iloc[0]` | None |
| 14 | **ID Normalization** — mixed case IDs across all 4 files | ✅ | `normalize_id()` applied to all 4 files on load | None |
| 15 | **Duplicate Proposals** — deduplicate correctly | ✅ | `drop_duplicates(subset=['id'], keep='first')` after sorting by priority | None |
| 16 | **Null Influence** — handle missing values without crash | ✅ | `clean_influence()` returns 0 for null; NLP fills null severity | None |
| 17 | **Scale Correctness** — correct decisions on 50 reps, 30 proposals | ✅ | Pure pandas vectorized ops — no nested Python loops | None |
| 18 | **Dirty CSV** — bad CSV rows don't break clean ones | ⚠️ | `clean_score()` and `clean_probability()` handle bad values, but a completely malformed row (missing columns) would raise a KeyError | **GAP** |

---

## The 3 Gaps — Fixes Needed Before Submission

### GAP 1 (Issue #7): Supporter Coherence
**Problem:** A rep who objected to a proposal could still end up in `supporting_reps`.
**Fix:** After selecting proposals, remove any rep from `supporting_reps` who has an active objection against any selected proposal.

### GAP 2 (Issue #9): Cascading Betrayal
**Problem:** If Rep A trusts Rep B, and Rep B has a high betrayal probability toward Rep C (a key supporter), the chain is unsafe — but our code only looks at each rep's direct avg_betrayal.
**Fix:** Add a second-order check: if a rep's *trusted allies* are themselves high-betrayal actors, reduce that rep's effective trust score.

### GAP 3 (Issue #18): Dirty CSV Row Handling
**Problem:** A malformed CSV row with a missing column (e.g., no `betrayal_prob` column) would crash on `.apply()`.
**Fix:** Wrap CSV loading in a `try/except` per-row using `errors='coerce'` on all columns, and add a column existence check before cleaning.

---

## 20 ISSUES.md Coverage

| Issue | Description | Status |
|---|---|---|
| 1 | Project structure | ✅ |
| 2 | Parse representatives | ✅ |
| 3 | Parse proposals | ✅ |
| 4 | Parse objections | ✅ |
| 5 | Parse relations CSV | ✅ |
| 6 | Sanitize IDs | ✅ `normalize_id()` |
| 7 | Invalid attribute types (string influence, null) | ✅ `clean_influence()` |
| 8 | Deduplicate proposals | ✅ |
| 9 | Validate ghost references | ✅ |
| 10 | Compute relationship scores | ✅ `weighted_rel_score` |
| 11 | Calculate objection weights | ✅ `severity * influence` |
| 12 | Filter Trojan Horses | ✅ |
| 13 | Reject Poison Pills | ✅ |
| 14 | Identify genuine alliances | ✅ bidirectional |
| 15 | Handle asymmetric trust (False Friends) | ✅ |
| 16 | Faction Infiltrators | ✅ |
| 17 | Formulate consensus output | ✅ |
| 18 | Format JSON output | ✅ `consensus_result.json` |
| 19 | Extreme edge cases | ✅ / ⚠️ Mostly; dirty CSV gap remains |
| 20 | Scale performance (50+ reps) | ✅ vectorized pandas |

---

## Verdict: Fix the 3 Gaps, Then Ready for Submission

> **Current estimated score: ~75-80/100 (Tier A)**
> After fixing 3 gaps: **~88-95/100 (Tier S)**
