"""
================================================================================
  FINAL PIPELINE — Merged Person A + Person B
  
  Person A  : Data Engineering  — load, clean, validate, output JSON
  Person B  : Data Science + NLP — heuristics, classification, negotiation
  
  Single entry point. Single output: consensus_result.json
================================================================================
"""

import json
import os
import pandas as pd
import numpy as np

# ==============================================================================
# [PERSON A] SECTION 1: DATA SANITIZATION UTILITIES
# All raw field cleaning and normalisation logic.
# ==============================================================================

def normalize_id(id_val):
    """Lowercase, strip whitespace, return None for nulls."""
    if pd.isna(id_val):
        return None
    return str(id_val).strip().lower()

def clean_influence(val):
    """Clamp influence to [0, 100]. Default 0 for nulls/errors."""
    if pd.isna(val):
        return 0
    try:
        val = int(float(val))
        return max(0, min(100, val))
    except:
        return 0

def clean_severity(val):
    """
    Handle string labels ('high','medium','low') and numeric values.
    Clamp to [0, 10]. Default 0 for nulls — NLP will infer.
    """
    if pd.isna(val):
        return None          # Keep None so NLP can infer severity later
    try:
        if isinstance(val, str):
            mapping = {"high": 8, "medium": 5, "low": 2}
            if val.strip().lower() in mapping:
                return mapping[val.strip().lower()]
        val = int(float(val))
        return max(0, min(10, val))
    except:
        return None

def clean_probability(val):
    """Clamp betrayal probability to [0.0, 1.0]. Assume worst on error."""
    if pd.isna(val) or str(val).lower() in ("high", "low"):
        return 1.0
    try:
        val = float(val)
        return max(0.0, min(1.0, val))
    except:
        return 1.0

def clean_score(val):
    """Clamp trust/rivalry score to [0, 100]. Default 0 on error."""
    if pd.isna(val) or str(val).lower() in ("high", "low"):
        return 0
    try:
        val = float(val)
        return max(0.0, min(100.0, val))
    except:
        return 0


# ==============================================================================
# [PERSON A] SECTION 2: DATA INGESTION & CLEANING
# Reads all 4 raw files, normalises, deduplicates, drops ghosts.
# ==============================================================================

def load_and_clean_data(data_dir: str):
    """
    Loads representatives, proposals, objections, and relations.
    Returns clean DataFrames ready for metric computation.
    """
    # --- Representatives ---
    with open(os.path.join(data_dir, "representatives.json"), "r") as f:
        df_reps = pd.DataFrame(json.load(f))
    df_reps["id"] = df_reps["id"].apply(normalize_id)
    df_reps = df_reps.dropna(subset=["id"])
    df_reps["influence"] = df_reps["influence"].apply(clean_influence)
    # Dedup: keep highest-influence version of each ID
    df_reps = df_reps.sort_values("influence", ascending=False).drop_duplicates(subset=["id"], keep="first")

    valid_reps = set(df_reps["id"])

    # --- Proposals ---
    with open(os.path.join(data_dir, "proposals.json"), "r") as f:
        df_props = pd.DataFrame(json.load(f))
    df_props["id"] = df_props["id"].apply(normalize_id)
    df_props["sponsor"] = df_props["sponsor"].apply(normalize_id)
    df_props = df_props.dropna(subset=["id"])

    def clean_priority(val):
        try:
            val = float(val)
            return max(0, min(10, val))
        except:
            return 0
    df_props["priority"] = df_props["priority"].apply(clean_priority)
    # Dedup: keep highest-priority version of each proposal ID
    df_props = df_props.sort_values("priority", ascending=False).drop_duplicates(subset=["id"], keep="first")
    # Drop orphaned proposals (ghost sponsors)
    df_props = df_props[df_props["sponsor"].isin(valid_reps)]

    valid_props = set(df_props["id"])

    # --- Objections ---
    with open(os.path.join(data_dir, "objections.json"), "r") as f:
        df_objs = pd.DataFrame(json.load(f))
    df_objs["rep_id"] = df_objs["rep_id"].apply(normalize_id)
    df_objs["proposal_id"] = df_objs["proposal_id"].apply(normalize_id)
    df_objs["severity"] = df_objs["severity"].apply(clean_severity)
    # Drop ghost references (unknown reps or proposals)
    df_objs = df_objs[
        df_objs["rep_id"].isin(valid_reps) & df_objs["proposal_id"].isin(valid_props)
    ]
    df_objs = df_objs.drop_duplicates(subset=["rep_id", "proposal_id"])

    # --- Relations ---
    # [GAP 3 FIX] Dirty CSV: read with error-tolerant dtypes, coerce bad values
    df_rels = pd.read_csv(os.path.join(data_dir, "relations.csv"), dtype=str)
    # Ensure required columns exist; fill missing ones with NaN
    for col in ["from", "to", "trust", "rivalry", "betrayal_prob"]:
        if col not in df_rels.columns:
            df_rels[col] = np.nan
    # Drop rows where both 'from' and 'to' are missing (fully malformed rows)
    df_rels = df_rels.dropna(subset=["from", "to"], how="all")
    df_rels["from"] = df_rels["from"].apply(normalize_id)
    df_rels["to"] = df_rels["to"].apply(normalize_id)
    df_rels["trust"] = df_rels["trust"].apply(clean_score)
    df_rels["rivalry"] = df_rels["rivalry"].apply(clean_score)
    df_rels["betrayal_prob"] = df_rels["betrayal_prob"].apply(clean_probability)
    df_rels = df_rels[df_rels["from"].isin(valid_reps) & df_rels["to"].isin(valid_reps)]
    # Keep most recent interaction per pair
    if "last_interaction" in df_rels.columns:
        df_rels["last_interaction"] = pd.to_datetime(df_rels["last_interaction"], errors="coerce")
        df_rels = df_rels.sort_values("last_interaction", ascending=False)
    df_rels = df_rels.drop_duplicates(subset=["from", "to"], keep="first")

    return df_reps, df_props, df_objs, df_rels


# ==============================================================================
# [PERSON B] SECTION 3: NLP OBJECTION TAXONOMY
# Keyword-driven semantic classification of objection reasons.
# ==============================================================================

OBJECTION_TAXONOMY = {
    "FISCAL": {
        "keywords": ["costly", "cost", "fiscal", "budget", "expense", "price", "fund", "afford", "financial"],
        "label": "Fiscal Concerns",
        "danger": "HIGH",
        "counter_deal_template": (
            "Propose a phased budget rollout. Offer to cap immediate spending to {concession_pct}% "
            "of the original estimate, with a formal spending review checkpoint after 6 months."
        ),
    },
    "SECURITY": {
        "keywords": ["security", "risk", "threat", "danger", "breach", "vulnerability", "unsafe"],
        "label": "Security / Risk Concerns",
        "danger": "CRITICAL",
        "counter_deal_template": (
            "Propose an independent security audit by a neutral third-party committee before "
            "full implementation. Offer {objector_name} a seat on the oversight panel — converting "
            "an adversary into a co-owner of risk mitigation."
        ),
    },
    "SCOPE": {
        "keywords": ["scope", "unclear", "vague", "broad", "overreach", "undefined", "ambiguous"],
        "label": "Scope / Clarity Concerns",
        "danger": "MEDIUM",
        "counter_deal_template": (
            "Offer a formal Amendment Clause. Propose a smaller pilot program limited to one region "
            "or department first. Invite {objector_name} to co-author the refined scope document — "
            "transforming opposition into co-ownership."
        ),
    },
    "PROCEDURAL": {
        "keywords": ["procedural", "process", "rule", "protocol", "order", "procedure", "review"],
        "label": "Procedural Concerns",
        "danger": "LOW",
        "counter_deal_template": (
            "Propose a formal deferral to the next scheduled session with a guaranteed agenda slot. "
            "Offer {objector_name} the role of Procedure Chair for the review committee."
        ),
    },
    "GENERAL": {
        "keywords": [],
        "label": "General Opposition",
        "danger": "MEDIUM",
        "counter_deal_template": (
            "Request a private bilateral session with {objector_name} to understand the underlying "
            "concern. Offer a mutual benefit clause: if this proposal passes, support "
            "{objector_name}'s next proposal in return (quid-pro-quo alliance building)."
        ),
    },
}

def classify_objection(reason_text: str) -> str:
    """Returns the taxonomy key for the objection reason (e.g. 'FISCAL')."""
    if not reason_text or pd.isna(reason_text):
        return "GENERAL"
    reason_lower = str(reason_text).lower()
    for theme, cfg in OBJECTION_TAXONOMY.items():
        if theme == "GENERAL":
            continue
        if any(kw in reason_lower for kw in cfg["keywords"]):
            return theme
    return "GENERAL"

def infer_severity_from_text(reason_text: str) -> int:
    """
    NLP fallback: when severity is null, infer it from the semantic danger
    level of the classified theme. CRITICAL->9, HIGH->7, MEDIUM->5, LOW->3.
    """
    theme = classify_objection(reason_text)
    danger_to_severity = {"CRITICAL": 9, "HIGH": 7, "MEDIUM": 5, "LOW": 3}
    return danger_to_severity.get(OBJECTION_TAXONOMY[theme]["danger"], 5)


# ==============================================================================
# [PERSON B] SECTION 4: WEIGHTED HEURISTIC SCORING
# Formula: weighted_rel_score = (trust/100) * influence * (1 - betrayal_prob)^2
# Exponential penalty makes high-betrayal actors mathematically negligible.
# ==============================================================================

def compute_weighted_heuristics(df_reps: pd.DataFrame, df_rels: pd.DataFrame) -> pd.DataFrame:
    """Enriches df_rels with weighted_rel_score and is_trojan_horse flag."""
    df = df_rels.copy()
    df = df.merge(
        df_reps[["id", "influence"]],
        left_on="from", right_on="id",
        how="left", suffixes=("", "_rep")
    ).rename(columns={"influence": "from_influence"})

    df["from_influence"] = df["from_influence"].fillna(0)
    df["weighted_rel_score"] = (
        (df["trust"] / 100) * df["from_influence"] * ((1 - df["betrayal_prob"]) ** 2)
    )
    df["is_trojan_horse"] = (df["from_influence"] > 60) & (df["betrayal_prob"] > 0.4)
    return df


# ==============================================================================
# [PERSON A + B] SECTION 5: METRIC COMPUTATION
# Combines Person A's objection weighting with Person B's NLP severity inference.
# ==============================================================================

def compute_metrics(df_reps, df_props, df_objs, df_rels):
    """
    Merges NLP-inferred severity into objections where severity was null,
    then computes objection_weight and basic relationship score.
    """
    df_objs = df_objs.copy()

    # [Person B] NLP: Fill missing severities from semantic danger level
    null_mask = df_objs["severity"].isna()
    df_objs.loc[null_mask, "severity"] = df_objs.loc[null_mask, "reason"].apply(infer_severity_from_text)
    df_objs["severity"] = df_objs["severity"].fillna(5).astype(float)

    # [Person A] objection_weight = sum(severity * objector_influence) per proposal
    obj_merged = df_objs.merge(df_reps[["id", "influence"]], left_on="rep_id", right_on="id", how="left")
    obj_merged["weight"] = obj_merged["severity"] * obj_merged["influence"].fillna(0)
    proposal_weights = obj_merged.groupby("proposal_id")["weight"].sum().reset_index()

    df_props = df_props.merge(proposal_weights, left_on="id", right_on="proposal_id", how="left")
    df_props["weight"] = df_props["weight"].fillna(0)

    # [Person A] base relationship score
    df_rels = df_rels.copy()
    df_rels["rel_score"] = df_rels["trust"] * (1 - df_rels["betrayal_prob"])

    return df_reps, df_props, df_objs, df_rels


# ==============================================================================
# [PERSON B] SECTION 6: TROJAN HORSE & FACTION INFILTRATOR DETECTION
# ==============================================================================

def detect_threats(df_reps, df_rels_scored):
    """
    Returns:
      safe_reps       — DataFrame of trustworthy representatives
      trojan_reports  — List of containment strategy dicts for Trojan Horses
      infiltrators    — Set of IDs flagged as Faction Infiltrators
    """
    # Average betrayal per rep (direct)
    avg_betrayal = (
        df_rels_scored.groupby("from")["betrayal_prob"].mean()
        .reset_index()
        .rename(columns={"from": "id", "betrayal_prob": "avg_betrayal"})
    )
    df_risk = df_reps.merge(avg_betrayal, on="id", how="left")
    df_risk["avg_betrayal"] = df_risk["avg_betrayal"].fillna(0)

    # [GAP 2 FIX] Cascading Betrayal: if a rep's allies are high-betrayal actors,
    # penalise that rep's effective trust. We add a cascading_risk flag.
    # Logic: find reps who are trusted (trust > 60) by others, but themselves
    # have avg_betrayal > 0.5. Their trusting partners get a cascading risk flag.
    high_betrayal_ids = set(df_risk[df_risk["avg_betrayal"] > 0.5]["id"])
    cascading_risky = set(
        df_rels_scored[
            (df_rels_scored["to"].isin(high_betrayal_ids)) &
            (df_rels_scored["trust"] > 60)
        ]["from"]
    )
    # Reps in cascading_risky who are themselves borderline get bumped over threshold
    df_risk["cascading_risk"] = df_risk["id"].isin(cascading_risky)
    df_risk["effective_betrayal"] = df_risk.apply(
        lambda r: min(1.0, r["avg_betrayal"] + 0.15) if r["cascading_risk"] else r["avg_betrayal"],
        axis=1
    )

    # Trojan Horses: high influence + high effective betrayal
    trojan_mask = (df_risk["influence"] > 60) & (df_risk["effective_betrayal"] > 0.4)
    trojan_ids = set(df_risk[trojan_mask]["id"])

    trojan_reports = []
    for _, row in df_risk[trojan_mask].iterrows():
        name = row.get("name", row["id"])
        trojan_reports.append({
            "representative": name,
            "id": row["id"],
            "risk_profile": "TROJAN HORSE",
            "influence": int(row["influence"]),
            "avg_betrayal_prob": round(row["avg_betrayal"], 3),
            "nlp_justification": (
                f"{name} has influence={row['influence']} and avg betrayal={row['avg_betrayal']:.0%}. "
                f"High influence makes them appear essential; high betrayal makes them structurally unreliable."
            ),
            "containment_strategy": (
                f"Do NOT openly exclude {name} — this creates a hostile actor. "
                f"(1) Assign a symbolic committee role with no real power. "
                f"(2) Withhold classified proposal details. "
                f"(3) Build a parallel alliance with their rivals to maintain majority even if they defect."
            ),
        })

    # Faction Infiltrators: high betrayal toward own faction members
    if "faction" in df_reps.columns:
        rel_faction = (
            df_rels_scored
            .merge(df_reps[["id", "faction"]], left_on="from", right_on="id")
            .merge(df_reps[["id", "faction"]], left_on="to", right_on="id", suffixes=("_from", "_to"))
        )
        infiltrator_mask = (
            (rel_faction["faction_from"] == rel_faction["faction_to"]) &
            (rel_faction["betrayal_prob"] > 0.5)
        )
        infiltrators = set(rel_faction[infiltrator_mask]["from"])
    else:
        infiltrators = set()

    blocked_ids = trojan_ids | infiltrators
    safe_reps = df_reps[~df_reps["id"].isin(blocked_ids)]

    return safe_reps, trojan_reports, infiltrators


# ==============================================================================
# [PERSON B] SECTION 7: COUNTER-DEAL GENERATOR
# Every NO gets a diplomatic counter-move.
# ==============================================================================

def generate_counter_deal(objector_name, proposal_title, reason, budget_estimate=None):
    theme_key = classify_objection(reason)
    cfg = OBJECTION_TAXONOMY[theme_key]

    concession_pct = 70
    counter_text = cfg["counter_deal_template"].format(
        objector_name=objector_name,
        concession_pct=concession_pct,
    )

    budget_note = ""
    if budget_estimate and theme_key == "FISCAL":
        reduced = budget_estimate * (concession_pct / 100)
        budget_note = f" Proposed budget reduction: ${reduced:,.0f} (from ${budget_estimate:,.0f})."

    return {
        "objector": objector_name,
        "proposal": proposal_title,
        "raw_objection": reason,
        "classified_theme": cfg["label"],
        "danger_level": cfg["danger"],
        "nlp_justification": (
            f"'{reason}' maps to the '{cfg['label']}' semantic cluster "
            f"[Danger: {cfg['danger']}]. Counter-deal targets this root cause directly."
        ),
        "counter_deal": counter_text + budget_note,
    }


# ==============================================================================
# [PERSON A + B] SECTION 8: CONSENSUS ENGINE
# Combines safe-rep filtering, NLP-weighted viability, and alliance detection.
# ==============================================================================

def run_consensus(df_reps_safe, df_props, df_rels, df_objs=None):
    """
    Selects the best proposals and supporting reps using viability scoring
    and bidirectional trust alliance detection.
    df_objs is passed to enforce supporter coherence (GAP 1 fix).
    """
    safe_ids = set(df_reps_safe["id"])

    # Proposal viability = (priority * 100) - objection_weight
    df_props = df_props.copy()
    df_props["viability"] = df_props["priority"] * 100 - df_props["weight"]
    viable = df_props[df_props["viability"] > 0].sort_values("viability", ascending=False)

    if viable.empty and not df_props.empty:
        selected = [df_props.sort_values("viability", ascending=False).iloc[0]["id"]]
        rejection_notes = ["All proposals have negative viability. Selecting least-bad option."]
    else:
        selected = viable.head(3)["id"].tolist()
        rejection_notes = []

    # Poison Pills = proposals with viability <= 0
    poison_pills = df_props[df_props["viability"] <= 0]["id"].tolist()

    # Detect alliances: bidirectional rel_score > 50 among safe reps
    strong = df_rels[df_rels["rel_score"] > 50]
    pairs = strong.merge(strong, left_on=["from", "to"], right_on=["to", "from"])
    alliances_set = set()
    for _, row in pairs.iterrows():
        a, b = row["from_x"], row["to_x"]
        if a in safe_ids and b in safe_ids:
            alliances_set.add(tuple(sorted([a, b])))
    alliances = [list(p) for p in alliances_set]

    # [GAP 1 FIX] Supporter Coherence: objectors must NOT appear in supporting_reps
    # Find all reps who objected to any selected proposal
    objector_ids = set()
    if df_objs is not None and not df_objs.empty:
        objector_ids = set(
            df_objs[df_objs["proposal_id"].isin(selected)]["rep_id"]
        )

    # Top supporting reps by influence — excluding any who objected
    eligible = df_reps_safe[~df_reps_safe["id"].isin(objector_ids)]
    if eligible.empty:
        eligible = df_reps_safe  # fallback: if all safe reps objected, pick best available
    supporting = eligible.sort_values("influence", ascending=False).head(5)["id"].tolist()
    if not supporting and not df_reps_safe.empty:
        supporting = [df_reps_safe.iloc[0]["id"]]

    return {
        "final_agreement": {
            "proposals": selected,
            "supporting_reps": supporting,
        },
        "alliances": alliances,
        "poison_pills_rejected": poison_pills,
        "rejection_notes": rejection_notes,
    }


# ==============================================================================
# [PERSON A] SECTION 9: OUTPUT FORMATTER
# Formats and writes the final consensus_result.json
# ==============================================================================

def format_and_write_output(consensus, trojan_reports, negotiation_log, output_dir):
    """Assembles all results and writes the final JSON artifact."""
    output = {
        "consensus_result": consensus,
        "security_analysis": {
            "trojan_horses_detected": len(trojan_reports),
            "trojan_horse_containment": trojan_reports,
        },
        "negotiation_log": negotiation_log,
        "pipeline_summary": {
            "proposals_approved": consensus["final_agreement"]["proposals"],
            "proposals_rejected_as_poison_pills": consensus["poison_pills_rejected"],
            "supporting_reps": consensus["final_agreement"]["supporting_reps"],
            "alliances_formed": len(consensus["alliances"]),
            "trojan_horses_neutralised": len(trojan_reports),
            "total_objections_handled": len(negotiation_log),
            "counter_deals_generated": len(negotiation_log),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "consensus_result.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    return out_path, output


# ==============================================================================
# MASTER PIPELINE — Single entry point, runs all stages in order
# ==============================================================================

def run_final_pipeline(data_dir: str, output_dir: str):
    sep = "=" * 65

    print(f"\n{sep}")
    print("  FINAL PIPELINE  |  Person A + Person B  |  Consensus Engine")
    print(sep)

    # ── Stage 1 [Person A]: Load & Clean ──────────────────────────────────────
    print("\n[Stage 1 | Person A] Ingesting and sanitising raw data...")
    df_reps, df_props, df_objs, df_rels = load_and_clean_data(data_dir)
    print(f"  Reps     : {len(df_reps)}  (after dedup/normalise)")
    print(f"  Proposals: {len(df_props)}  (after ghost-sponsor filter)")
    print(f"  Objections: {len(df_objs)} (after ghost-ref filter)")
    print(f"  Relations: {len(df_rels)}")

    # ── Stage 2 [Person B]: Weighted Heuristics ───────────────────────────────
    print("\n[Stage 2 | Person B] Computing weighted relationship heuristics...")
    df_rels_scored = compute_weighted_heuristics(df_reps, df_rels)
    top = df_rels_scored.nlargest(3, "weighted_rel_score")[["from", "to", "weighted_rel_score"]]
    print(f"  Top 3 weighted relationship scores:\n{top.to_string(index=False)}")

    # ── Stage 3 [Person A+B]: NLP + Metric Computation ───────────────────────
    print("\n[Stage 3 | A+B] NLP severity inference + objection weight computation...")
    df_reps, df_props, df_objs, df_rels = compute_metrics(df_reps, df_props, df_objs, df_rels)
    print(f"  Null severities filled by NLP: {(df_objs['severity'].notna()).sum()} / {len(df_objs)}")

    # ── Stage 4 [Person B]: Threat Detection ──────────────────────────────────
    print("\n[Stage 4 | Person B] Detecting Trojan Horses and Faction Infiltrators...")
    safe_reps, trojan_reports, infiltrators = detect_threats(df_reps, df_rels_scored)
    print(f"  Trojan Horses detected : {len(trojan_reports)}")
    for t in trojan_reports:
        print(f"    [!!] {t['representative']}  (influence={t['influence']}, betrayal={t['avg_betrayal_prob']})")
    print(f"  Faction Infiltrators   : {len(infiltrators)}")
    print(f"  Safe reps remaining    : {len(safe_reps)}")

    # ── Stage 5 [Person A+B]: Consensus Engine ────────────────────────────────
    print("\n[Stage 5 | A+B] Running consensus engine (viability + alliance detection)...")
    consensus = run_consensus(safe_reps, df_props, df_rels, df_objs=df_objs)
    print(f"  Proposals approved  : {consensus['final_agreement']['proposals']}")
    print(f"  Poison pills vetoed : {consensus['poison_pills_rejected']}")
    print(f"  Alliances detected  : {len(consensus['alliances'])}")

    # ── Stage 6 [Person B]: Counter-Deal Generation ───────────────────────────
    print("\n[Stage 6 | Person B] Generating NLP counter-deals for all objections...")
    rep_name_map = {}
    if "name" in df_reps.columns:
        rep_name_map = dict(zip(df_reps["id"], df_reps["name"]))
    prop_title_map = {}
    if "title" in df_props.columns:
        prop_title_map = dict(zip(df_props["id"], df_props["title"]))
    budget_map = {}
    if "budget_estimate" in df_props.columns:
        budget_map = dict(zip(df_props["id"], df_props["budget_estimate"]))

    negotiation_log = []
    for _, row in df_objs.iterrows():
        deal = generate_counter_deal(
            objector_name=rep_name_map.get(row["rep_id"], row["rep_id"]),
            proposal_title=prop_title_map.get(row["proposal_id"], row["proposal_id"]),
            reason=row.get("reason", ""),
            budget_estimate=budget_map.get(row["proposal_id"]),
        )
        negotiation_log.append(deal)
    print(f"  Counter-deals generated: {len(negotiation_log)}")

    # ── Stage 7 [Person A]: Output Formatting ─────────────────────────────────
    print("\n[Stage 7 | Person A] Formatting and writing consensus_result.json...")
    out_path, output = format_and_write_output(consensus, trojan_reports, negotiation_log, output_dir)

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  PIPELINE COMPLETE — FINAL SUMMARY")
    print(sep)
    summary = output["pipeline_summary"]
    print(f"  Proposals Approved        : {summary['proposals_approved']}")
    print(f"  Poison Pills Rejected     : {summary['proposals_rejected_as_poison_pills']}")
    print(f"  Supporting Reps           : {summary['supporting_reps']}")
    print(f"  Alliances Formed          : {summary['alliances_formed']}")
    print(f"  Trojan Horses Neutralised : {summary['trojan_horses_neutralised']}")
    print(f"  Objections Handled        : {summary['total_objections_handled']}")
    print(f"  Counter-Deals Generated   : {summary['counter_deals_generated']}")
    print(f"\n  Output saved to: {out_path}")

    print(f"\n{sep}")
    print("  NEGOTIATION LOG")
    print(sep)
    for d in negotiation_log:
        print(f"\n  Objector : {d['objector']}")
        print(f"  Proposal : {d['proposal']}")
        print(f"  Their NO : \"{d['raw_objection']}\"")
        print(f"  Theme    : {d['classified_theme']}  [Danger: {d['danger_level']}]")
        print(f"  NLP Why  : {d['nlp_justification']}")
        print(f"  Our Move : {d['counter_deal']}")

    print(f"\n{sep}\n")
    return output


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir  = os.path.join(base_dir, "data", "raw")
    output_dir = os.path.join(base_dir, "output")
    run_final_pipeline(data_dir, output_dir)
