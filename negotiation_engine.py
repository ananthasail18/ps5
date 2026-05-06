"""
================================================================================
  NEGOTIATION ENGINE — Person B Module
  Role: Data Scientist | UN Security Council AI Strategist
  
  PHILOSOPHY:
  A 'NO' is never the end of a negotiation. It is an information payload.
  This engine intercepts every veto, decodes its semantic root cause,
  and generates a targeted counter-deal to recover the relationship
  and find a safe passage to consensus.
================================================================================
"""

import json
import os
import numpy as np
import pandas as pd
from consensus_engine import load_and_clean_data, compute_metrics, run_consensus

# ==============================================================================
# SECTION 1: OBJECTION TAXONOMY
# We define the semantic "themes" that represent the root of a NO.
# Each theme maps to a counter-deal strategy.
# ==============================================================================
OBJECTION_TAXONOMY = {
    "FISCAL": {
        "keywords": ["costly", "cost", "fiscal", "budget", "expense", "price", "fund", "afford"],
        "label": "Fiscal Concerns",
        "counter_deal_template": (
            "Counter-Deal: Propose a phased budget rollout. Offer to cap immediate spending "
            "to {concession_pct}% of the original estimate, with a formal spending review "
            "checkpoint after 6 months. This directly addresses the fiscal objection while "
            "keeping the proposal alive."
        ),
    },
    "SECURITY": {
        "keywords": ["security", "risk", "threat", "danger", "breach", "vulnerability", "unsafe"],
        "label": "Security / Risk Concerns",
        "counter_deal_template": (
            "Counter-Deal: Propose an independent security audit by a neutral third-party "
            "committee before full implementation. Offer {objector_name} a seat on the "
            "oversight panel to give them direct control over risk mitigation. This converts "
            "an adversary into a stakeholder."
        ),
    },
    "SCOPE": {
        "keywords": ["scope", "unclear", "vague", "broad", "overreach", "undefined", "ambiguous"],
        "label": "Scope / Clarity Concerns",
        "counter_deal_template": (
            "Counter-Deal: Offer a formal Amendment Clause. Propose a smaller pilot program "
            "limited to a single region or department first. Invite {objector_name} to co-author "
            "the refined scope document. This transforms opposition into co-ownership."
        ),
    },
    "PROCEDURAL": {
        "keywords": ["procedural", "process", "rule", "protocol", "order", "procedure", "review"],
        "label": "Procedural Concerns",
        "counter_deal_template": (
            "Counter-Deal: Propose a formal deferral to the next scheduled session with a "
            "guaranteed agenda slot. Acknowledge the procedural gaps and offer {objector_name} "
            "the role of Procedure Chair for the review committee. This respects their "
            "concerns without killing the proposal."
        ),
    },
    "GENERAL": {
        "keywords": [],
        "label": "General Opposition",
        "counter_deal_template": (
            "Counter-Deal: Request a private bilateral session with {objector_name} to understand "
            "the underlying concern. Offer a mutual benefit clause: if this proposal passes, "
            "support {objector_name}'s next proposal in return. This is a classic alliance-building "
            "quid-pro-quo that protects the relationship."
        ),
    },
}

# ==============================================================================
# SECTION 2: WEIGHTED HEURISTIC SCORING
# The new heuristic penalises betrayal exponentially.
# Formula: weighted_rel_score = (trust/100) * influence * ((1 - betrayal_prob)^2)
# ==============================================================================
def compute_weighted_heuristics(df_reps: pd.DataFrame, df_rels: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Google-style weighted heuristics to produce a rich relationship score.
    Exponential betrayal penalty ensures high-risk actors are mathematically neutralised.
    """
    df_rels = df_rels.copy()

    # Merge influence of the 'from' representative into relations
    df_rels = df_rels.merge(
        df_reps[["id", "influence"]],
        left_on="from",
        right_on="id",
        how="left",
        suffixes=("", "_rep"),
    ).rename(columns={"influence": "from_influence"})

    # Core heuristic formula
    df_rels["weighted_rel_score"] = (
        (df_rels["trust"] / 100)
        * df_rels["from_influence"].fillna(0)
        * ((1 - df_rels["betrayal_prob"]) ** 2)
    )

    # Trojan Horse flag: high influence + high betrayal
    df_rels["is_trojan_horse"] = (
        (df_rels["from_influence"] > 60) & (df_rels["betrayal_prob"] > 0.4)
    )

    return df_rels


# ==============================================================================
# SECTION 3: NLP TAXONOMY CLASSIFIER
# Classifies a raw text reason into a semantic theme without any external library.
# Uses keyword-matching as a lightweight, dependency-free alternative.
# ==============================================================================
def classify_objection(reason_text: str) -> str:
    """
    Classifies the semantic theme of an objection reason using keyword matching.
    Returns the taxonomy key (e.g., 'FISCAL', 'SECURITY').
    """
    if not reason_text or pd.isna(reason_text):
        return "GENERAL"

    reason_lower = str(reason_text).lower()
    for theme, config in OBJECTION_TAXONOMY.items():
        if theme == "GENERAL":
            continue
        for keyword in config["keywords"]:
            if keyword in reason_lower:
                return theme
    return "GENERAL"


# ==============================================================================
# SECTION 4: COUNTER-DEAL GENERATOR
# The core of the "NO → Counter-Deal" logic.
# ==============================================================================
def generate_counter_deal(
    objector_name: str,
    proposal_title: str,
    reason: str,
    budget_estimate: float = None,
) -> dict:
    """
    Given a single objection, this function:
    1. Classifies the semantic theme of the 'NO'.
    2. Generates a targeted, diplomatic counter-deal.
    3. Returns a structured advice payload.
    """
    theme_key = classify_objection(reason)
    theme_config = OBJECTION_TAXONOMY[theme_key]

    # Calculate a budget concession (offer 70% of original if fiscal concern)
    concession_pct = 70
    if budget_estimate:
        reduced_budget = budget_estimate * (concession_pct / 100)
        concession_str = f"${reduced_budget:,.0f} (reduced from ${budget_estimate:,.0f})"
    else:
        concession_str = f"{concession_pct}%"

    counter_deal_text = theme_config["counter_deal_template"].format(
        objector_name=objector_name,
        concession_pct=concession_pct,
        reduced_budget=concession_str,
    )

    # Compute a "danger level" for the objection theme
    danger_map = {"FISCAL": "HIGH", "SECURITY": "CRITICAL", "SCOPE": "MEDIUM", "PROCEDURAL": "LOW", "GENERAL": "MEDIUM"}
    danger_level = danger_map.get(theme_key, "MEDIUM")

    return {
        "objector": objector_name,
        "proposal": proposal_title,
        "raw_objection": reason,
        "classified_theme": theme_config["label"],
        "danger_level": danger_level,
        "nlp_justification": (
            f"The objection '{reason}' was classified under the '{theme_config['label']}' "
            f"semantic cluster. This is a '{danger_level}' danger signal. The counter-deal "
            f"is designed to directly address the root cause of this cluster."
        ),
        "counter_deal": counter_deal_text,
    }


# ==============================================================================
# SECTION 5: TROJAN HORSE DIPLOMATIC HANDLER
# Instead of silently dropping a Trojan Horse, we generate a de-escalation strategy.
# ==============================================================================
def handle_trojan_horse(rep_name: str, influence: float, avg_betrayal: float) -> dict:
    """
    When a Trojan Horse is detected, generate a safe containment strategy
    instead of simply excluding them (which could create an enemy).
    """
    return {
        "representative": rep_name,
        "risk_profile": "TROJAN HORSE",
        "influence": influence,
        "average_betrayal_probability": round(avg_betrayal, 3),
        "nlp_justification": (
            f"{rep_name} has a weighted influence of {influence} and an average betrayal "
            f"probability of {avg_betrayal:.0%}. This profile matches the classic Trojan Horse "
            f"pattern: high enough influence to appear essential, but structurally unreliable."
        ),
        "containment_strategy": (
            f"Do NOT confront or exclude {rep_name} openly — this creates a hostile actor. "
            f"Instead: (1) Give them a symbolic committee role with no real power. "
            f"(2) Ensure no classified proposal details are shared with them. "
            f"(3) Build a parallel alliance with their rivals to maintain a majority "
            f"even if {rep_name} defects. This keeps them 'inside the tent' while "
            f"neutralising their threat."
        ),
    }


# ==============================================================================
# SECTION 6: MASTER ORCHESTRATOR — Full Negotiation Pipeline
# ==============================================================================
def run_negotiation_pipeline(data_dir: str, output_dir: str):
    print("\n" + "=" * 60)
    print("  CONSENSUS + NEGOTIATION ENGINE — Starting Pipeline")
    print("=" * 60)

    # --- Stage 1: Load & Clean ---
    print("\n[Stage 1] Loading and cleaning data...")
    df_reps, df_props, df_objs, df_rels = load_and_clean_data(data_dir)

    # --- Stage 2: Heuristic Scoring ---
    print("[Stage 2] Computing weighted heuristics...")
    df_rels_scored = compute_weighted_heuristics(df_reps, df_rels)
    df_reps, df_props, df_rels = compute_metrics(df_reps, df_props, df_objs, df_rels)

    # --- Stage 3: Trojan Horse Detection with Containment ---
    print("[Stage 3] Detecting Trojan Horses...")
    avg_betrayal = df_rels_scored.groupby("from")["betrayal_prob"].mean().reset_index()
    avg_betrayal.columns = ["id", "avg_betrayal"]
    df_reps_risk = df_reps.merge(avg_betrayal, on="id", how="left")
    df_reps_risk["avg_betrayal"] = df_reps_risk["avg_betrayal"].fillna(0)

    trojan_horse_reports = []
    trojan_horse_ids = set()
    for _, row in df_reps_risk.iterrows():
        if row["influence"] > 60 and row["avg_betrayal"] > 0.4:
            trojan_horse_ids.add(row["id"])
            report = handle_trojan_horse(
                rep_name=row.get("name", row["id"]),
                influence=row["influence"],
                avg_betrayal=row["avg_betrayal"],
            )
            trojan_horse_reports.append(report)
            print(f"  [!!] TROJAN HORSE DETECTED: {row.get('name', row['id'])}")

    # --- Stage 4: Baseline Consensus ---
    print("[Stage 4] Running baseline consensus logic...")
    consensus = run_consensus(df_reps, df_props, df_rels)

    # --- Stage 5: Counter-Deal Generation for all NOes ---
    print("[Stage 5] Generating counter-deals for all objections...")
    rep_name_map = dict(zip(df_reps["id"], df_reps.get("name", df_reps["id"])))
    prop_title_map = dict(zip(df_props["id"], df_props.get("title", df_props["id"])))
    prop_budget_map = {}
    if "budget_estimate" in df_props.columns:
        prop_budget_map = dict(zip(df_props["id"], df_props["budget_estimate"]))

    negotiation_log = []
    for _, obj_row in df_objs.iterrows():
        rep_name = rep_name_map.get(obj_row["rep_id"], obj_row["rep_id"])
        prop_title = prop_title_map.get(obj_row["proposal_id"], obj_row["proposal_id"])
        budget = prop_budget_map.get(obj_row["proposal_id"], None)

        deal = generate_counter_deal(
            objector_name=rep_name,
            proposal_title=prop_title,
            reason=obj_row.get("reason", ""),
            budget_estimate=budget,
        )
        negotiation_log.append(deal)

    # --- Stage 6: Assemble Final Output ---
    final_output = {
        "consensus": consensus,
        "trojan_horse_containment": trojan_horse_reports,
        "negotiation_log": negotiation_log,
        "pipeline_summary": {
            "total_objections_handled": len(negotiation_log),
            "trojan_horses_detected": len(trojan_horse_reports),
            "proposals_in_consensus": len(consensus.get("final_agreement", {}).get("proposals", [])),
            "supporting_reps": consensus.get("final_agreement", {}).get("supporting_reps", []),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "negotiation_result.json")
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"\n[OK] Final negotiation output saved to: {out_path}")
    print("\n--- PIPELINE SUMMARY ---")
    print(json.dumps(final_output["pipeline_summary"], indent=2))

    print("\n--- NEGOTIATION LOG (Counter-Deals) ---")
    for deal in negotiation_log:
        print(f"\n  Objector : {deal['objector']}")
        print(f"  Proposal : {deal['proposal']}")
        print(f"  Their NO : \"{deal['raw_objection']}\"")
        print(f"  Theme    : {deal['classified_theme']} [{deal['danger_level']}]")
        print(f"  NLP Why  : {deal['nlp_justification']}")
        print(f"  Our Move : {deal['counter_deal']}")

    print("\n" + "=" * 60)
    print("  NEGOTIATION ENGINE — Pipeline Complete")
    print("=" * 60 + "\n")

    return final_output


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "raw")
    output_dir = os.path.join(base_dir, "output")
    run_negotiation_pipeline(data_dir, output_dir)
