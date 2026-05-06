import streamlit as st
import json
import pandas as pd
import os

st.set_page_config(page_title="Phantom Consensus Dashboard", layout="wide", page_icon="🕵️")
st.title("🕵️ Phantom Consensus Engine — Final Dashboard")
st.markdown("*Full A+B Pipeline | NLP Taxonomy | Cascading Betrayal | Negotiation Engine*")

base_dir = os.path.dirname(os.path.abspath(__file__))
out_file = os.path.join(base_dir, 'output', 'consensus_result.json')

if not os.path.exists(out_file):
    st.warning("No output found. Run `python consensus_engine.py` first.")
    st.stop()

with open(out_file, 'r') as f:
    data = json.load(f)

# Support both old and new schema shapes
consensus = data.get("consensus_result", data)
summary = data.get("pipeline_summary", {})
trojan_list = data.get("security_analysis", {}).get("trojan_horse_containment", [])
neg_log = data.get("negotiation_log", [])

# ── Row 1: Summary Metrics ────────────────────────────────────────────────────
st.header("📊 Pipeline Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("✅ Proposals Approved", len(consensus.get("final_agreement", {}).get("proposals", [])))
m2.metric("☠️ Poison Pills Vetoed", len(consensus.get("poison_pills_rejected", [])))
m3.metric("🤝 Alliances Formed", summary.get("alliances_formed", len(consensus.get("alliances", []))))
m4.metric("🐴 Trojans Neutralised", summary.get("trojan_horses_neutralised", len(trojan_list)))
m5.metric("💬 Counter-Deals", summary.get("counter_deals_generated", len(neg_log)))

st.divider()

# ── Row 2: Final Agreement ────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📜 Approved Proposals")
    for p in consensus.get("final_agreement", {}).get("proposals", []):
        st.success(f"✅ **{p}**")
    if consensus.get("poison_pills_rejected"):
        st.subheader("❌ Poison Pills Vetoed")
        for p in consensus["poison_pills_rejected"]:
            st.error(f"🚫 {p}")

with col2:
    st.subheader("🛡️ Supporting Reps")
    for r in consensus.get("final_agreement", {}).get("supporting_reps", []):
        st.info(f"👤 **{r}**")

with col3:
    st.subheader("🔗 Alliances Detected")
    if not consensus.get("alliances"):
        st.warning("No strong bidirectional alliances found.")
    for a in consensus.get("alliances", []):
        st.markdown(f"**`{a[0]}`** ↔ **`{a[1]}`**")

st.divider()

# ── Row 3: Trojan Horse Containment ───────────────────────────────────────────
st.subheader("🐴 Trojan Horse Containment Reports")
if not trojan_list:
    st.success("No Trojan Horses detected in this dataset.")
else:
    for t in trojan_list:
        with st.expander(f"⚠️ {t.get('representative', t.get('id', 'Unknown'))} — {t.get('risk_profile', 'TROJAN HORSE')}"):
            c1, c2 = st.columns(2)
            c1.metric("Influence", t.get("influence", "N/A"))
            c2.metric("Avg Betrayal Prob", t.get("avg_betrayal_prob", "N/A"))
            st.markdown(f"**NLP Justification:** {t.get('nlp_justification', '')}")
            st.info(f"**Containment Strategy:** {t.get('containment_strategy', '')}")

st.divider()

# ── Row 4: Negotiation Log ────────────────────────────────────────────────────
st.subheader("💬 Negotiation Log — Counter-Deals")
if not neg_log:
    st.info("No objections required negotiation.")
else:
    for deal in neg_log:
        danger = deal.get("danger_level", "MEDIUM")
        color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(danger, "⚪")
        with st.expander(f"{color} **{deal.get('objector')}** objected to **{deal.get('proposal')}** [{deal.get('classified_theme')}]"):
            st.markdown(f"**Their Objection:** *\"{deal.get('raw_objection')}\"*")
            st.markdown(f"**NLP Analysis:** {deal.get('nlp_justification', '')}")
            st.success(f"**Our Counter-Deal:** {deal.get('counter_deal', '')}")
