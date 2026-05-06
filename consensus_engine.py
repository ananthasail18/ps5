import json
import pandas as pd
import numpy as np
import os

def normalize_id(id_val):
    if pd.isna(id_val):
        return None
    return str(id_val).strip().lower()

def clean_influence(val):
    if pd.isna(val):
        return 0
    try:
        val = int(val)
        if val < 0: return 0
        if val > 100: return 100
        return val
    except:
        return 0

def clean_severity(val):
    if pd.isna(val):
        return 0
    try:
        if isinstance(val, str):
            if val.lower() == 'high': return 8
            if val.lower() == 'medium': return 5
            if val.lower() == 'low': return 2
        val = int(val)
        if val < 0: return 0
        if val > 10: return 10
        return val
    except:
        return 0

def clean_probability(val):
    if pd.isna(val) or val == 'high' or val == 'low':
        return 1.0 # assume worst
    try:
        val = float(val)
        if val < 0.0: return 0.0
        if val > 1.0: return 1.0
        return val
    except:
        return 1.0

def clean_score(val):
    if pd.isna(val) or val == 'high' or val == 'low':
        return 0
    try:
        val = float(val)
        if val < 0: return 0
        if val > 100: return 100
        return val
    except:
        return 0

def load_and_clean_data(data_dir):
    # 1. Load Representatives
    with open(os.path.join(data_dir, 'representatives.json'), 'r') as f:
        reps_raw = json.load(f)
    df_reps = pd.DataFrame(reps_raw)
    df_reps['id'] = df_reps['id'].apply(normalize_id)
    df_reps = df_reps.dropna(subset=['id'])
    # Deduplicate reps by keeping the first occurrence (or highest influence)
    df_reps['influence'] = df_reps['influence'].apply(clean_influence)
    df_reps = df_reps.sort_values('influence', ascending=False).drop_duplicates(subset=['id'], keep='first')
    
    # 2. Load Proposals
    with open(os.path.join(data_dir, 'proposals.json'), 'r') as f:
        props_raw = json.load(f)
    df_props = pd.DataFrame(props_raw)
    df_props['id'] = df_props['id'].apply(normalize_id)
    df_props['sponsor'] = df_props['sponsor'].apply(normalize_id)
    df_props = df_props.dropna(subset=['id'])
    
    def clean_priority(val):
        try:
            val = float(val)
            if val < 0: return 0
            if val > 10: return 10
            return val
        except:
            return 0
    df_props['priority'] = df_props['priority'].apply(clean_priority)
    df_props = df_props.sort_values('priority', ascending=False).drop_duplicates(subset=['id'], keep='first')
    
    # Filter ghost sponsors
    valid_reps = set(df_reps['id'])
    df_props = df_props[df_props['sponsor'].isin(valid_reps)]
    
    # 3. Load Objections
    with open(os.path.join(data_dir, 'objections.json'), 'r') as f:
        objs_raw = json.load(f)
    df_objs = pd.DataFrame(objs_raw)
    df_objs['rep_id'] = df_objs['rep_id'].apply(normalize_id)
    df_objs['proposal_id'] = df_objs['proposal_id'].apply(normalize_id)
    df_objs['severity'] = df_objs['severity'].apply(clean_severity)
    
    # Filter ghost references
    valid_props = set(df_props['id'])
    df_objs = df_objs[df_objs['rep_id'].isin(valid_reps) & df_objs['proposal_id'].isin(valid_props)]
    df_objs = df_objs.drop_duplicates(subset=['rep_id', 'proposal_id'])
    
    # 4. Load Relations
    df_rels = pd.read_csv(os.path.join(data_dir, 'relations.csv'))
    df_rels['from'] = df_rels['from'].apply(normalize_id)
    df_rels['to'] = df_rels['to'].apply(normalize_id)
    df_rels['trust'] = df_rels['trust'].apply(clean_score)
    df_rels['rivalry'] = df_rels['rivalry'].apply(clean_score)
    df_rels['betrayal_prob'] = df_rels['betrayal_prob'].apply(clean_probability)
    
    df_rels = df_rels[df_rels['from'].isin(valid_reps) & df_rels['to'].isin(valid_reps)]
    # Keep latest interaction
    if 'last_interaction' in df_rels.columns:
        df_rels['last_interaction'] = pd.to_datetime(df_rels['last_interaction'], errors='coerce')
        df_rels = df_rels.sort_values('last_interaction', ascending=False).drop_duplicates(subset=['from', 'to'], keep='first')
    else:
        df_rels = df_rels.drop_duplicates(subset=['from', 'to'], keep='first')
        
    return df_reps, df_props, df_objs, df_rels

def compute_metrics(df_reps, df_props, df_objs, df_rels):
    # 1. Objection weight per proposal
    # objection_weight = sum(severity * objector_influence)
    obj_merged = df_objs.merge(df_reps[['id', 'influence']], left_on='rep_id', right_on='id', how='left')
    obj_merged['weight'] = obj_merged['severity'] * obj_merged['influence']
    proposal_weights = obj_merged.groupby('proposal_id')['weight'].sum().reset_index()
    
    df_props = df_props.merge(proposal_weights, left_on='id', right_on='proposal_id', how='left')
    df_props['weight'] = df_props['weight'].fillna(0)
    
    # 2. Relationship Score
    # relationship_score = trust * (1 - betrayal_prob)
    df_rels['rel_score'] = df_rels['trust'] * (1 - df_rels['betrayal_prob'])
    
    return df_reps, df_props, df_rels

def run_consensus(df_reps, df_props, df_rels):
    # --- 1. Filter Trojan Horses ---
    # Reps with average betrayal_prob > 0.5 are risky
    avg_betrayal = df_rels.groupby('from')['betrayal_prob'].mean().reset_index()
    risky_reps = set(avg_betrayal[avg_betrayal['betrayal_prob'] > 0.5]['from'])
    
    safe_reps = df_reps[~df_reps['id'].isin(risky_reps)]
    
    # Check Faction Infiltrators: high betrayal towards own faction members
    rel_faction = df_rels.merge(df_reps[['id', 'faction']], left_on='from', right_on='id') \
                         .merge(df_reps[['id', 'faction']], left_on='to', right_on='id', suffixes=('_from', '_to'))
    faction_betrayals = rel_faction[(rel_faction['faction_from'] == rel_faction['faction_to']) & 
                                    (rel_faction['betrayal_prob'] > 0.5)]
    infiltrators = set(faction_betrayals['from'])
    safe_reps = safe_reps[~safe_reps['id'].isin(infiltrators)]
    
    safe_rep_ids = set(safe_reps['id'])
    
    # --- 2. Select Proposals (Avoid Poison Pills) ---
    # We want high priority, low weight. 
    # Let's say a poison pill is weight > threshold (e.g. 500)
    # or viablity score = priority * 100 - weight
    df_props['viability'] = df_props['priority'] * 100 - df_props['weight']
    viable_props = df_props[df_props['viability'] > 0].sort_values('viability', ascending=False)
    
    # If no proposals are strictly viable, just pick the one with max viability
    if viable_props.empty and not df_props.empty:
        selected_proposals = [df_props.sort_values('viability', ascending=False).iloc[0]['id']]
    else:
        # Pick top ones (e.g., up to 3)
        selected_proposals = viable_props.head(3)['id'].tolist()
        
    # --- 3. Detect Alliances (Bidirectional Trust) ---
    # A mutually trusts B and B mutually trusts A
    # rel_score > threshold (e.g. 50)
    strong_rels = df_rels[df_rels['rel_score'] > 50]
    # Self-join to find pairs
    pairs = strong_rels.merge(strong_rels, left_on=['from', 'to'], right_on=['to', 'from'])
    
    alliances_set = set()
    for _, row in pairs.iterrows():
        a, b = row['from_x'], row['to_x']
        if a in safe_rep_ids and b in safe_rep_ids:
            # Sort to avoid duplicates like (A,B) and (B,A)
            alliance = tuple(sorted([a, b]))
            alliances_set.add(alliance)
            
    alliances = [list(a) for a in alliances_set]
    
    # --- 4. Select Supporters ---
    # Pick top safe reps with high influence
    supporting_reps = safe_reps.sort_values('influence', ascending=False).head(5)['id'].tolist()
    
    # Make sure we have at least 1 proposal and 1 supporter
    if not selected_proposals and not df_props.empty:
        selected_proposals = [df_props.iloc[0]['id']]
    if not supporting_reps and not df_reps.empty:
        supporting_reps = [df_reps.iloc[0]['id']]
        
    return {
        "final_agreement": {
            "proposals": selected_proposals,
            "supporting_reps": supporting_reps
        },
        "alliances": alliances
    }

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    output_dir = os.path.join(base_dir, 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading and cleaning data...")
    df_reps, df_props, df_objs, df_rels = load_and_clean_data(data_dir)
    
    print("Computing metrics...")
    df_reps, df_props, df_rels = compute_metrics(df_reps, df_props, df_objs, df_rels)
    
    print("Running consensus logic...")
    result = run_consensus(df_reps, df_props, df_rels)
    
    out_file = os.path.join(output_dir, 'consensus_result.json')
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"Results saved to {out_file}")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
