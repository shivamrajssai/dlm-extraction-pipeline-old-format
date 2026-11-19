"""
LHS/RHS hierarchy coordinator and display defect ID generator.
Ported from Databricks DLM Extraction Pipeline (Cell 14).

Coordinates:
  - Parent/child assembly across all defects
  - Cracking/ravelling top-level ID linking
  - Display defect ID assignment (D/{DATE}/N/{05d} and {parent}/{04d})
"""

from typing import Dict, Any, List, Optional
import re
import pandas as pd
import numpy as np

from src.common.constants import (
    CHILD_10M,
    CHILD_100M,
    PARENT_LENGTH_M,
    TABLE_SPEC,
    DEFECTS_LIN,
    ALL_DEFECTS,
    has_parent,
    t_parent,
    parent_len,
    child_tables,
    child_kind,
    t_top,
    all_tables,
    ca_iv,
    SRC_10M,
    SRC_100,
    SRC_1KM,
    CA_CRACK_PCT,
    CA_BI_THRESHOLD,
    CA_RUT_THRESHOLD,
    RAVELLING_RATING_THRESHOLD,
    IRC_TV,
    normalize_lane,
)
from src.common.logger import get_logger
from src.hierarchy.parent_builder import build_side_parent
from src.hierarchy.child_builder import (
    build_child_ca,
    build_child_10m,
    build_child_irc100,
)

logger = get_logger(__name__)


def build_potholes(pothole_df: pd.DataFrame, upc: str) -> pd.DataFrame:
    """
    Format master_pothole as top-level parent defect records.
    """
    if pothole_df is None or pothole_df.empty:
        return pd.DataFrame()

    out = pothole_df.copy()
    t_p = t_parent('pothole')
    out['collection_name'] = t_p
    out['level'] = 'parent'

    if 'id' in out.columns:
        out['id'] = out['id'].str.replace('master_pothole', t_p, regex=False)

    norm_lane = out['lane'].apply(normalize_lane)
    out['side'] = np.where(
        norm_lane.str.upper().str.startswith('L'),
        'LHS',
        np.where(norm_lane.str.upper().str.startswith('R'), 'RHS', None)
    )
    out['parent_id'] = out['id']
    out['display_defect_id'] = None
    logger.info("Formatted master_pothole: %d rows", len(out))
    return out


def assign_display_defect_ids(
    tables: Dict[str, pd.DataFrame],
    date_str: str,
    starting_seq: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Assign display_defect_id:
      - Top tables (parents / cracking top): D/{ddmmyy}/N/{seq:05d} for breach_flag == 1
      - Child tables: {parent_display_id}/{seq:04d} for breach children with parent ID
      - Orphan breach children: assigned new top counter D/{ddmmyy}/N/{seq:05d}
    """
    res = {k: v.copy() for k, v in tables.items() if v is not None and not v.empty}
    current_seq = starting_seq

    def sort_key_df(df: pd.DataFrame) -> pd.DataFrame:
        temp = df.copy()
        temp['_s_side'] = temp.get('side', pd.Series(dtype=object)).fillna('ZZZ')
        temp['_s_lane'] = temp.get('lane', pd.Series(dtype=object)).fillna('ZZZ')
        temp['_s_ch'] = pd.to_numeric(temp.get('start_chainage', 0), errors='coerce').fillna(999999999)
        temp['_s_id'] = temp.get('id', pd.Series(dtype=object)).fillna('')
        return temp.sort_values(['_s_side', '_s_lane', '_s_ch', '_s_id']).drop(
            columns=['_s_side', '_s_lane', '_s_ch', '_s_id']
        )

    # 1. Number top tables
    for d in ALL_DEFECTS:
        tt = t_top(d)
        if tt not in res or res[tt].empty:
            continue

        df = res[tt]
        df['display_defect_id'] = None
        mask = (pd.to_numeric(df.get('breach_flag', 0), errors='coerce').fillna(0).astype(int) == 1)

        breach_rows = df[mask].copy()
        non_breach = df[~mask].copy()

        if not breach_rows.empty:
            breach_rows = sort_key_df(breach_rows)
            n = len(breach_rows)
            seqs = np.arange(current_seq + 1, current_seq + n + 1)
            breach_rows['display_defect_id'] = [f"D/{date_str}/N/{s:05d}" for s in seqs]
            current_seq += n

        res[tt] = pd.concat([breach_rows, non_breach], ignore_index=True)

    # 2. Number child tables
    for d in DEFECTS_LIN:
        tt = t_top(d)
        if tt not in res or res[tt].empty:
            continue

        top_df = res[tt]
        top_breaches = top_df[top_df['display_defect_id'].notna()]
        top_key_col = 'parent_id' if has_parent(d) else 'id'

        if top_key_col in top_breaches.columns:
            lk = top_breaches.drop_duplicates(subset=[top_key_col])[[top_key_col, 'display_defect_id']].copy()
            lk.columns = ['_pj', '_pdid']
        else:
            lk = pd.DataFrame(columns=['_pj', '_pdid'])

        for t in child_tables(d):
            if t == tt or t not in res or res[t].empty:
                continue

            cdf = res[t].copy()
            cdf = cdf.merge(lk, left_on='parent_id', right_on='_pj', how='left')

            bf = (pd.to_numeric(cdf.get('breach_flag', 0), errors='coerce').fillna(0).astype(int) == 1)
            has_p = bf & cdf['_pdid'].notna()
            orphan = bf & cdf['_pdid'].isna()

            cdf['display_defect_id'] = None

            # Has parent display id -> {parent}/{child_seq:04d}
            if has_p.any():
                has_p_rows = cdf[has_p].copy()
                has_p_rows = sort_key_df(has_p_rows)
                seq_per_parent = has_p_rows.groupby('parent_id').cumcount() + 1
                has_p_rows['display_defect_id'] = (
                    has_p_rows['_pdid'].astype(str) + '/' + seq_per_parent.apply(lambda x: f"{x:04d}")
                )
            else:
                has_p_rows = pd.DataFrame(columns=cdf.columns)

            # Orphan breach -> gets top counter D/{date_str}/N/{seq:05d}
            if orphan.any():
                orphan_rows = cdf[orphan].copy()
                orphan_rows = sort_key_df(orphan_rows)
                n = len(orphan_rows)
                seqs = np.arange(current_seq + 1, current_seq + n + 1)
                orphan_rows['display_defect_id'] = [f"D/{date_str}/N/{s:05d}" for s in seqs]
                current_seq += n
            else:
                orphan_rows = pd.DataFrame(columns=cdf.columns)

            non_breach_child = cdf[~bf].copy()

            recombined = pd.concat([has_p_rows, orphan_rows, non_breach_child], ignore_index=True)
            recombined.drop(columns=['_pj', '_pdid'], inplace=True, errors='ignore')
            res[t] = recombined

    logger.info("Assigned display defect IDs (final counter: %d)", current_seq)
    return res


def build_all_silver_hierarchy(
    output_dfs: Dict[str, pd.DataFrame],
    upc: str,
    date_str: str,
    ca_intervals: Optional[Dict[str, int]] = None,
    ca_thresholds: Optional[Dict[str, float]] = None,
    irc_thresholds: Optional[Dict[str, float]] = None,
    starting_seq: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Build complete set of Silver hierarchy tables:
      - Parents (1000m LHS/RHS)
      - Children (10m, CA interval, 100m)
      - Potholes
      - Display defect IDs
    """
    ca_ivs = ca_intervals or {}
    ca_ths = ca_thresholds or {
        'roughness': CA_BI_THRESHOLD,
        'rutting': CA_RUT_THRESHOLD,
        'ravelling': float(RAVELLING_RATING_THRESHOLD),
        'cracking': CA_CRACK_PCT,
    }
    irc_ths = irc_thresholds or IRC_TV

    new_outputs = {}

    for d in DEFECTS_LIN:
        # Build side parent if applicable
        if has_parent(d):
            p_table = t_parent(d)
            p_src = SRC_1KM[d] if parent_len(d) == PARENT_LENGTH_M else SRC_100[d]
            src_df = output_dfs.get(p_src)
            new_outputs[p_table] = build_side_parent(d, src_df, upc, ca_ths[d], PARENT_LENGTH_M)

        # Build children
        for t, kind in TABLE_SPEC[d]['children']:
            if kind == 'ca':
                base_df = output_dfs.get(SRC_100[d])
                src_name = SRC_100[d]
                new_outputs[t] = build_child_ca(d, t, base_df, src_name, upc, ca_ivs.get(d))
            elif kind == '10m':
                src_10m = output_dfs.get(SRC_10M[d])
                src_name = SRC_10M[d]
                new_outputs[t] = build_child_10m(
                    d, t, src_10m, src_name, upc, ca_ivs.get(d), irc_ths.get(d)
                )
            elif kind == 'irc100':
                src_100 = output_dfs.get(SRC_100[d])
                src_name = SRC_100[d]
                new_outputs[t] = build_child_irc100(
                    d, t, src_100, src_name, upc, ca_ivs.get(d), irc_ths.get(d)
                )

    # Build Potholes
    pothole_df = output_dfs.get('master_pothole')
    new_outputs[t_parent('pothole')] = build_potholes(pothole_df, upc)

    # Handle no-parent defects (cracking, ravelling)
    for d in DEFECTS_LIN:
        if has_parent(d):
            continue
        tt = t_top(d)
        if tt not in new_outputs or new_outputs[tt].empty:
            continue

        top_df = new_outputs[tt]
        key_df = top_df[['parent_id', 'id']].drop_duplicates(subset=['parent_id']).copy()
        key_df.columns = ['_k', '_top_id']

        for ct in child_tables(d):
            if ct == tt or ct not in new_outputs or new_outputs[ct].empty:
                continue
            child_df = new_outputs[ct].copy()
            merged = child_df.merge(key_df, left_on='parent_id', right_on='_k', how='left')
            merged['parent_id'] = merged['_top_id']
            merged.drop(columns=['_k', '_top_id'], inplace=True, errors='ignore')
            new_outputs[ct] = merged

        top_df['parent_id'] = None
        new_outputs[tt] = top_df

    # Assign display defect IDs
    new_outputs = assign_display_defect_ids(new_outputs, date_str, starting_seq=starting_seq)
    return new_outputs
