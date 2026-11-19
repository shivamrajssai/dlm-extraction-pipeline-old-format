"""
Side parent builder module (LHS/RHS).
Ported from Databricks DLM Extraction Pipeline (Cell 14).

Builds aggregated 1000m parent records for LHS and RHS across lanes
for linear defects (Roughness, Rutting).
"""

from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np

from src.common.constants import (
    PARENT_LENGTH_M,
    TABLE_SPEC,
    has_parent,
    parent_len,
    normalize_lane,
    lane_to_direction,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def build_side_parent(
    defect: str,
    src_df: pd.DataFrame,
    upc: str,
    ca_threshold: float,
    parent_len_m: int = PARENT_LENGTH_M,
) -> pd.DataFrame:
    """
    Build LHS / RHS 1000m parent table for a given defect from its 1km CA rows.
    Value = average of the side's lanes; breach if any lane breaches the CA threshold.
    """
    if not has_parent(defect) or src_df is None or src_df.empty:
        return pd.DataFrame()

    table, L = TABLE_SPEC[defect]['parent']
    L = parent_len_m

    # Filter to 'As per CA'
    df = src_df.copy()
    if 'threshold_name' in df.columns:
        df = df[df['threshold_name'] == 'As per CA']

    if df.empty:
        logger.warning("No CA rows found for %s parent aggregation", defect)
        return pd.DataFrame(columns=src_df.columns)

    cols = list(src_df.columns)

    # Determine side (LHS for L-lanes, RHS for R-lanes)
    df['_norm_lane'] = df['lane'].apply(normalize_lane)
    df['_side'] = np.where(
        df['_norm_lane'].str.upper().str.startswith('L'),
        'LHS',
        np.where(df['_norm_lane'].str.upper().str.startswith('R'), 'RHS', None)
    )
    df = df.dropna(subset=['_side'])

    s_col = pd.to_numeric(df['start_chainage'], errors='coerce')
    e_col = pd.to_numeric(df['end_chainage'], errors='coerce')
    df['_lo'] = np.minimum(s_col, e_col)
    df['_hi'] = np.maximum(s_col, e_col)
    df['_b'] = (np.floor(df['_lo'] / L) * L).astype(int)

    rows = []
    grouped = df.groupby(['_side', '_b'])

    for (side, b), g in grouped:
        b = int(b)
        idx_min = g['_lo'].idxmin()
        idx_max = g['_hi'].idxmax()
        first = g.loc[idx_min]
        last = g.loc[idx_max]

        mv = pd.to_numeric(g['metric_value'], errors='coerce')
        lane_bf = pd.to_numeric(g['breach_flag'], errors='coerce').fillna(0).astype(int)
        bf = int(lane_bf.max())

        worst = g.loc[mv.idxmax()] if mv.notna().any() else first
        pav = sorted(set(g['pavement_type'].dropna()))

        pid = f"{upc}_{side}_{b}_{b + L}"

        def _sum(c: str) -> float:
            if c in g.columns:
                return float(pd.to_numeric(g[c], errors='coerce').fillna(0).sum())
            return 0.0

        rec = {c: first.get(c) for c in cols}
        rec.update({
            'id': f"{upc}_{table}_{side}_{b}",
            'collection_name': table,
            'level': 'parent',
            'lane': side,
            'side': side,
            'direction': 'Increasing' if side == 'LHS' else 'Decreasing',
            'parent_id': pid,
            'record_id': pid,
            'source_defect_id': pid,
            'defect_id': pid if bf else None,
            'display_defect_id': None,
            'metric_value': round(float(mv.mean()), 6) if mv.notna().any() else None,
            'breach_flag': bf,
            'threshold_name': 'As per CA',
            'threshold_value': float(ca_threshold),
            'total_children': int(len(g)),
            'segment_count': int(_sum('segment_count')),
            'start_chainage': int(first['_lo']),
            'end_chainage': int(last['_hi']),
            'length_m': float(last['_hi'] - first['_lo']),
            'total_length': _sum('length_m'),
            'defect_chainage': (float(first['_lo']) + float(last['_hi'])) / 2.0,
            'start_lat': first.get('start_lat'),
            'start_long': first.get('start_long'),
            'end_lat': last.get('end_lat'),
            'end_long': last.get('end_long'),
            'defect_lat': first.get('start_lat'),
            'defect_long': first.get('start_long'),
            'pavement_type': pav[0] if len(pav) == 1 else ('Hybrid' if pav else None),
            'parent_start_chainage': b,
            'parent_end_chainage': b + L,
            'reporting_interval_m': L,
            'rating_ca': worst.get('rating_ca'),
            'rating_irc': worst.get('rating_irc'),
            'road_condition': 'Bad' if bf else 'Good',
            'affected_length': _sum('affected_length'),
            'affected_length_ca': _sum('affected_length'),
            'affected_length_irc': 0.0,
            'measures': json.dumps({
                'side': side,
                'lanes': sorted(g['lane'].astype(str)),
                'lane_values': {
                    str(l): (None if pd.isna(v) else round(float(v), 6))
                    for l, v in zip(g['lane'], mv)
                },
                'lane_breach': {str(l): int(x) for l, x in zip(g['lane'], lane_bf)},
                'calculation': (
                    f"{side} {b}-{b + L}: value = average of {len(g)} lane {L}m values; "
                    f"breach if any lane breaches the CA threshold ({ca_threshold})"
                ),
            }),
        })

        if 'area_sqm' in cols:
            rec['area_sqm'] = _sum('area_sqm')
        if 'cracking_pct' in cols:
            rec['cracking_pct'] = rec['metric_value']
        if 'roughness_bi' in cols:
            rec['roughness_bi'] = rec['metric_value']
        if 'iri_value' in cols:
            iri_vals = pd.to_numeric(g['iri_value'], errors='coerce')
            rec['iri_value'] = round(float(iri_vals.mean()), 6) if iri_vals.notna().any() else None
        if 'rutting_avg' in cols:
            rec['rutting_avg'] = rec['metric_value']

        rows.append(rec)

    parent_df = pd.DataFrame(rows)
    logger.info("Built side parent table %s: %d rows", table, len(parent_df))
    return parent_df
