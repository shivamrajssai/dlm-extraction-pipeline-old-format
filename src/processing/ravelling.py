"""
Ravelling defect processing, area percentage calculation, and aggregation.

Migrated from Notebook Cell 4, Cell 7, and Cell 9 (Linear Output Sets).
Preserves all business logic:
- Ravelling % (10m) = Ravelling_Area_sq_m * 100.0 / 35.0
- Ravelling % (100m/1km) = sum(Area_sq_m) / (3.5 × length_m) * 100.0
- CA breach: Ravelling % >= RAVELLING_RATING_THRESHOLD
- IRC breach: Ravelling % >= IRC_RAVELLING_THRESHOLD
- Dual records (As per CA and As per IRC) for 10m child
"""

import json
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.constants import (
    CHILD_100M,
    CHILD_10M,
    PARENT_LENGTH_M,
    as_float,
    as_int,
    as_text,
    first_non_null,
    lane_to_direction,
    normalize_direction,
    normalize_lane,
    standardize_nh_number,
)
from config.table_schema import BASE_DEFECT_COLUMNS
from src.ingestion.d1_loader import D1Context
from src.ingestion.d3_loader import lookup_d3_condition
from src.ingestion.tracker_loader import get_reporting_interval, get_threshold_name


def compute_ravelling_breach_ca(rav_val: Optional[float], ca_threshold: float) -> int:
    if rav_val is not None and pd.notna(rav_val) and rav_val >= ca_threshold:
        return 1
    return 0


def compute_ravelling_breach_irc(rav_val: Optional[float], irc_threshold: float) -> int:
    if rav_val is not None and pd.notna(rav_val) and rav_val >= irc_threshold:
        return 1
    return 0


def build_ravelling_linear_set(
    df_raw: pd.DataFrame,
    upc: str,
    zone: str,
    phase: str,
    cycle: str,
    d1_ctx: D1Context,
    tracker_ctx: Dict[str, Any],
    d3_conditions: Dict[str, pd.DataFrame],
    run_date: date,
    ca_ravelling_threshold: float,
    irc_ravelling_threshold: float,
    ca_interval_m: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build master_ravelling_child_10m, master_ravelling_child_100m, master_ravelling_parent.
    """
    target_columns = columns or list(BASE_DEFECT_COLUMNS)
    if df_raw.empty:
        return {
            "master_ravelling_child_10m": pd.DataFrame(columns=target_columns),
            "master_ravelling_child_100m": pd.DataFrame(columns=target_columns),
            "master_ravelling_parent": pd.DataFrame(columns=target_columns),
        }

    survey_date_str = run_date.strftime("%d%m%y")
    now_ts = pd.Timestamp.now()
    upc_slash = upc.replace("_", "/")

    norm_rows = []
    for idx, row in df_raw.iterrows():
        s = as_int(row.get("Start_Chainage"))
        e = as_int(row.get("End_Chainage"))
        if s is None and e is None:
            continue
        if s is None: s = e
        if e is None: e = s
        if e < s: s, e = e, s

        lane = normalize_lane(row.get("Lane"))
        seg_len = e - s
        rav_area = as_float(row.get("Ravelling_Area_sq_m"))
        rav_pct = as_float(row.get("Ravelling"))
        sev = as_text(row.get("Condition"))

        norm_rows.append({
            "row_id": idx + 1,
            "lane": lane,
            "nh_number": standardize_nh_number(row.get("NH_Number")),
            "direction": first_non_null(lane_to_direction(lane), normalize_direction(row.get("Direction"))),
            "survey_date": pd.to_datetime(row.get("SurveyDate"), errors="coerce"),
            "start_chainage": s,
            "end_chainage": e,
            "length_m": seg_len,
            "start_lat": as_float(row.get("Start_Lat")),
            "start_long": as_float(row.get("Start_Lon")),
            "end_lat": as_float(row.get("End_Lat")),
            "end_long": as_float(row.get("End_Lon")),
            "ravelling_area": rav_pct,
            "rav_area_sqm": rav_area,
            "severity": sev,
            "metric_value": rav_pct,
            "affected_length": float(seg_len) if compute_ravelling_breach_ca(rav_pct, ca_ravelling_threshold) else 0.0,
            "total_length": float(seg_len),
            "affected_length_ca": float(seg_len) if compute_ravelling_breach_ca(rav_pct, ca_ravelling_threshold) else 0.0,
            "affected_length_irc": float(seg_len) if compute_ravelling_breach_irc(rav_pct, irc_ravelling_threshold) else 0.0,
        })
    norm_pdf = pd.DataFrame(norm_rows)

    # 10m Child Rows
    child_10m_rows = []
    for _, row in norm_pdf.iterrows():
        s, e = row["start_chainage"], row["end_chainage"]
        if s is None or e is None or e <= s:
            child_10m_rows.append(row.to_dict())
            continue
        s, e = int(round(s)), int(round(e))
        for seg_start in range(s, e, CHILD_10M):
            sub = row.to_dict().copy()
            sub["start_chainage"] = seg_start
            sub["end_chainage"] = min(seg_start + CHILD_10M, e)
            sub["length_m"] = sub["end_chainage"] - sub["start_chainage"]
            child_10m_rows.append(sub)
    child_10m_pdf = pd.DataFrame(child_10m_rows)

    # CA (mid-level) Aggregation
    ca_iv_m = int(ca_interval_m) if ca_interval_m is not None else get_reporting_interval(tracker_ctx, "ravelling")
    c100 = norm_pdf.copy()
    c100["bin_start"] = (c100["start_chainage"] // ca_iv_m) * ca_iv_m
    c100["bin_end"] = c100["bin_start"] + ca_iv_m
    gk100 = ["lane", "bin_start", "bin_end"]
    agg_dict = {
        "survey_date": "first", "start_lat": "first", "start_long": "first",
        "end_lat": "last", "end_long": "last", "length_m": "sum", "row_id": "count",
        "start_chainage": "min", "end_chainage": "max",
        "rav_area_sqm": "sum", "affected_length": "sum",
        "total_length": "sum", "affected_length_ca": "sum", "affected_length_irc": "sum",
    }
    c100_agg = c100.groupby(gk100, dropna=False).agg(agg_dict).reset_index().rename(columns={"row_id": "segment_count"})
    lane_area_100 = 3.5 * pd.to_numeric(c100_agg["length_m"], errors="coerce")
    v100 = c100_agg["rav_area_sqm"].notna() & (lane_area_100 > 0)
    c100_agg.loc[v100, "ravelling_area"] = c100_agg.loc[v100, "rav_area_sqm"] / lane_area_100[v100] * 100.0
    c100_agg.loc[v100, "metric_value"] = c100_agg.loc[v100, "ravelling_area"]

    # 1000m Parent Aggregation
    c1k = norm_pdf.copy()
    c1k["parent_start_chainage"] = (c1k["start_chainage"] // PARENT_LENGTH_M) * PARENT_LENGTH_M
    c1k["parent_end_chainage"] = c1k["parent_start_chainage"] + PARENT_LENGTH_M
    gk1k = ["lane", "parent_start_chainage", "parent_end_chainage"]
    c1k_agg = c1k.groupby(gk1k, dropna=False).agg(agg_dict).reset_index().rename(columns={"row_id": "segment_count"})
    lane_area_1k = 3.5 * pd.to_numeric(c1k_agg["length_m"], errors="coerce")
    v1k = c1k_agg["rav_area_sqm"].notna() & (lane_area_1k > 0)
    c1k_agg.loc[v1k, "ravelling_area"] = c1k_agg.loc[v1k, "rav_area_sqm"] / lane_area_1k[v1k] * 100.0
    c1k_agg.loc[v1k, "metric_value"] = c1k_agg.loc[v1k, "ravelling_area"]

    def _to_table_records(pdf_data, level_name, coll_name, is_ca_only=False):
        ca_recs = []
        for seq, r in pdf_data.reset_index(drop=True).iterrows():
            s_ch = as_int(r["start_chainage"])
            e_ch = as_int(r["end_chainage"])
            ln = normalize_lane(r.get("lane"))
            dir_val = first_non_null(lane_to_direction(ln), normalize_direction(r.get("direction")))
            road_ctx = d1_ctx.get_road_context(s_ch, e_ch, lane=ln, direction=dir_val)

            p_start = (s_ch // PARENT_LENGTH_M) * PARENT_LENGTH_M if s_ch is not None else None
            p_end = p_start + PARENT_LENGTH_M if p_start is not None else None
            did = f"D/{survey_date_str}/N/{upc}/{s_ch if s_ch is not None else 0}/Ravelling/{seq+1:04d}"
            rid = f"ravelling_{level_name}_{seq+1:06d}"

            rav_pct_val = as_float(r.get("ravelling_area"))
            rav_area_val = as_float(r.get("rav_area_sqm"))
            bf_ca = compute_ravelling_breach_ca(rav_pct_val, ca_ravelling_threshold)
            seg_len = as_float(r.get("length_m")) or 0.0

            d3_cond = lookup_d3_condition(d3_conditions, "ravelling", s_ch, e_ch, ln)
            sev = first_non_null(as_text(r.get("severity")), d3_cond)

            rec = {col: None for col in target_columns}
            rec.update({
                "id": f"{upc}_{coll_name}_{seq+1:06d}",
                "collection_name": coll_name,
                "defect_base": "ravelling",
                "level": level_name,
                "defect_id": did if bf_ca == 1 else None,
                "display_defect_id": did if bf_ca == 1 else None,
                "source_defect_id": rid,
                "record_id": rid,
                "parent_id": f"{upc}_{ln}_{p_start}_{p_end}" if ln and p_start is not None else None,
                "upc": upc_slash,
                "upc_code": upc,
                "nh_number": standardize_nh_number(first_non_null(as_text(r.get("nh_number")), road_ctx.get("nh_number"))),
                "section_code": first_non_null(road_ctx.get("section_code")),
                "zone": zone,
                "phase": phase,
                "cycle": int(cycle),
                "source_system": "NSV",
                "sources": ["NSV"],
                "defect_type": "Ravelling",
                "type_of_defect": "Ravelling",
                "category": "LENGTH",
                "severity": sev,
                "pavement_type": first_non_null(road_ctx.get("pavement_type")),
                "pci": tracker_ctx.get("pci"),
                "lane": ln,
                "breach_flag": bf_ca,
                "metric_name": "ravelling_area",
                "metric_value": rav_pct_val,
                "threshold_name": "As per CA",
                "threshold_value": ca_ravelling_threshold,
                "total_children": int(r.get("segment_count", 1) or 1),
                "length_m": seg_len,
                "total_length": seg_len,
                "affected_length": round(seg_len, 4) if bf_ca == 1 else 0.0,
                "affected_length_ca": round(seg_len, 4) if bf_ca == 1 else 0.0,
                "affected_length_irc": 0.0,
                "area_sqm": rav_area_val,
                "start_chainage": s_ch,
                "end_chainage": e_ch,
                "defect_chainage": s_ch,
                "start_lat": as_float(r.get("start_lat")),
                "start_long": as_float(r.get("start_long")),
                "end_lat": as_float(r.get("end_lat")),
                "end_long": as_float(r.get("end_long")),
                "defect_lat": as_float(r.get("start_lat")),
                "defect_long": as_float(r.get("start_long")),
                "datetime": now_ts,
                "observed_at": now_ts,
                "survey_start_date": tracker_ctx.get("survey_start_date"),
                "survey_end_date": tracker_ctx.get("survey_end_date"),
                "is_duplicate": False,
                "nsv": json.dumps({"metric_source": "ravelling"}),
                "project_details": json.dumps({"project_name": upc, "zone": zone, "phase": phase, "cycle": cycle}),
                "created_at": now_ts,
                "updated_at": now_ts,
                "carriageway": first_non_null(road_ctx.get("carriageway")),
                "direction": first_non_null(dir_val, road_ctx.get("direction")),
                "parent_start_chainage": p_start,
                "parent_end_chainage": p_end,
                "reporting_interval_m": ca_iv_m if level_name == "child_100m" else (CHILD_10M if level_name == "child_10m" else PARENT_LENGTH_M),
                "segment_count": int(r.get("segment_count", 1) or 1),
                "ravelling_area": rav_pct_val,
                "rating_ca": sev,
                "rating_irc": None,
                "road_condition": "Bad" if bf_ca == 1 else "Good",
                "defect_type_filter": "Ravelling",
                "has_standard_defect": True,
                "standard_flags": "{}",
                "measures": json.dumps({
                    "ravelling_pct": rav_pct_val,
                    "ravelling_area_sqm": rav_area_val,
                    "lane_area_sqm": round(3.5 * seg_len, 4),
                    "segments": int(r.get("segment_count", 1) or 1),
                    "calculation": "Ravelling % = sum(area) / (3.5 x segment length) x 100",
                }),
                "_id_candidate": did,
                "run_date": run_date,
                "detected_on": tracker_ctx.get("survey_end_date"),
            })
            ca_recs.append(rec)

        if is_ca_only:
            return pd.DataFrame(ca_recs)

        # Clone CA -> IRC
        irc_recs = []
        for rec in ca_recs:
            irec = dict(rec)
            irec["id"] = irec["id"].replace(f"{coll_name}_", f"{coll_name}_irc_")
            irec["record_id"] = irec.get("record_id", "") + "_irc"
            irec["source_defect_id"] = irec.get("source_defect_id", "") + "_irc"
            irec["threshold_name"] = "As per IRC"
            irec["threshold_value"] = irc_ravelling_threshold
            r_val = irec.get("ravelling_area")
            bf_irc = compute_ravelling_breach_irc(r_val, irc_ravelling_threshold)
            irec["breach_flag"] = bf_irc
            seg = irec.get("length_m") or 0.0
            irec["affected_length"] = round(float(seg), 4) if bf_irc == 1 else 0.0
            irec["affected_length_irc"] = round(float(seg), 4) if bf_irc == 1 else 0.0
            irec["affected_length_ca"] = 0.0
            if bf_irc == 0:
                irec["defect_id"] = None
                irec["display_defect_id"] = None
            elif not irec.get("display_defect_id"):
                irec["defect_id"] = irec.get("_id_candidate")
                irec["display_defect_id"] = irec.get("_id_candidate")
            irc_recs.append(irec)

        return pd.DataFrame(ca_recs + irc_recs)

    df_10m = _to_table_records(child_10m_pdf, "child_10m", "master_ravelling_child_10m", is_ca_only=False)
    df_100m = _to_table_records(c100_agg, "child_100m", "master_ravelling_child_100m", is_ca_only=True)
    df_1km = _to_table_records(c1k_agg, "parent", "master_ravelling_parent", is_ca_only=True)

    return {
        "master_ravelling_child_10m": df_10m,
        "master_ravelling_child_100m": df_100m,
        "master_ravelling_parent": df_1km,
    }
