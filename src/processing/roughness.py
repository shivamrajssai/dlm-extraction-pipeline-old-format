"""
Roughness defect processing, BI derivation, aggregation, and breach detection.

Migrated from Notebook Cell 4 (Roughness LWP+RWP Derivation), Cell 7, and Cell 9 (Linear Output Sets).
Preserves all business logic:
- Deriving Lane IRI = (LwpIri + RwpIri) / 2
- Roughness BI (mm/km) = 630 × (Lane IRI)^1.12
- Recomputing 100m and 1000m BI from averaged LWP+RWP IRI (not mean of per-10m BIs)
- CA breach: BI >= CA_BI_THRESHOLD
- IRC breach: BI >= IRC_IRI_POOR (or IRI > 2400 mm/km)
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
    DEFECT_INTERVALS,
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


def compute_roughness_breach_ca(bi_value: Optional[float], ca_bi_threshold: float) -> int:
    if bi_value is not None and pd.notna(bi_value) and bi_value >= ca_bi_threshold:
        return 1
    return 0


def compute_roughness_breach_irc(bi_value: Optional[float], irc_threshold: float) -> int:
    if bi_value is not None and pd.notna(bi_value) and bi_value >= irc_threshold:
        return 1
    return 0


def build_roughness_linear_set(
    df_raw: pd.DataFrame,
    upc: str,
    zone: str,
    phase: str,
    cycle: str,
    d1_ctx: D1Context,
    tracker_ctx: Dict[str, Any],
    d3_conditions: Dict[str, pd.DataFrame],
    run_date: date,
    ca_bi_threshold: float,
    irc_iri_threshold: float,
    ca_interval_m: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build master_roughness_child_10m, master_roughness_child_100m, master_roughness_parent.
    """
    target_columns = columns or list(BASE_DEFECT_COLUMNS)
    if df_raw.empty:
        return {
            "master_roughness_child_10m": pd.DataFrame(columns=target_columns),
            "master_roughness_child_100m": pd.DataFrame(columns=target_columns),
            "master_roughness_parent": pd.DataFrame(columns=target_columns),
        }

    survey_date_str = run_date.strftime("%d%m%y")
    now_ts = pd.Timestamp.now()
    upc_slash = upc.replace("_", "/")

    # 1. Normalize Raw 10m Rows
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
        bi_val = as_float(row.get("BI m/km"))
        iri_val = as_float(row.get("IRI"))
        lwp = as_float(row.get("LwpIri"))
        rwp = as_float(row.get("RwpIri"))
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
            "roughness_bi": bi_val,
            "iri_value": iri_val,
            "lwp_iri": lwp,
            "rwp_iri": rwp,
            "severity": sev,
            "metric_value": first_non_null(bi_val, iri_val),
            "affected_length": float(seg_len) if compute_roughness_breach_ca(bi_val, ca_bi_threshold) else 0.0,
            "total_length": float(seg_len),
            "affected_length_ca": float(seg_len) if compute_roughness_breach_ca(bi_val, ca_bi_threshold) else 0.0,
            "affected_length_irc": float(seg_len) if compute_roughness_breach_irc(bi_val, irc_iri_threshold) else 0.0,
        })
    norm_pdf = pd.DataFrame(norm_rows)

    # 2. Child 10m Rows (split if longer than 10m)
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

    # 3. Child CA (mid-level) Aggregation
    ca_iv_m = int(ca_interval_m) if ca_interval_m is not None else get_reporting_interval(tracker_ctx, "roughness")
    c100 = norm_pdf.copy()
    c100["bin_start"] = (c100["start_chainage"] // ca_iv_m) * ca_iv_m
    c100["bin_end"] = c100["bin_start"] + ca_iv_m
    gk100 = ["lane", "bin_start", "bin_end"]
    agg_dict = {
        "survey_date": "first", "start_lat": "first", "start_long": "first",
        "end_lat": "last", "end_long": "last", "length_m": "sum", "row_id": "count",
        "start_chainage": "min", "end_chainage": "max",
        "lwp_iri": "mean", "rwp_iri": "mean", "affected_length": "sum",
        "total_length": "sum", "affected_length_ca": "sum", "affected_length_irc": "sum",
    }
    c100_agg = c100.groupby(gk100, dropna=False).agg(agg_dict).reset_index().rename(columns={"row_id": "segment_count"})
    # Recompute BI from averaged LWP+RWP IRI
    valid100 = c100_agg["lwp_iri"].notna() & c100_agg["rwp_iri"].notna()
    avg_iri_100 = (c100_agg.loc[valid100, "lwp_iri"] + c100_agg.loc[valid100, "rwp_iri"]) / 2
    c100_agg.loc[valid100, "iri_value"] = avg_iri_100
    c100_agg.loc[valid100, "roughness_bi"] = 630.0 * (avg_iri_100 ** 1.12)
    c100_agg.loc[valid100, "metric_value"] = c100_agg.loc[valid100, "roughness_bi"]

    # 4. Parent 1000m Aggregation (average of 100m CA roughness_bi values per lane)
    c1k = c100_agg.copy()
    c1k["parent_start_chainage"] = (c1k["start_chainage"] // PARENT_LENGTH_M) * PARENT_LENGTH_M
    c1k["parent_end_chainage"] = c1k["parent_start_chainage"] + PARENT_LENGTH_M
    gk1k = ["lane", "parent_start_chainage", "parent_end_chainage"]
    agg_dict_parent = {
        "survey_date": "first", "start_lat": "first", "start_long": "first",
        "end_lat": "last", "end_long": "last", "length_m": "sum", "segment_count": "sum",
        "start_chainage": "min", "end_chainage": "max",
        "lwp_iri": "mean", "rwp_iri": "mean", "iri_value": "mean",
        "roughness_bi": "mean", "affected_length": "sum",
        "total_length": "sum", "affected_length_ca": "sum", "affected_length_irc": "sum",
    }
    c1k_agg = c1k.groupby(gk1k, dropna=False).agg(agg_dict_parent).reset_index()
    c1k_agg["metric_value"] = c1k_agg["roughness_bi"]

    # 5. Build Records Function
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
            did = f"D/{survey_date_str}/N/{upc}/{s_ch if s_ch is not None else 0}/Roughness/{seq+1:04d}"
            rid = f"roughness_{level_name}_{seq+1:06d}"

            bi = as_float(r.get("roughness_bi"))
            iri = as_float(r.get("iri_value"))
            mval = first_non_null(bi, iri)
            bf_ca = compute_roughness_breach_ca(bi, ca_bi_threshold)
            seg_len = as_float(r.get("length_m")) or 0.0

            d3_cond = lookup_d3_condition(d3_conditions, "roughness", s_ch, e_ch, ln)
            sev = first_non_null(as_text(r.get("severity")), d3_cond)

            rec = {col: None for col in target_columns}
            rec.update({
                "id": f"{upc}_{coll_name}_{seq+1:06d}",
                "collection_name": coll_name,
                "defect_base": "roughness",
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
                "defect_type": "Roughness",
                "type_of_defect": "Roughness",
                "category": "LENGTH",
                "severity": sev,
                "pavement_type": first_non_null(road_ctx.get("pavement_type")),
                "pci": tracker_ctx.get("pci"),
                "lane": ln,
                "breach_flag": bf_ca,
                "metric_name": "roughness_bi",
                "metric_value": mval,
                "threshold_name": "As per CA",
                "threshold_value": ca_bi_threshold,
                "total_children": int(r.get("segment_count", 1) or 1),
                "length_m": seg_len,
                "total_length": seg_len,
                "affected_length": round(seg_len, 4) if bf_ca == 1 else 0.0,
                "affected_length_ca": round(seg_len, 4) if bf_ca == 1 else 0.0,
                "affected_length_irc": 0.0,
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
                "nsv": json.dumps({"metric_source": "roughness"}),
                "project_details": json.dumps({"project_name": upc, "zone": zone, "phase": phase, "cycle": cycle}),
                "created_at": now_ts,
                "updated_at": now_ts,
                "carriageway": first_non_null(road_ctx.get("carriageway")),
                "direction": first_non_null(dir_val, road_ctx.get("direction")),
                "parent_start_chainage": p_start,
                "parent_end_chainage": p_end,
                "reporting_interval_m": ca_iv_m if level_name == "child_100m" else (CHILD_10M if level_name == "child_10m" else PARENT_LENGTH_M),
                "segment_count": int(r.get("segment_count", 1) or 1),
                "roughness_bi": bi,
                "iri_value": iri,
                "iri_condition": d3_cond,
                "rating_ca": sev,
                "rating_irc": None,
                "road_condition": "Bad" if bf_ca == 1 else "Good",
                "defect_type_filter": "Roughness",
                "has_standard_defect": True,
                "standard_flags": "{}",
                "measures": json.dumps({
                    "roughness_bi": mval,
                    "lwp_iri": as_float(r.get("lwp_iri")),
                    "rwp_iri": as_float(r.get("rwp_iri")),
                    "lane_iri": iri,
                    "segments": int(r.get("segment_count", 1) or 1),
                    "calculation": "Lane IRI = (LWP IRI + RWP IRI)/2; BI = 630 x IRI^1.12",
                }),
                "_id_candidate": did,
                "run_date": run_date,
                "detected_on": tracker_ctx.get("survey_end_date"),
            })
            ca_recs.append(rec)

        if is_ca_only:
            return pd.DataFrame(ca_recs)

        # Clone CA -> IRC for 10m child
        irc_recs = []
        for rec in ca_recs:
            irec = dict(rec)
            irec["id"] = irec["id"].replace(f"{coll_name}_", f"{coll_name}_irc_")
            irec["record_id"] = irec.get("record_id", "") + "_irc"
            irec["source_defect_id"] = irec.get("source_defect_id", "") + "_irc"
            irec["threshold_name"] = "As per IRC"
            irec["threshold_value"] = irc_iri_threshold
            bi_val = irec.get("roughness_bi")
            bf_irc = compute_roughness_breach_irc(bi_val, irc_iri_threshold)
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

    df_10m = _to_table_records(child_10m_pdf, "child_10m", "master_roughness_child_10m", is_ca_only=False)
    df_100m = _to_table_records(c100_agg, "child_100m", "master_roughness_child_100m", is_ca_only=True)
    df_1km = _to_table_records(c1k_agg, "parent", "master_roughness_parent", is_ca_only=True)

    return {
        "master_roughness_child_10m": df_10m,
        "master_roughness_child_100m": df_100m,
        "master_roughness_parent": df_1km,
    }
