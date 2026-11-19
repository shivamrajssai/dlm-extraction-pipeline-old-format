"""
Constants and core utility functions for the DLM Extraction Pipeline.

Migrated from Notebook Cell 2 (Configuration and Imports) & Cell 7 (Schemas and Helpers).
Preserves all business logic:
- Clustering lengths (Parent=1000m, Child Mid=100m, Child 10m=10m, Raw=10m)
- Lane areas (10m=35.0 m², 100m=350.0 m², 1000m=3500.0 m²)
- PCI Weights (Crack=0.12, Ravel=0.10, Pothole=0.16, Patch=0.08, Rut=0.14, Rough=0.40)
- IRC Fixed Thresholds (Roughness=2400, Rutting=10, Cracking=30, Ravelling=10)
- Lane normalization & lane-to-direction mapping (L* -> Increasing, R* -> Decreasing)
- NH Number standardization: NH-<number><suffix>
"""

import re
import numpy as np
import pandas as pd
from typing import Any, List, Optional, Set, Tuple


# ── Clustering Lengths (meters) ──────────────────────────────────────────────
PARENT_LENGTH_M = 1000
CHILD_100M = 100
CHILD_10M = 10
RAW_SEGMENT_M = 10

# ── Lane Areas (m² assuming 3.5m standard lane width) ─────────────────────────
LANE_AREA_10M = 35.0
LANE_AREA_100M = 350.0
LANE_AREA_1000M = 3500.0

DEFECT_INTERVALS = {
    d: {"parent_m": PARENT_LENGTH_M, "child_mid_m": CHILD_100M, "child_10m_m": CHILD_10M}
    for d in ["roughness", "rutting", "cracking", "ravelling", "pothole"]
}

LEVEL_STANDARDS = {
    "parent": ["CA"],
    "child_100m": ["IRC", "CA"],
    "child_10m": ["CA", "IRC"],
}

# ── PCI Weights (sum to 1.0) ─────────────────────────────────────────────────
W_CRACK = 0.12
W_RAVEL = 0.10
W_POTHOLE = 0.16
W_PATCH = 0.08
W_RUT = 0.14
W_ROUGH = 0.40

PCI_WEIGHTS = {
    "cracking": W_CRACK,
    "ravelling": W_RAVEL,
    "pothole": W_POTHOLE,
    "patching": W_PATCH,
    "rutting": W_RUT,
    "roughness": W_ROUGH,
}

# ── Default Threshold Values (Fallback when D2 is missing/partial) ───────────
CA_CRACK_PCT = 5.0
CA_BI_THRESHOLD = 2000.0
CA_RUT_THRESHOLD = 5.0
RAVELLING_RATING_THRESHOLD = 10.0

# ── IRC Fixed Thresholds (breach when value > threshold) ─────────────────────
IRC_FIXED_THRESHOLDS = {
    "roughness": 2400.0,  # mm/km IRI
    "rutting": 10.0,       # mm
    "cracking": 30.0,      # %
    "ravelling": 10.0,     # %
}
IRC_CHILD_M = 100

# ── Table Specifications for Silver Layer (LHS/RHS hierarchy) ────────────────
DEFECTS_LINEAR = ["roughness", "rutting", "ravelling", "cracking"]
ALL_DEFECTS = DEFECTS_LINEAR + ["pothole"]

TABLE_SPEC = {
    "roughness": {
        "parent": ("master_roughness_parent", PARENT_LENGTH_M),
        "children": [("master_roughness_child_10m", "10m"), ("master_roughness_as_per_ca", "ca")],
    },
    "rutting": {
        "parent": ("master_rutting_parent", PARENT_LENGTH_M),
        "children": [("master_rutting_child_10m", "10m"), ("master_rutting_as_per_ca", "ca")],
    },
    "cracking": {
        "parent": None,  # no LHS/RHS parent table
        "children": [("master_cracking_10m", "10m"), ("master_cracking_as_per_ca", "ca")],
    },
    "pothole": {
        "parent": ("master_pothole", None),  # defect records, no child
        "children": [],
    },
    "ravelling": {
        "parent": None,
        "ca_interval_m": None,  # dynamic CA reporting interval
        "children": [("master_ravelling_10m", "10m"), ("master_ravelling", "ca")],
    },
}

SRC_10M = {
    "roughness": "master_roughness_child_10m",
    "rutting": "master_rutting_child_10m",
    "ravelling": "master_ravelling_child_10m",
    "cracking": "master_cracking",  # from initial 10m build
}
SRC_100 = {d: f"master_{d}_child_100m" for d in DEFECTS_LINEAR}
SRC_1KM = {d: f"master_{d}_parent" for d in DEFECTS_LINEAR}

DEFECTS_LIN = DEFECTS_LINEAR
IRC_TV = IRC_FIXED_THRESHOLDS


def has_parent(d: str) -> bool:
    """True if defect has LHS/RHS parent (roughness, rutting, pothole)."""
    return TABLE_SPEC.get(d, {}).get("parent") is not None


def t_parent(d: str) -> Optional[str]:
    """Parent table name for defect d."""
    return TABLE_SPEC[d]["parent"][0] if has_parent(d) else None


def parent_len(d: str) -> Optional[int]:
    """Parent aggregation length (e.g. 1000m)."""
    return TABLE_SPEC[d]["parent"][1] if has_parent(d) else None


def child_tables(d: str) -> List[str]:
    """List of child table names for defect d."""
    return [t for t, _ in TABLE_SPEC.get(d, {}).get("children", [])]


def child_kind(d: str, t: str) -> Optional[str]:
    """Kind of child table: 'ca', '10m', 'irc100'."""
    return dict(TABLE_SPEC.get(d, {}).get("children", [])).get(t)


def t_top(d: str) -> str:
    """Table holding the top-level display IDs (the parent, else cracking/ravelling CA table)."""
    if has_parent(d):
        return t_parent(d)
    ca_children = [t for t, k in TABLE_SPEC.get(d, {}).get("children", []) if k == "ca"]
    return ca_children[0] if ca_children else ""


def all_tables(d: str) -> List[str]:
    """All table names for defect d (parent + children)."""
    p = [t_parent(d)] if has_parent(d) else []
    return p + child_tables(d)


def ca_iv(d: str, default_iv: Optional[int] = None) -> int:
    """Interval of the 'ca' table: from default_iv (D2/Tracker), spec, or fallback."""
    if default_iv is not None:
        return int(default_iv)
    spec_iv = TABLE_SPEC.get(d, {}).get("ca_interval_m")
    if spec_iv is not None:
        return int(spec_iv)
    return CHILD_100M


# ── Lane Normalization & Direction ───────────────────────────────────────────
def normalize_lane(v: Any) -> Optional[str]:
    """
    Normalize lane strings: 'R-1' / 'r 1' / 'Lane R1' / 'LANE_L2' / 'L.3' -> 'R1' / 'L2' / 'L3'.
    Null/NaN stays None; unknown text is upper-cased.
    """
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = re.sub(r"[\s\-_\.]+", "", str(v).strip().upper().replace("LANE", ""))
    return s or None


def lane_to_direction(lane_value: Any) -> Optional[str]:
    """
    Direction always follows the lane:
    Any L-lane (L1..L8) = 'Increasing'
    Any R-lane (R1..R8) = 'Decreasing'
    """
    s = normalize_lane(lane_value)
    if s is None:
        return None
    if re.fullmatch(r"L\d+", s):
        return "Increasing"
    if re.fullmatch(r"R\d+", s):
        return "Decreasing"
    return None


def normalize_direction(v: Any) -> Optional[str]:
    """
    Normalize direction text from D1: 'Increasing' / 'Decreasing'.
    Handles case, whitespace, 'Inc' / 'Dec'; other text kept.
    """
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if not s:
        return None
    if s.lower().startswith("inc"):
        return "Increasing"
    if s.lower().startswith("dec"):
        return "Decreasing"
    return s


LANE_TO_DIRECTION = {f"{p}{i}": lane_to_direction(f"{p}{i}") for p in "LR" for i in range(1, 9)}


# ── NH Number Standardization ────────────────────────────────────────────────
_NH_RE = re.compile(r"^\s*(?:N\.?\s*H\.?)?\s*[-_/ ]*0*(\d+)\s*([A-Za-z]*)\s*$", re.I)


def standardize_nh_number(value: Any) -> Optional[str]:
    """
    Standardize NH number format: always 'NH-<number><suffix>'.
    e.g., NH0027 -> NH-27, NH151A -> NH-151A, 'NH 8a' -> NH-8A, '27' -> NH-27.
    """
    v = as_text(value)
    if v is None:
        return None
    m = _NH_RE.match(v)
    if not m:
        return v.strip().upper()
    return f"NH-{int(m.group(1))}{m.group(2).upper()}"


# ── Scalar Type Helpers ──────────────────────────────────────────────────────
def as_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def as_int(value: Any) -> Optional[int]:
    num = as_float(value)
    return int(round(num)) if num is not None else None


def as_chainage_m(value: Any, is_km: bool = False) -> Optional[int]:
    num = as_float(value)
    if num is None:
        return None
    if is_km and num < 1000:
        num = num * 1000
    return int(round(num))


def as_text(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    return s if s and s != "<NA>" and s.lower() != "nan" else None


def first_non_null(*values: Any) -> Optional[Any]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def uniq(items: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen: Set[Any] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def norm_name(name: Any) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def pick_col(columns: List[str], *candidates: str) -> Optional[str]:
    normalized = {norm_name(c): c for c in columns}
    for cand in candidates:
        key = norm_name(cand)
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = norm_name(cand)
        for col in columns:
            if key in norm_name(col):
                return col
    return None
