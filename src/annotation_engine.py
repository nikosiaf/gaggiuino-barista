"""
Gaggiuino Barista - Annotation Engine Module

This module performs deterministic telemetry analysis on espresso shots.
It extracts features, detects events, and classifies extraction quality WITHOUT
calling any AI/LLM - the analysis is purely mathematical from raw telemetry data.

Key capabilities:
1. Profile Matching: Compare shot profile against bundled community profiles
2. Feature Extraction: Compute quantitative metrics from telemetry
3. Event Detection: Identify significant moments (first drops, pressure issues, etc.)
4. Taste Classification: Predict taste profile (well_extracted, balanced, etc.)
5. Profile Adherence: Score how well shot followed matched profile

This analysis runs BEFORE any LLM call. The LLM then phrases these findings
in natural language. This hybrid approach is more stable than letting an LLM
interpret raw data directly.

Profile Templates:
- lever: Manual lever machines (higher pressure range, slower flow)
- flow_control: Flow-profiled shots (multiple phases, variable flow)
- pressure: Traditional pressure-profiled shots (single pressure target)
- filter: Long extraction filter-style shots (low pressure, extended time)
"""

import json
import math
import os
from pathlib import Path
from statistics import mean


# =========================
# SEVERITY LEVELS
# =========================
# Used to rank event importance for display and scoring
SEVERITY_ORDER = {"good": 0, "info": 1, "warning": 2, "critical": 3}


# =========================
# PROFILE TEMPLATES
# =========================
# Profile-specific thresholds derived from Gaggiuino community profiles.
# Each profile type has different expectations for pressure, flow, timing.

PROFILE_TEMPLATES = {
    "lever": {
        # Lever machines have slower flow and higher pressure targets
        "flow_range": (1.0, 2.2),
        "pressure_range": (6.0, 10.0),
        "pressure_stability_threshold": 0.8,   # Higher tolerance for pressure variation
        "flow_stability_threshold": 0.5,
        "first_drops_early": 5.0,             # First drops expected between 5-10s
        "first_drops_late": 10.0,
        "target_flow": 1.6,
        "min_flow": 0.6,
        "max_flow": 2.8,
    },
    "flow_control": {
        # Flow-profiled shots have variable flow phases
        "flow_range": (0.8, 3.0),
        "pressure_range": (4.0, 11.0),         # Wider pressure range
        "pressure_stability_threshold": 0.9,
        "flow_stability_threshold": 0.6,        # More flow variation expected
        "first_drops_early": 4.5,
        "first_drops_late": 9.0,
        "target_flow": 1.8,
        "min_flow": 0.5,
        "max_flow": 3.5,
    },
    "pressure": {
        # Traditional pressure-profiled shots
        "flow_range": (1.0, 2.8),
        "pressure_range": (7.0, 9.5),          # Tighter pressure target
        "pressure_stability_threshold": 0.7,
        "flow_stability_threshold": 0.45,
        "first_drops_early": 4.5,
        "first_drops_late": 8.0,
        "target_flow": 2.0,
        "min_flow": 0.8,
        "max_flow": 2.8,
    },
    "filter": {
        # Long extraction filter-style shots (v60-style)
        "flow_range": (0.2, 2.0),
        "pressure_range": (0.5, 3.0),           # Very low pressure
        "pressure_stability_threshold": 0.5,
        "flow_stability_threshold": 0.3,
        "first_drops_early": 6.0,               # Much slower first drops
        "first_drops_late": 15.0,
        "target_flow": 1.0,
        "min_flow": 0.2,
        "max_flow": 2.0,
    },
}

# Default template used when no profile match is found
DEFAULT_TEMPLATE = PROFILE_TEMPLATES["pressure"]


# =========================
# PROFILE LOADING
# =========================
# Cached loaded profiles to avoid repeated file I/O
_PROFILES_CACHE = None

# Paths where profiles are stored (container internal and HA accessible)
_PROFILES_DIR_CONTAINER = Path("/app/profiles")      # Bundled profiles in Docker image
_PROFILES_DIR_OUT = Path("/homeassistant/www/gaggiuino-barista/profiles")  # User profiles


def load_local_profiles():
    """
    Load all JSON profile files from known directories.
    
    Profiles are loaded from two locations:
    1. /app/profiles/ - bundled community profiles in Docker image
    2. /homeassistant/www/gaggiuino-barista/profiles/ - user-provided profiles
    
    Returns:
        List of profile dicts, deduplicated by profile name
    """
    global _PROFILES_CACHE
    if _PROFILES_CACHE is not None:
        return _PROFILES_CACHE
    
    profiles = []
    seen_names = set()
    
    # Load from both directories
    for profiles_dir in [_PROFILES_DIR_CONTAINER, _PROFILES_DIR_OUT]:
        if profiles_dir.exists():
            for json_file in profiles_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        profile_data = json.load(f)
                        profile_name = profile_data.get("name", "")
                        # Deduplicate by name
                        if profile_name and profile_name not in seen_names:
                            profile_data["_file_name"] = json_file.stem
                            profiles.append(profile_data)
                            seen_names.add(profile_name)
                except Exception:
                    pass
    
    _PROFILES_CACHE = profiles
    return profiles


# =========================
# PROFILE MATCHING
# =========================
def normalize_profile_name(name: str) -> str:
    """
    Normalize profile name for comparison by removing common variations.
    
    Handles:
    - Case differences
    - Whitespace
    - Hyphens vs underscores
    """
    if not name:
        return ""
    return name.lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def match_profile_by_name(shot_profile_name: str, local_profiles: list) -> dict | None:
    """
    Try to match a shot's profile name against loaded profiles.
    
    Matching strategy:
    1. Exact match (after normalization)
    2. Partial match (shot name contained in profile name or vice versa)
    
    Args:
        shot_profile_name: Name of profile from Gaggiuino shot
        local_profiles: List of loaded profile dicts
        
    Returns:
        Matched profile dict or None
    """
    if not shot_profile_name:
        return None
    
    shot_normalized = normalize_profile_name(shot_profile_name)
    
    # First pass: exact match
    for lp in local_profiles:
        lp_name = lp.get("name", "")
        lp_normalized = normalize_profile_name(lp_name)
        
        if lp_normalized == shot_normalized:
            return lp
    
    # Second pass: partial match
    for lp in local_profiles:
        lp_name = lp.get("name", "")
        lp_normalized = normalize_profile_name(lp_name)
        
        if shot_normalized in lp_normalized or lp_normalized in shot_normalized:
            return lp
    
    return None


def extract_phase_signature(phases: list) -> dict:
    """
    Extract a structural signature from profile phases for comparison.
    
    Used when name matching fails - compares phase structure:
    - Number of phases
    - Phase types (FLOW vs PRESSURE)
    - Stop conditions (time, weight, pressure)
    
    Args:
        phases: List of phase dicts from profile
        
    Returns:
        Signature dict with structural info
    """
    if not phases:
        return {
            "type_sequence": "",
            "num_phases": 0,
            "has_flow": False,
            "has_pressure": False,
        }
    
    type_sequence = ""
    phase_types = []
    total_time_ms = 0
    has_flow = False
    has_pressure = False
    max_restriction = 0
    stop_by_weight = False
    stop_by_time = False
    stop_by_pressure = False
    
    for phase in phases:
        phase_type = phase.get("type", "").upper()
        if phase_type == "FLOW":
            has_flow = True
            phase_types.append("F")
        elif phase_type == "PRESSURE":
            has_pressure = True
            phase_types.append("P")
        else:
            phase_types.append("?")
        
        type_sequence += phase_type[0] if phase_type else "?"
        
        stop = phase.get("stopConditions", {})
        if stop.get("time", 0) > 0:
            total_time_ms += stop["time"]
        if stop.get("weight", 0) > 0:
            stop_by_weight = True
        if stop.get("pressureAbove", 0) > 0 or stop.get("pressureBelow", 0) > 0:
            stop_by_pressure = True
        
        restriction = phase.get("restriction", 0)
        if restriction > max_restriction:
            max_restriction = restriction
    
    return {
        "type_sequence": type_sequence,
        "num_phases": len(phases),
        "phase_types": phase_types,
        "has_flow": has_flow,
        "has_pressure": has_pressure,
        "total_time_ms": total_time_ms,
        "max_restriction": max_restriction,
        "stop_by_weight": stop_by_weight,
        "stop_by_time": stop_by_time,
        "stop_by_pressure": stop_by_pressure,
    }


def score_phase_match(shot_sig: dict, local_sig: dict) -> float:
    """
    Score how well two phase signatures match (0-100).
    
    Scoring factors:
    - Same number of phases (+30)
    - Similar phase count (+15)
    - Same phase type (FLOW/PRESSURE) (+15 each)
    - Same type sequence (+25)
    - Same first phase type (+15)
    """
    score = 0.0
    
    if shot_sig["num_phases"] == local_sig["num_phases"]:
        score += 30
    elif abs(shot_sig["num_phases"] - local_sig["num_phases"]) == 1:
        score += 15
    
    if shot_sig["has_flow"] == local_sig["has_flow"]:
        score += 15
    if shot_sig["has_pressure"] == local_sig["has_pressure"]:
        score += 15
    
    if shot_sig["type_sequence"] == local_sig["type_sequence"]:
        score += 25
    
    shot_first = shot_sig["type_sequence"][0] if shot_sig["type_sequence"] else ""
    local_first = local_sig["type_sequence"][0] if local_sig["type_sequence"] else ""
    if shot_first == local_first:
        score += 15
    
    return score


def match_profile_by_phases(shot_phases: list, local_profiles: list) -> dict | None:
    """
    Find best matching profile by comparing phase structures.
    
    Used as fallback when name matching fails.
    Requires minimum match score of 40/100.
    """
    if not shot_phases or not local_profiles:
        return None
    
    shot_sig = extract_phase_signature(shot_phases)
    
    best_match = None
    best_score = 0
    
    for lp in local_profiles:
        lp_phases = lp.get("phases", [])
        if not lp_phases:
            continue
        
        lp_sig = extract_phase_signature(lp_phases)
        match_score = score_phase_match(shot_sig, lp_sig)
        
        if match_score > best_score:
            best_score = match_score
            best_match = lp
    
    if best_score >= 40:
        return best_match
    
    return None


def get_matched_profile(shot_data: dict) -> dict:
    """
    Get best matching profile for a shot, with match metadata.
    
    Returns:
        Dict with:
        - profile: Matched profile dict or None
        - match_type: "name", "phases", or "none"
        - confidence: 1.0 for name match, 0.7 for phase match, 0.0 for none
    """
    shot_profile = shot_data.get("profile", {})
    shot_name = shot_profile.get("name", "")
    shot_phases = shot_profile.get("phases", [])
    
    local_profiles = load_local_profiles()
    
    # Try name matching first (highest confidence)
    matched = match_profile_by_name(shot_name, local_profiles)
    if matched:
        return {"profile": matched, "match_type": "name", "confidence": 1.0}
    
    # Fall back to phase structure matching
    matched = match_profile_by_phases(shot_phases, local_profiles)
    if matched:
        return {"profile": matched, "match_type": "phases", "confidence": 0.7}
    
    return {"profile": None, "match_type": "none", "confidence": 0.0}


def get_profile_template_from_matched(matched_profile: dict) -> dict:
    """
    Derive profile-specific thresholds from matched profile data.
    
    Uses actual values from the matched profile where available,
    falls back to template defaults.
    
    Args:
        matched_profile: Profile dict from matching
        
    Returns:
        Complete template dict with matched profile info
    """
    if not matched_profile:
        return DEFAULT_TEMPLATE
    
    phases = matched_profile.get("phases", [])
    recipe = matched_profile.get("recipe", {})
    
    # Determine profile type from phases
    has_flow = any(p.get("type", "").upper() == "FLOW" for p in phases)
    has_pressure = any(p.get("type", "").upper() == "PRESSURE" for p in phases)
    
    # Get recipe info
    target_weight = matched_profile.get("globalStopConditions", {}).get("weight", 0)
    water_temp = matched_profile.get("waterTemperature", 93)
    coffee_in = recipe.get("coffeeIn", 0)
    coffee_out = recipe.get("coffeeOut", 0)
    
    if coffee_in > 0 and coffee_out > 0:
        ratio = coffee_out / coffee_in
    else:
        ratio = 2.0
    
    num_phases = len(phases)
    
    # Determine template based on profile characteristics
    if num_phases == 1 and has_pressure:
        template_name = "pressure"
    elif num_phases >= 3 and has_flow:
        template_name = "flow_control"
    elif "filter" in matched_profile.get("name", "").lower() or ratio < 1.5:
        template_name = "filter"
    elif "leva" in matched_profile.get("name", "").lower():
        template_name = "lever"
    elif has_flow:
        template_name = "flow_control"
    else:
        template_name = "pressure"
    
    template = PROFILE_TEMPLATES.get(template_name, DEFAULT_TEMPLATE).copy()
    
    # Add matched profile metadata to template
    template["matched_profile_name"] = matched_profile.get("name", "")
    template["matched_target_weight"] = target_weight
    template["matched_water_temp"] = water_temp
    template["matched_ratio"] = ratio
    template["matched_num_phases"] = num_phases
    
    # Calculate expected duration based on yield target
    if target_weight > 100:
        template["expected_duration_s"] = 120
        template["filter_style"] = True
    elif target_weight > 30:
        template["expected_duration_s"] = 30
        template["filter_style"] = False
    else:
        template["expected_duration_s"] = 25
        template["filter_style"] = False
    
    return template


def detect_profile_type(profile_name: str, phases: list) -> dict:
    """
    Detect profile type from name keywords and phase structure.
    
    Used when no profile match is found - infers type from available info.
    
    Args:
        profile_name: Name of the profile
        phases: List of phases from profile
        
    Returns:
        Profile template dict
    """
    name_lower = profile_name.lower() if profile_name else ""
    
    # Check name for type indicators
    if any(kw in name_lower for kw in ["lever", "leva", "e61", "manual"]):
        return PROFILE_TEMPLATES["lever"]
    
    if any(kw in name_lower for kw in ["filter", "v60", "pour over"]):
        return PROFILE_TEMPLATES["filter"]
    
    if any(kw in name_lower for kw in ["flow", "turbo", "saturated"]):
        return PROFILE_TEMPLATES["flow_control"]
    
    if any(kw in name_lower for kw in ["pressure", "brew", "classic", "stock"]):
        return PROFILE_TEMPLATES["pressure"]
    
    # Infer from phase structure
    if phases:
        phase_types = [p.get("type", "").upper() for p in phases]
        if "FLOW" in phase_types:
            return PROFILE_TEMPLATES["flow_control"]
        if "PRESSURE" in phase_types and len(phases) == 1:
            return PROFILE_TEMPLATES["pressure"]
    
    return DEFAULT_TEMPLATE


# =========================
# MATH HELPERS
# =========================
def safe_mean(values):
    """Compute mean of list, return 0 for empty list."""
    return mean(values) if values else 0.0


def safe_max(values):
    """Get max of list, return 0 for empty list."""
    return max(values) if values else 0.0


def safe_min(values):
    """Get min of list, return 0 for empty list."""
    return min(values) if values else 0.0


def stddev(values):
    """Compute standard deviation of list."""
    if not values or len(values) < 2:
        return 0.0
    avg = safe_mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def nearest_time_index(times, target_s):
    """Find index in times array closest to target time."""
    if not times:
        return 0
    return min(range(len(times)), key=lambda i: abs(times[i] - target_s))


def window_by_time(times, values, start_s, end_s):
    """Extract values within a time window."""
    if not times or not values:
        return []
    return [v for ts, v in zip(times, values) if start_s <= ts <= end_s]


def window_indices_by_time(times, start_s, end_s):
    """Get indices of times within a window."""
    if not times:
        return []
    return [i for i, ts in enumerate(times) if start_s <= ts <= end_s]


def first_time_above(times, values, threshold, min_consecutive=2):
    """
    Find first time when value crosses threshold and stays there.
    
    Used to detect first drops (weight > 0.6g, flow > 0.3ml/s).
    Requires min_consecutive data points above threshold.
    """
    streak = 0
    first_idx = None
    for i, value in enumerate(values):
        if value >= threshold:
            if first_idx is None:
                first_idx = i
            streak += 1
            if streak >= min_consecutive:
                return times[first_idx]
        else:
            streak = 0
            first_idx = None
    return None


def slope_between(times, values, start_s, end_s):
    """Calculate slope (rate of change) between two times."""
    if not times or not values:
        return 0.0
    i0 = nearest_time_index(times, start_s)
    i1 = nearest_time_index(times, end_s)
    if i1 <= i0:
        return 0.0
    dt = times[i1] - times[i0]
    if dt <= 0:
        return 0.0
    return (values[i1] - values[i0]) / dt


def roundf(value, digits=2):
    """Round float to specified digits."""
    return round(float(value), digits)


# =========================
# PRE-INFUSION ANALYSIS
# =========================
def extract_preinfusion_phases(t, pressure, pump_flow, shot_weight, profile_template):
    """
    Analyze pre-infusion phase characteristics.
    
    Pre-infusion has three sub-phases:
    1. Wetting: Low pressure (<0.5 bar) - water contacts puck
    2. Soaking: Medium pressure - puck saturates
    3. Compression: Higher pressure - puck compresses, first drops appear
    
    Also calculates:
    - Total pre-infusion duration
    - Pressure rise rate (uniformity indicator)
    - Pre-infusion uniformity score
    - Pre-infusion type classification
    
    Args:
        t: Time series
        pressure: Pressure series
        pump_flow: Pump flow series
        shot_weight: Shot weight series
        profile_template: Profile thresholds for comparison
        
    Returns:
        Dict with pre-infusion metrics
    """
    if not t or not pressure:
        return {
            "wetting_duration_s": 0.0,
            "soaking_duration_s": 0.0,
            "compression_duration_s": 0.0,
            "total_preinfusion_s": 0.0,
            "pressure_rise_rate": 0.0,
            "preinfusion_uniformity": 0.0,
            "first_drops_s": 0.0,
            "preinfusion_type": "unknown",
        }
    
    # Find first drops (when shot weight or flow starts)
    first_drops = first_time_above(t, shot_weight, 0.6, min_consecutive=2)
    if first_drops is None:
        first_drops = first_time_above(t, pump_flow, 0.3, min_consecutive=2)
    if first_drops is None:
        first_drops = len(t) * 0.25 * (t[-1] / len(t)) if t else 0.0
    
    preinfusion_end = first_drops
    preinfusion_indices = window_indices_by_time(t, 0, preinfusion_end)
    
    if not preinfusion_indices:
        return {
            "wetting_duration_s": 0.0,
            "soaking_duration_s": 0.0,
            "compression_duration_s": 0.0,
            "total_preinfusion_s": roundf(preinfusion_end, 1),
            "pressure_rise_rate": 0.0,
            "preinfusion_uniformity": 0.0,
            "first_drops_s": roundf(first_drops, 1),
            "preinfusion_type": "unknown",
        }
    
    # Classify each pre-infusion moment
    wetting_duration = 0.0
    soaking_duration = 0.0
    compression_duration = 0.0
    
    pressure_values = [pressure[i] for i in preinfusion_indices]
    
    wetting_threshold = 0.5
    soaking_upper = profile_template.get("pressure_range", (6.0, 10.0))[1] * 0.25
    
    for i, idx in enumerate(preinfusion_indices):
        p = pressure[idx]
        if p < wetting_threshold:
            wetting_duration += t[idx + 1] - t[idx] if idx + 1 < len(t) else 0.1
        elif p < soaking_upper:
            soaking_duration += t[idx + 1] - t[idx] if idx + 1 < len(t) else 0.1
        else:
            compression_duration += t[idx + 1] - t[idx] if idx + 1 < len(t) else 0.1
    
    preinfusion_duration = preinfusion_end
    pressure_rise_rate = slope_between(t, pressure, 0, min(preinfusion_end, 10.0))
    
    # Calculate uniformity: 1.0 = perfectly uniform pressure, 0.0 = highly variable
    uniformity = 1.0
    if len(pressure_values) > 2:
        avg_pressure = safe_mean(pressure_values)
        uniformity = max(0.0, 1.0 - stddev(pressure_values) / (avg_pressure + 0.1))
    
    # Classify pre-infusion type
    if wetting_duration > preinfusion_duration * 0.5:
        preinfusion_type = "gentle"
    elif compression_duration > preinfusion_duration * 0.5:
        preinfusion_type = "aggressive"
    elif preinfusion_duration < 4.0:
        preinfusion_type = "fast"
    elif preinfusion_duration > 8.0:
        preinfusion_type = "slow"
    else:
        preinfusion_type = "standard"
    
    return {
        "wetting_duration_s": roundf(wetting_duration, 1),
        "soaking_duration_s": roundf(soaking_duration, 1),
        "compression_duration_s": roundf(compression_duration, 1),
        "total_preinfusion_s": roundf(preinfusion_duration, 1),
        "pressure_rise_rate": roundf(pressure_rise_rate, 3),
        "preinfusion_uniformity": roundf(uniformity, 2),
        "first_drops_s": roundf(first_drops, 1),
        "preinfusion_type": preinfusion_type,
    }


# =========================
# FLOW RATIO ANALYSIS
# =========================
def extract_flow_ratio_metrics(t, pump_flow, weight_flow, main_start_s, main_end_s):
    """
    Analyze flow ratio (weight_flow / pump_flow) for channeling detection.
    
    Flow ratio indicates how much pump flow actually makes it through the puck:
    - ratio ~1.0: Normal extraction (all pump flow becomes weight flow)
    - ratio >1.0: Channeling (water finds paths of least resistance)
    - ratio <1.0: Restriction (puck restricting flow, backed up pressure)
    
    Also calculates:
    - Flow ratio stability (stdev)
    - Channeling score (max ratio - 1.0)
    - Blockage score (1.0 - min ratio)
    - Flow ratio trend (increasing/decreasing/stable)
    """
    if not t or not pump_flow or not weight_flow:
        return {
            "avg_flow_ratio": 0.0,
            "flow_ratio_stdev": 0.0,
            "channeling_score": 0.0,
            "blockage_score": 0.0,
            "flow_ratio_trend": "unknown",
        }
    
    main_indices = window_indices_by_time(t, main_start_s, main_end_s)
    
    if not main_indices:
        return {
            "avg_flow_ratio": 0.0,
            "flow_ratio_stdev": 0.0,
            "channeling_score": 0.0,
            "blockage_score": 0.0,
            "flow_ratio_trend": "unknown",
        }
    
    # Calculate flow ratios for each moment
    ratios = []
    for idx in main_indices:
        pf = pump_flow[idx] if idx < len(pump_flow) else 0
        wf = weight_flow[idx] if idx < len(weight_flow) else 0
        if pf > 0.1:
            ratios.append(wf / pf)
        else:
            ratios.append(1.0)
    
    if not ratios:
        return {
            "avg_flow_ratio": 0.0,
            "flow_ratio_stdev": 0.0,
            "channeling_score": 0.0,
            "blockage_score": 0.0,
            "flow_ratio_trend": "unknown",
        }
    
    avg_ratio = safe_mean(ratios)
    ratio_stdev = stddev(ratios)
    
    # Channeling: high ratio means water rushing through channels
    channeling_score = max(0.0, max(ratios) - 1.0) if ratios else 0.0
    # Blockage: low ratio means flow backed up
    blockage_score = max(0.0, 1.0 - min(ratios)) if ratios else 0.0
    
    # Detect trend: increasing ratio = increasing channeling
    ratio_trend = "stable"
    if len(ratios) > 3:
        first_third = safe_mean(ratios[:len(ratios) // 3])
        last_third = safe_mean(ratios[-len(ratios) // 3:])
        if last_third > first_third * 1.15:
            ratio_trend = "increasing"
        elif last_third < first_third * 0.85:
            ratio_trend = "decreasing"
    
    return {
        "avg_flow_ratio": roundf(avg_ratio, 3),
        "flow_ratio_stdev": roundf(ratio_stdev, 3),
        "channeling_score": roundf(channeling_score, 3),
        "blockage_score": roundf(blockage_score, 3),
        "flow_ratio_trend": ratio_trend,
    }


# =========================
# EXTRACTION PROFILE CLASSIFICATION
# =========================
def classify_extraction_profile(events: list) -> str:
    """
    Classify extraction as balanced, fast, slow, channeling, or stalling.
    
    Based on detected event types:
    - balanced: Stable core + controlled tail
    - channeling: Fast opening + tail runaway/unstable flow
    - fast: Early drops + fast flow
    - slow/stalling: Late drops + restricted flow
    """
    event_types = {e["type"] for e in events}
    
    # Balanced: good core + tail
    if "stable_core" in event_types and "tail_controlled" in event_types:
        if not any(e in event_types for e in ["unstable_pressure", "unstable_flow", "tail_runaway", "possible_channeling"]):
            return "balanced"
    
    # Channeling: fast + unstable
    if any(e in event_types for e in ["early_first_drops", "fast_core_flow"]) and \
       any(e in event_types for e in ["tail_runaway", "unstable_flow", "possible_channeling"]):
        return "channeling"
    
    # Stalling: slow/restricted
    if "restricted_flow" in event_types or "late_first_drops" in event_types:
        if any(e in event_types for e in ["tail_controlled", "target_hit"]):
            return "slow"
        return "stalling"
    
    # Fast: early drops + runaway tail
    if any(e in event_types for e in ["early_first_drops", "fast_core_flow"]) and \
       "tail_runaway" in event_types:
        return "fast"
    
    # Slow: late drops or restricted
    if any(e in event_types for e in ["late_first_drops", "restricted_flow"]):
        return "slow"
    
    # Fast: early drops
    if any(e in event_types for e in ["early_first_drops", "fast_core_flow"]):
        return "fast"
    
    return "balanced"


# =========================
# PROFILE ADHERENCE SCORING
# =========================
def score_profile_adherence(features: dict, profile_template: dict) -> dict:
    """
    Score how well shot followed matched profile expectations.
    
    Checks:
    - Duration vs expected
    - Temperature vs target
    - Yield vs target
    - Pressure stability
    - Flow stability
    - Flow ratio
    - Pre-infusion uniformity
    
    Returns adherence score (50-100) and list of issues.
    """
    score = 100
    issues = []
    
    expected_duration = profile_template.get("expected_duration_s", 30)
    duration = features.get("duration_s", 0)
    filter_style = profile_template.get("filter_style", False)
    
    # Duration check (relaxed for filter profiles)
    if filter_style:
        if duration < expected_duration * 0.5:
            score -= 10
            issues.append("Shot ended much earlier than expected for filter profile")
        elif duration > expected_duration * 1.3:
            score -= 5
            issues.append("Shot ran longer than typical for filter profile")
    else:
        if duration < expected_duration * 0.5:
            score -= 10
            issues.append("Shot ended very early")
        elif duration > expected_duration * 1.5:
            score -= 5
            issues.append("Shot ran longer than expected")
    
    # Temperature check
    matched_temp = profile_template.get("matched_water_temp", 0)
    if matched_temp > 0:
        actual_temp = features.get("avg_temp_c", 0)
        if actual_temp > 0:
            temp_diff = abs(actual_temp - matched_temp)
            if temp_diff > 6:
                score -= 5
                issues.append(f"Temperature off by {temp_diff:.1f}C")
    
    # Yield check
    matched_weight = profile_template.get("matched_target_weight", 0)
    if matched_weight > 0:
        target_diff = features.get("target_diff_g", 0)
        if target_diff is not None and abs(target_diff) > 5:
            score -= 8
            issues.append(f"Yield off by {target_diff:.1f}g from target")
    
    # Pressure check
    pressure_range = profile_template.get("pressure_range", (7.0, 9.5))
    pressure_stdev = features.get("pressure_stdev_main", 0)
    pressure_stability_threshold = profile_template.get("pressure_stability_threshold", 0.7)
    avg_pressure = features.get("avg_pressure_main_bar", 0)
    
    if pressure_stdev > pressure_stability_threshold * 4.0:
        score -= 8
        issues.append(f"Pressure varied significantly ({pressure_stdev:.2f} stdev)")
    elif pressure_stdev > pressure_stability_threshold * 3.0:
        score -= 3
        issues.append(f"Pressure slightly variable ({pressure_stdev:.2f} stdev)")
    
    if avg_pressure < pressure_range[0] * 0.6:
        score -= 8
        issues.append(f"Avg pressure {avg_pressure:.1f} bar below target {pressure_range[0]} bar")
    elif avg_pressure > pressure_range[1] * 1.4:
        score -= 3
        issues.append(f"Avg pressure {avg_pressure:.1f} bar above target {pressure_range[1]} bar")
    
    # Flow stability check
    flow_stdev = features.get("pump_flow_stdev_main", 0)
    flow_stability_threshold = profile_template.get("flow_stability_threshold", 0.45)
    if flow_stdev > flow_stability_threshold * 4.0:
        score -= 8
        issues.append(f"Flow variable ({flow_stdev:.2f} stdev)")
    elif flow_stdev > flow_stability_threshold * 3.0:
        score -= 3
        issues.append(f"Flow slightly variable ({flow_stdev:.2f} stdev)")
    
    # Flow ratio check
    flow_ratio = features.get("avg_flow_ratio", 1.0)
    if flow_ratio < 0.4:
        score -= 5
        issues.append(f"Flow ratio {flow_ratio:.2f} indicates restriction")
    elif flow_ratio > 1.8:
        score -= 5
        issues.append(f"Flow ratio {flow_ratio:.2f} indicates channeling")
    
    # Pre-infusion uniformity check
    preinfusion_uniformity = features.get("preinfusion", {}).get("preinfusion_uniformity", 1.0)
    if preinfusion_uniformity < 0.25:
        score -= 5
        issues.append(f"Pre-infusion uneven ({preinfusion_uniformity:.2f} uniformity)")
    elif preinfusion_uniformity < 0.4:
        score -= 3
        issues.append("Pre-infusion could be more uniform")
    
    # Filter profile specific
    profile_name = profile_template.get("matched_profile_name", "")
    if profile_name and "filter" in profile_name.lower():
        if features.get("duration_s", 0) < 60:
            score -= 5
            issues.append("Filter profile: shorter than typical")
    
    score = max(50, min(100, score))
    
    return {
        "adherence_score": score,
        "adherence_issues": issues[:3],
    }


# =========================
# TASTE-BASED SCORING
# =========================
def taste_based_scoring(features: dict, events: list, profile_template: dict) -> dict:
    """
    Predict taste profile and calculate score modifier.
    
    Counts signals for:
    - Under-extraction (fast flow, early drops, high flow ratio)
    - Over-extraction (slow flow, late drops, low flow ratio)
    - Balanced extraction (on target, stable, controlled)
    
    Returns taste profile and score modifier.
    """
    under_extra = 0.0
    over_extra = 0.0
    balanced_count = 0
    
    # Flow rate analysis
    avg_flow = features.get("avg_pump_flow_main_ml_s", 2.0)
    min_flow = profile_template.get("min_flow", 0.8)
    max_flow = profile_template.get("max_flow", 2.8)
    target_flow = profile_template.get("target_flow", 2.0)
    
    if avg_flow > max_flow * 1.3:
        under_extra += 1.5
    elif avg_flow > max_flow * 1.15:
        under_extra += 0.5
    elif avg_flow < min_flow * 0.7:
        over_extra += 1.5
    elif avg_flow < min_flow * 0.85:
        over_extra += 0.5
    else:
        balanced_count += 1
    
    # First drops timing
    first_drops = features.get("first_drops_s", 6.0)
    early_threshold = profile_template.get("first_drops_early", 4.5)
    late_threshold = profile_template.get("first_drops_late", 8.0)
    
    if first_drops < early_threshold - 2.5:
        under_extra += 1
    elif first_drops < early_threshold - 1.0:
        under_extra += 0.5
    elif first_drops > late_threshold + 4.0:
        over_extra += 1
    elif first_drops > late_threshold + 2.0:
        over_extra += 0.5
    else:
        balanced_count += 1
    
    # End flow slope
    end_flow_slope = features.get("end_flow_slope", 0.0)
    if end_flow_slope > 0.35:
        under_extra += 1
    elif end_flow_slope > 0.25:
        under_extra += 0.5
    elif end_flow_slope < -0.3:
        balanced_count += 0.5
    
    # Flow ratio
    flow_ratio = features.get("avg_flow_ratio", 1.0)
    if flow_ratio > 1.6:
        under_extra += 1
    elif flow_ratio > 1.4:
        under_extra += 0.5
    elif flow_ratio < 0.4:
        over_extra += 1
    elif flow_ratio < 0.55:
        over_extra += 0.5
    
    # Target hit bonus
    if features.get("target_hit"):
        balanced_count += 3
    
    # Stable core bonus
    if features.get("stable_core"):
        balanced_count += 2
    
    # Pressure stability
    pressure_stdev = features.get("pressure_stdev_main", 1.0)
    stability_threshold = profile_template.get("pressure_stability_threshold", 0.7)
    if pressure_stdev <= stability_threshold:
        balanced_count += 1
    elif pressure_stdev > stability_threshold * 3.5:
        under_extra += 0.5
    
    # Calculate net score
    net = balanced_count - (under_extra + over_extra)
    
    if net >= 1:
        taste_profile = "well_extracted"
        score_modifier = 0
    elif net >= -1:
        taste_profile = "mostly_balanced"
        score_modifier = -3
    elif net >= -3:
        taste_profile = "slightly_off"
        score_modifier = -8
    elif net >= -5:
        taste_profile = "unbalanced"
        score_modifier = -15
    else:
        taste_profile = "poorly_extracted"
        score_modifier = -20
    
    return {
        "taste_profile": taste_profile,
        "score_modifier": score_modifier,
        "under_extraction_signals": under_extra,
        "over_extraction_signals": over_extra,
        "balanced_signals": balanced_count,
    }


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(shot_data, series):
    """
    Extract all quantitative features from shot telemetry.
    
    This is the main entry point for feature extraction. It:
    1. Loads and matches profiles
    2. Calculates phase boundaries
    3. Extracts pre-infusion metrics
    4. Calculates flow ratios
    5. Scores profile adherence
    6. Returns complete features dict
    
    Args:
        shot_data: Full shot data from Gaggiuino API
        series: Preprocessed telemetry series from plot_logic
        
    Returns:
        Dict with all extracted features
    """
    t = series["time_s"]
    pressure = series["pressure_bar"]
    pump_flow = series["pump_flow_ml_s"]
    weight_flow = series["weight_flow_g_s"]
    shot_weight = series["shot_weight_g"]
    temp = series["temp_c"]

    profile = shot_data.get("profile", {})
    phases = profile.get("phases", [])
    duration_raw = shot_data.get("duration", 0)
    duration_s = duration_raw / 10.0 if duration_raw else (t[-1] if t else 0.0)
    final_weight = shot_weight[-1] if shot_weight else 0.0
    target_weight = profile.get("globalStopConditions", {}).get("weight")

    # Match profile or detect type
    matched = get_matched_profile(shot_data)
    matched_profile = matched["profile"]
    profile_match_type = matched["match_type"]
    profile_match_confidence = matched["confidence"]
    
    if matched_profile:
        profile_template = get_profile_template_from_matched(matched_profile)
    else:
        profile_template = detect_profile_type(profile.get("name", ""), phases)
    
    # Extract pre-infusion metrics
    preinfusion = extract_preinfusion_phases(t, pressure, pump_flow, shot_weight, profile_template)
    first_drops_s = preinfusion["first_drops_s"]
    
    # Define phase boundaries
    main_start_s = min(max(first_drops_s + 1.5, duration_s * 0.25), duration_s)
    main_end_s = min(max(main_start_s + 4.0, duration_s * 0.8), duration_s)
    tail_start_s = min(max(duration_s * 0.8, main_end_s - 2.0), duration_s)

    # Extract data for each phase
    main_pressure = window_by_time(t, pressure, main_start_s, main_end_s)
    main_flow = window_by_time(t, pump_flow, main_start_s, main_end_s)
    main_weight_flow = window_by_time(t, weight_flow, main_start_s, main_end_s)
    tail_pressure = window_by_time(t, pressure, tail_start_s, duration_s)
    tail_flow = window_by_time(t, pump_flow, tail_start_s, duration_s)
    tail_weight_flow = window_by_time(t, weight_flow, tail_start_s, duration_s)

    # Peak values
    peak_pressure = safe_max(pressure)
    peak_pressure_time = t[pressure.index(peak_pressure)] if pressure else 0.0
    peak_pump_flow = safe_max(pump_flow)
    peak_flow_time = t[pump_flow.index(peak_pump_flow)] if pump_flow else 0.0

    # Yield accuracy
    target_diff_g = None
    stop_accuracy_g = None
    target_hit = False
    if isinstance(target_weight, (int, float)):
        target_diff_g = final_weight - float(target_weight)
        stop_accuracy_g = abs(target_diff_g)
        target_hit = stop_accuracy_g <= 1.5

    # Flow slopes
    end_flow_slope = slope_between(t, pump_flow, max(0.0, duration_s - 6.0), duration_s)
    end_weight_flow_slope = slope_between(t, weight_flow, max(0.0, duration_s - 6.0), duration_s)
    end_pressure_slope = slope_between(t, pressure, max(0.0, duration_s - 6.0), duration_s)
    pressure_ramp_bar_s = slope_between(t, pressure, 0.0, min(8.0, duration_s))
    flow_ramp_ml_s2 = slope_between(t, pump_flow, 0.0, min(8.0, duration_s))
    
    # Flow ratio analysis
    flow_ratio_metrics = extract_flow_ratio_metrics(t, pump_flow, weight_flow, main_start_s, main_end_s)

    # Profile adherence scoring
    adherence = score_profile_adherence({
        "duration_s": duration_s,
        "avg_temp_c": safe_mean(temp),
        "target_diff_g": target_diff_g,
        "avg_pressure_main_bar": safe_mean(main_pressure),
        "pressure_stdev_main": stddev(main_pressure),
        "pump_flow_stdev_main": stddev(main_flow),
        "avg_flow_ratio": flow_ratio_metrics["avg_flow_ratio"],
        "preinfusion": preinfusion,
    }, profile_template)

    # Build features dict
    features = {
        "profile_name": profile.get("name", "Unknown Profile"),
        "profile_type": "pressure",
        "water_temp_target_c": profile.get("waterTemperature"),
        "duration_s": roundf(duration_s, 1),
        "final_weight_g": roundf(final_weight, 1),
        "target_weight_g": target_weight,
        "target_diff_g": roundf(target_diff_g, 1) if target_diff_g is not None else None,
        "stop_accuracy_g": roundf(stop_accuracy_g, 1) if stop_accuracy_g is not None else None,
        "target_hit": target_hit,
        "first_drops_s": roundf(first_drops_s, 1),
        "main_start_s": roundf(main_start_s, 1),
        "main_end_s": roundf(main_end_s, 1),
        "tail_start_s": roundf(tail_start_s, 1),
        "peak_pressure_bar": roundf(peak_pressure, 1),
        "peak_pressure_time_s": roundf(peak_pressure_time, 1),
        "avg_pressure_main_bar": roundf(safe_mean(main_pressure), 2),
        "pressure_stdev_main": roundf(stddev(main_pressure), 2),
        "avg_pump_flow_main_ml_s": roundf(safe_mean(main_flow), 2),
        "pump_flow_stdev_main": roundf(stddev(main_flow), 2),
        "avg_weight_flow_main_g_s": roundf(safe_mean(main_weight_flow), 2),
        "weight_flow_stdev_main": roundf(stddev(main_weight_flow), 2),
        "tail_avg_pump_flow_ml_s": roundf(safe_mean(tail_flow), 2),
        "tail_avg_weight_flow_g_s": roundf(safe_mean(tail_weight_flow), 2),
        "tail_avg_pressure_bar": roundf(safe_mean(tail_pressure), 2),
        "pressure_ramp_bar_s": roundf(pressure_ramp_bar_s, 2),
        "flow_ramp_ml_s2": roundf(flow_ramp_ml_s2, 2),
        "end_flow_slope": roundf(end_flow_slope, 2),
        "end_weight_flow_slope": roundf(end_weight_flow_slope, 2),
        "end_pressure_slope": roundf(end_pressure_slope, 2),
        "avg_temp_c": roundf(safe_mean(temp), 1),
        "min_temp_c": roundf(safe_min(temp), 1),
        "max_temp_c": roundf(safe_max(temp), 1),
        "peak_pump_flow_ml_s": roundf(peak_pump_flow, 2),
        "peak_flow_time_s": roundf(peak_flow_time, 1),
        "stable_core": False,
        "avg_flow_ratio": flow_ratio_metrics["avg_flow_ratio"],
        "flow_ratio_stdev": flow_ratio_metrics["flow_ratio_stdev"],
        "channeling_score": flow_ratio_metrics["channeling_score"],
        "flow_ratio_trend": flow_ratio_metrics["flow_ratio_trend"],
        "preinfusion": preinfusion,
        "profile_match_type": profile_match_type,
        "profile_match_confidence": profile_match_confidence,
        "matched_profile_name": profile_template.get("matched_profile_name", ""),
        "profile_adherence_score": adherence["adherence_score"],
        "profile_adherence_issues": adherence["adherence_issues"],
    }
    
    # Set profile type
    if matched_profile:
        if profile_template.get("filter_style"):
            features["profile_type"] = "filter"
        elif "lever" in profile_template.get("matched_profile_name", "").lower():
            features["profile_type"] = "lever"
        elif "flow" in profile_template.get("matched_profile_name", "").lower():
            features["profile_type"] = "flow_control"
        else:
            features["profile_type"] = "pressure"
    else:
        if profile_template == PROFILE_TEMPLATES["lever"]:
            features["profile_type"] = "lever"
        elif profile_template == PROFILE_TEMPLATES["flow_control"]:
            features["profile_type"] = "flow_control"
        elif profile_template == PROFILE_TEMPLATES["filter"]:
            features["profile_type"] = "filter"

    return features


# =========================
# EVENT DETECTION
# =========================
def _severity_rank(severity):
    """Get numeric rank for severity (lower = more important)."""
    return SEVERITY_ORDER.get(severity, 1)


def add_event(events, event_type, time_s, severity, reason, metric=None):
    """Helper to add event to list."""
    events.append({
        "type": event_type,
        "time": roundf(time_s, 1),
        "severity": severity,
        "reason": reason,
        "metric": metric,
    })


def detect_events(features):
    """
    Detect significant extraction events from features.
    
    Events are timestamped moments with:
    - Type (e.g., "late_first_drops", "stable_core")
    - Time (when it occurred)
    - Severity (good, info, warning, critical)
    - Reason (human-readable explanation)
    - Metric (the quantitative value that triggered detection)
    
    These events are used both for fallback analysis and as anchors
    for LLM annotation placement.
    """
    events = []
    
    # Get profile-specific thresholds
    profile_template_name = features.get("profile_type", "pressure")
    profile_template = PROFILE_TEMPLATES.get(profile_template_name, DEFAULT_TEMPLATE)
    
    # Extract key features
    duration = features["duration_s"]
    first_drops = features["first_drops_s"]
    avg_pressure_main = features["avg_pressure_main_bar"]
    pressure_stdev = features["pressure_stdev_main"]
    avg_flow_main = features["avg_pump_flow_main_ml_s"]
    flow_stdev = features["pump_flow_stdev_main"]
    end_flow_slope = features["end_flow_slope"]
    end_weight_flow_slope = features["end_weight_flow_slope"]
    target_diff = features.get("target_diff_g")
    stop_accuracy = features.get("stop_accuracy_g")
    peak_pressure = features["peak_pressure_bar"]
    tail_start = features["tail_start_s"]
    peak_pressure_time = features["peak_pressure_time_s"]
    main_mid = (features["main_start_s"] + features["main_end_s"]) / 2.0
    flow_ratio = features.get("avg_flow_ratio", 1.0)
    channeling_score = features.get("channeling_score", 0.0)
    
    preinfusion = features.get("preinfusion", {})
    preinfusion_type = preinfusion.get("preinfusion_type", "standard")
    preinfusion_uniformity = preinfusion.get("preinfusion_uniformity", 1.0)
    
    # Get thresholds
    early_threshold = profile_template.get("first_drops_early", 4.5)
    late_threshold = profile_template.get("first_drops_late", 8.0)
    min_flow = profile_template.get("min_flow", 0.8)
    max_flow = profile_template.get("max_flow", 2.8)
    pressure_range = profile_template.get("pressure_range", (7.0, 9.5))
    flow_stability_threshold = profile_template.get("flow_stability_threshold", 0.45)
    pressure_stability_threshold = profile_template.get("pressure_stability_threshold", 0.7)
    
    # === FIRST DROPS TIMING ===
    if first_drops >= late_threshold:
        add_event(events, "late_first_drops", first_drops, "warning",
                 "First drops arrived late, suggesting a tight puck or conservative opening.",
                 metric=first_drops)
    elif first_drops <= early_threshold:
        add_event(events, "early_first_drops", first_drops, "info",
                 "First drops arrived early, suggesting a fast opening or coarse puck.",
                 metric=first_drops)
    else:
        add_event(events, "first_drops_on_time", first_drops, "good",
                 "First drops timing landed in a healthy window.",
                 metric=first_drops)
    
    # === PRE-INFUSION TYPE ===
    if preinfusion_type == "slow":
        add_event(events, "slow_preinfusion", preinfusion.get("total_preinfusion_s", 0), "info",
                 "Pre-infusion took longer than typical.",
                 metric=preinfusion.get("total_preinfusion_s"))
    elif preinfusion_type == "fast":
        add_event(events, "fast_preinfusion", preinfusion.get("total_preinfusion_s", 0), "info",
                 "Pre-infusion was brief.",
                 metric=preinfusion.get("total_preinfusion_s"))
    
    if preinfusion_uniformity < 0.5 and duration > 15:
        add_event(events, "uneven_preinfusion", first_drops / 2, "warning",
                 "Pre-infusion pressure was uneven, may indicate channeling risk.",
                 metric=preinfusion_uniformity)
    
    # === PRESSURE PEAKS ===
    peak_upper = pressure_range[1]
    peak_lower = pressure_range[0]
    
    if peak_pressure >= peak_upper:
        add_event(events, "high_peak_pressure", peak_pressure_time, "warning",
                 "Peak pressure ran high and may increase harshness or channel risk.",
                 metric=peak_pressure)
    elif peak_pressure < peak_lower * 0.8 and duration > 15:
        add_event(events, "low_peak_pressure", peak_pressure_time, "info",
                 "Peak pressure stayed modest.",
                 metric=peak_pressure)
    
    # === MAIN EXTRACTION STABILITY ===
    flow_upper = max_flow
    flow_lower = min_flow
    
    if avg_pressure_main >= pressure_range[0] * 0.9 and pressure_stdev <= pressure_stability_threshold and \
       flow_lower <= avg_flow_main <= flow_upper and flow_stdev <= flow_stability_threshold:
        features["stable_core"] = True
        add_event(events, "stable_core", main_mid, "good",
                 "Main extraction stayed controlled with stable pressure and flow.",
                 metric=avg_pressure_main)
    else:
        if pressure_stdev > pressure_stability_threshold * 1.5:
            add_event(events, "unstable_pressure", main_mid, "warning",
                     "Pressure moved around more than ideal during the body of the shot.",
                     metric=pressure_stdev)
        if flow_stdev > flow_stability_threshold * 1.5:
            add_event(events, "unstable_flow", main_mid, "warning",
                     "Flow varied notably during the main extraction.",
                     metric=flow_stdev)
        if avg_flow_main < min_flow:
            add_event(events, "restricted_flow", main_mid, "warning",
                     "Main extraction flow stayed quite low, suggesting a restrictive puck.",
                     metric=avg_flow_main)
        elif avg_flow_main > max_flow:
            add_event(events, "fast_core_flow", main_mid, "info",
                     "Main extraction flow ran fast, which can thin body.",
                     metric=avg_flow_main)
    
    # === CHANNELING ===
    if flow_ratio > 1.4 and channeling_score > 0.3:
        add_event(events, "possible_channeling", main_mid, "warning",
                 "Flow ratio suggests water may have found paths of least resistance.",
                 metric=flow_ratio)
    
    # === TAIL BEHAVIOR ===
    if end_flow_slope > 0.20 or end_weight_flow_slope > 0.20:
        add_event(events, "tail_runaway", tail_start, "warning",
                 "Flow accelerated late in the shot, a sign of puck weakening.",
                 metric=max(end_flow_slope, end_weight_flow_slope))
    elif end_flow_slope < -0.15 and avg_flow_main > 0:
        add_event(events, "tail_controlled", tail_start, "good",
                 "Flow tapered down cleanly in the final phase.",
                 metric=end_flow_slope)
    
    # === YIELD ACCURACY ===
    if stop_accuracy is not None:
        if stop_accuracy <= 1.5:
            add_event(events, "target_hit", duration, "good",
                     "Shot stopped very close to the target yield.",
                     metric=target_diff)
        elif target_diff is not None and target_diff < -2.0:
            add_event(events, "stopped_early", duration, "warning",
                     "Shot ended noticeably before the target yield.",
                     metric=target_diff)
        elif target_diff is not None and target_diff > 2.0:
            add_event(events, "ran_past_target", duration, "info",
                     "Shot ran past the target yield.",
                     metric=target_diff)
    
    # === PROFILE ADHERENCE ===
    adherence_score = features.get("profile_adherence_score", 100)
    if adherence_score < 70:
        add_event(events, "poor_profile_adherence", duration * 0.5, "warning",
                 "Shot deviated significantly from profile expectations.",
                 metric=adherence_score)

    # Deduplicate by type (keep highest severity)
    deduped = {}
    for event in events:
        key = event["type"]
        existing = deduped.get(key)
        if existing is None or _severity_rank(event["severity"]) > _severity_rank(existing["severity"]):
            deduped[key] = event
    
    # Sort by time and limit to 9 events
    ordered = sorted(deduped.values(), key=lambda e: e["time"])
    return ordered[:9]


# =========================
# TENDENCY CLASSIFICATION
# =========================
def classify_extraction_tendency(features, events):
    """
    Classify extraction tendency and calculate score hint.
    
    Combines:
    - Taste-based scoring (under/over extraction signals)
    - Profile adherence score
    - Individual metric deviations
    
    Returns score hint (0-100), confidence, tendency description.
    """
    profile_template_name = features.get("profile_type", "pressure")
    profile_template = PROFILE_TEMPLATES.get(profile_template_name, DEFAULT_TEMPLATE)
    
    # Get taste profile
    taste_result = taste_based_scoring(features, events, profile_template)
    
    # Classify extraction profile
    extraction_profile = classify_extraction_profile(events)
    
    # Get adherence score
    adherence_score = features.get("profile_adherence_score", 100)
    
    # Calculate base score
    score = 80
    score += taste_result["score_modifier"]
    score += int((adherence_score - 80) / 10)
    reasons = []
    
    # Get thresholds
    early_threshold = profile_template.get("first_drops_early", 4.5)
    late_threshold = profile_template.get("first_drops_late", 8.0)
    min_flow = profile_template.get("min_flow", 0.8)
    max_flow = profile_template.get("max_flow", 2.8)
    pressure_stability_threshold = profile_template.get("pressure_stability_threshold", 0.7)
    flow_stability_threshold = profile_template.get("flow_stability_threshold", 0.45)
    
    # First drops bonus/penalty
    if features["first_drops_s"] >= late_threshold + 2:
        score -= 5
        reasons.append("late first drops")
    elif features["first_drops_s"] <= early_threshold - 2.0:
        score -= 3
        reasons.append("early first drops")
    else:
        score += 2
    
    # Pressure stability
    if features["pressure_stdev_main"] <= pressure_stability_threshold:
        score += 3
    elif features["pressure_stdev_main"] <= pressure_stability_threshold * 2.0:
        pass
    else:
        score -= 3
        if "pressure instability" not in reasons:
            reasons.append("pressure instability")
    
    # Flow stability
    if features["pump_flow_stdev_main"] <= flow_stability_threshold:
        score += 3
    elif features["pump_flow_stdev_main"] <= flow_stability_threshold * 2.0:
        pass
    else:
        score -= 3
        if "flow instability" not in reasons:
            reasons.append("flow instability")
    
    # Flow rate
    if features["avg_pump_flow_main_ml_s"] < min_flow * 0.75:
        score -= 5
        if "restricted flow" not in reasons:
            reasons.append("restricted flow")
    elif features["avg_pump_flow_main_ml_s"] > max_flow * 1.25:
        score -= 3
        if "fast main flow" not in reasons:
            reasons.append("fast main flow")
    else:
        score += 2
    
    # Pressure peak
    peak_upper = profile_template.get("pressure_range", (7.0, 9.5))[1]
    if features["peak_pressure_bar"] >= peak_upper * 1.3:
        score -= 5
        reasons.append("high peak pressure")
    elif 7.5 <= features["avg_pressure_main_bar"] <= 9.5:
        score += 2
    
    # Target hit bonus
    if features.get("target_hit"):
        score += 10
    elif features.get("stop_accuracy_g") is not None:
        acc = features.get("stop_accuracy_g")
        if acc <= 3.0:
            score -= 1
        else:
            score -= 4
            if "yield missed" not in reasons:
                reasons.append("yield missed")
    
    # Tail behavior
    if features["end_flow_slope"] > 0.30 or features["end_weight_flow_slope"] > 0.25:
        score -= 5
        if "late tail acceleration" not in reasons:
            reasons.append("late tail acceleration")
    elif features["end_flow_slope"] < -0.15:
        score += 1
    
    # Flow ratio
    flow_ratio = features.get("avg_flow_ratio", 1.0)
    if flow_ratio > 1.6:
        score -= 3
        reasons.append("possible channeling")
    elif flow_ratio < 0.5:
        score -= 3
        reasons.append("flow restriction")
    
    # Severity penalties
    severity_penalty = sum(2 for e in events if e["severity"] == "warning") + sum(4 for e in events if e["severity"] == "critical")
    score -= severity_penalty
    score = max(25, min(98, score))
    
    # Tendency description
    tendency = "balanced"
    if features["first_drops_s"] >= late_threshold or features["avg_pump_flow_main_ml_s"] < min_flow * 1.1:
        tendency = "overextracting / restrictive"
    elif features["first_drops_s"] <= early_threshold or features["avg_pump_flow_main_ml_s"] > max_flow * 0.9 or (features.get("target_diff_g") or 0) > 2.0:
        tendency = "underextracting / fast"
    
    # Confidence (lower for short shots, few events, variable data)
    confidence = 0.9
    if len(events) < 3:
        confidence -= 0.08
    if features["duration_s"] < 12:
        confidence -= 0.12
    if features["weight_flow_stdev_main"] > 0.7:
        confidence -= 0.05
    if features.get("preinfusion", {}).get("preinfusion_uniformity", 1.0) < 0.6:
        confidence -= 0.08
    if features.get("profile_match_confidence", 0) > 0.5:
        confidence += 0.05

    return {
        "score_hint": int(round(score)),
        "confidence_hint": round(max(0.50, min(0.97, confidence)), 2),
        "tendency": tendency,
        "score_reasons": reasons[:4],
        "taste_profile": taste_result["taste_profile"],
        "extraction_profile": extraction_profile,
        "under_extraction_signals": taste_result["under_extraction_signals"],
        "over_extraction_signals": taste_result["over_extraction_signals"],
        "balanced_signals": taste_result["balanced_signals"],
        "profile_adherence_score": adherence_score,
        "profile_match_type": features.get("profile_match_type", "none"),
        "profile_match_confidence": features.get("profile_match_confidence", 0),
        "matched_profile_name": features.get("matched_profile_name", ""),
    }


# =========================
# PROMPT SUMMARIZATION
# =========================
def summarize_for_prompt(features, events, heuristic):
    """
    Create a compact summary of analysis for LLM context window.
    
    Packages features, events, and heuristic into a single dict
    for inclusion in the LLM prompt.
    """
    return {
        "features": features,
        "detected_events": events,
        "heuristic": heuristic,
    }
