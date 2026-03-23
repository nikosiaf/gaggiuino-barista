import math
from statistics import mean

SEVERITY_ORDER = {"good": 0, "info": 1, "warning": 2, "critical": 3}


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_max(values):
    return max(values) if values else 0.0


def safe_min(values):
    return min(values) if values else 0.0


def stddev(values):
    if not values or len(values) < 2:
        return 0.0
    avg = safe_mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def nearest_time_index(times, target_s):
    if not times:
        return 0
    return min(range(len(times)), key=lambda i: abs(times[i] - target_s))


def window_by_time(times, values, start_s, end_s):
    if not times or not values:
        return []
    return [v for ts, v in zip(times, values) if start_s <= ts <= end_s]


def first_time_above(times, values, threshold, min_consecutive=2):
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
    return round(float(value), digits)


def extract_features(shot_data, series):
    t = series["time_s"]
    pressure = series["pressure_bar"]
    pump_flow = series["pump_flow_ml_s"]
    weight_flow = series["weight_flow_g_s"]
    shot_weight = series["shot_weight_g"]
    temp = series["temp_c"]

    profile = shot_data.get("profile", {})
    duration_raw = shot_data.get("duration", 0)
    duration_s = duration_raw / 10.0 if duration_raw else (t[-1] if t else 0.0)
    final_weight = shot_weight[-1] if shot_weight else 0.0
    target_weight = profile.get("globalStopConditions", {}).get("weight")

    first_drops_s = first_time_above(t, shot_weight, 0.6, min_consecutive=2)
    if first_drops_s is None:
        first_drops_s = first_time_above(t, weight_flow, 0.25, min_consecutive=2)
    if first_drops_s is None:
        first_drops_s = duration_s * 0.25 if duration_s else 0.0

    main_start_s = min(max(first_drops_s + 1.5, duration_s * 0.25), duration_s)
    main_end_s = min(max(main_start_s + 4.0, duration_s * 0.8), duration_s)
    tail_start_s = min(max(duration_s * 0.8, main_end_s - 2.0), duration_s)

    main_pressure = window_by_time(t, pressure, main_start_s, main_end_s)
    main_flow = window_by_time(t, pump_flow, main_start_s, main_end_s)
    main_weight_flow = window_by_time(t, weight_flow, main_start_s, main_end_s)
    tail_pressure = window_by_time(t, pressure, tail_start_s, duration_s)
    tail_flow = window_by_time(t, pump_flow, tail_start_s, duration_s)
    tail_weight_flow = window_by_time(t, weight_flow, tail_start_s, duration_s)

    peak_pressure = safe_max(pressure)
    peak_pressure_time = t[pressure.index(peak_pressure)] if pressure else 0.0
    peak_pump_flow = safe_max(pump_flow)
    peak_flow_time = t[pump_flow.index(peak_pump_flow)] if pump_flow else 0.0

    target_diff_g = None
    stop_accuracy_g = None
    if isinstance(target_weight, (int, float)):
        target_diff_g = final_weight - float(target_weight)
        stop_accuracy_g = abs(target_diff_g)

    end_flow_slope = slope_between(t, pump_flow, max(0.0, duration_s - 6.0), duration_s)
    end_weight_flow_slope = slope_between(t, weight_flow, max(0.0, duration_s - 6.0), duration_s)
    end_pressure_slope = slope_between(t, pressure, max(0.0, duration_s - 6.0), duration_s)
    pressure_ramp_bar_s = slope_between(t, pressure, 0.0, min(8.0, duration_s))
    flow_ramp_ml_s2 = slope_between(t, pump_flow, 0.0, min(8.0, duration_s))

    features = {
        "profile_name": profile.get("name", "Unknown Profile"),
        "water_temp_target_c": profile.get("waterTemperature"),
        "duration_s": roundf(duration_s, 1),
        "final_weight_g": roundf(final_weight, 1),
        "target_weight_g": target_weight,
        "target_diff_g": roundf(target_diff_g, 1) if target_diff_g is not None else None,
        "stop_accuracy_g": roundf(stop_accuracy_g, 1) if stop_accuracy_g is not None else None,
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
    }

    return features


def _severity_rank(severity):
    return SEVERITY_ORDER.get(severity, 1)


def add_event(events, event_type, time_s, severity, reason, metric=None):
    events.append({
        "type": event_type,
        "time": roundf(time_s, 1),
        "severity": severity,
        "reason": reason,
        "metric": metric,
    })


def detect_events(features):
    events = []
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

    if first_drops >= 8.0:
        add_event(events, "late_first_drops", first_drops, "warning", "First drops arrived late, suggesting a tight puck or conservative opening.", metric=first_drops)
    elif first_drops <= 4.5:
        add_event(events, "early_first_drops", first_drops, "info", "First drops arrived very early, suggesting a fast opening or coarse puck prep.", metric=first_drops)
    else:
        add_event(events, "first_drops_on_time", first_drops, "good", "First drops timing landed in a healthy window.", metric=first_drops)

    if peak_pressure >= 10.0:
        add_event(events, "high_peak_pressure", peak_pressure_time, "warning", "Peak pressure ran high and may increase harshness or channel risk.", metric=peak_pressure)
    elif peak_pressure < 7.0 and duration > 15:
        add_event(events, "low_peak_pressure", peak_pressure_time, "info", "Peak pressure stayed modest, which can fit turbo-style shots but may under-drive classic espresso.", metric=peak_pressure)

    if avg_pressure_main >= 8.0 and pressure_stdev <= 0.7 and avg_flow_main >= 1.0 and avg_flow_main <= 2.5 and flow_stdev <= 0.45:
        add_event(events, "stable_core", main_mid, "good", "Main extraction stayed controlled with stable pressure and flow.", metric=avg_pressure_main)
    else:
        if pressure_stdev > 1.0:
            add_event(events, "unstable_pressure", main_mid, "warning", "Pressure moved around more than ideal during the body of the shot.", metric=pressure_stdev)
        if flow_stdev > 0.65:
            add_event(events, "unstable_flow", main_mid, "warning", "Flow varied notably during the main extraction, hinting at uneven puck resistance.", metric=flow_stdev)
        if avg_flow_main < 0.8:
            add_event(events, "restricted_flow", main_mid, "warning", "Main extraction flow stayed quite low, suggesting a restrictive puck.", metric=avg_flow_main)
        elif avg_flow_main > 2.8:
            add_event(events, "fast_core_flow", main_mid, "info", "Main extraction flow ran fast, which can thin body and shorten contact time.", metric=avg_flow_main)

    if end_flow_slope > 0.18 or end_weight_flow_slope > 0.18:
        add_event(events, "tail_runaway", tail_start, "warning", "Flow accelerated late in the shot, a common sign of puck weakening or blonding.", metric=max(end_flow_slope, end_weight_flow_slope))
    elif end_flow_slope < -0.15 and avg_flow_main > 0:
        add_event(events, "tail_controlled", tail_start, "good", "Flow tapered down cleanly in the final phase.", metric=end_flow_slope)

    if stop_accuracy is not None:
        if stop_accuracy <= 1.5:
            add_event(events, "target_hit", duration, "good", "Shot stopped very close to the target yield.", metric=target_diff)
        elif target_diff is not None and target_diff < -2.0:
            add_event(events, "stopped_early", duration, "warning", "Shot ended noticeably before the target yield.", metric=target_diff)
        elif target_diff is not None and target_diff > 2.0:
            add_event(events, "ran_past_target", duration, "info", "Shot ran past the target yield, which can flatten sweetness in the tail.", metric=target_diff)

    deduped = {}
    for event in events:
        key = event["type"]
        existing = deduped.get(key)
        if existing is None or _severity_rank(event["severity"]) > _severity_rank(existing["severity"]):
            deduped[key] = event
    ordered = sorted(deduped.values(), key=lambda e: e["time"])
    return ordered[:6]


def classify_extraction_tendency(features, events):
    score = 80
    reasons = []

    if features["first_drops_s"] >= 8.0:
        score -= 8
        reasons.append("late first drops")
    elif features["first_drops_s"] <= 4.5:
        score -= 4
        reasons.append("very early first drops")
    else:
        score += 3

    if features["pressure_stdev_main"] <= 0.7:
        score += 4
    else:
        score -= min(10, int(features["pressure_stdev_main"] * 4))
        reasons.append("pressure instability")

    if features["pump_flow_stdev_main"] <= 0.45:
        score += 4
    else:
        score -= min(10, int(features["pump_flow_stdev_main"] * 5))
        reasons.append("flow instability")

    if features["avg_pump_flow_main_ml_s"] < 0.8:
        score -= 8
        reasons.append("restricted flow")
    elif features["avg_pump_flow_main_ml_s"] > 2.8:
        score -= 5
        reasons.append("fast main flow")
    else:
        score += 3

    if features["peak_pressure_bar"] >= 10.0:
        score -= 5
        reasons.append("high peak pressure")
    elif 7.5 <= features["avg_pressure_main_bar"] <= 9.2:
        score += 2

    if features.get("stop_accuracy_g") is not None:
        acc = features["stop_accuracy_g"]
        if acc <= 1.5:
            score += 5
        elif acc <= 3.0:
            score -= 2
        else:
            score -= 6
            reasons.append("yield missed")

    if features["end_flow_slope"] > 0.18 or features["end_weight_flow_slope"] > 0.18:
        score -= 7
        reasons.append("late tail acceleration")
    elif features["end_flow_slope"] < -0.12:
        score += 2

    severity_penalty = sum(4 for e in events if e["severity"] == "warning") + sum(7 for e in events if e["severity"] == "critical")
    score -= severity_penalty
    score = max(35, min(97, score))

    tendency = "balanced"
    if features["first_drops_s"] >= 8.0 or features["avg_pump_flow_main_ml_s"] < 0.9:
        tendency = "overextracting / restrictive"
    elif features["first_drops_s"] <= 4.5 or features["avg_pump_flow_main_ml_s"] > 2.8 or (features.get("target_diff_g") or 0) > 2.0:
        tendency = "underextracting / fast"

    confidence = 0.9
    if len(events) < 3:
        confidence -= 0.08
    if features["duration_s"] < 12:
        confidence -= 0.12
    if features["weight_flow_stdev_main"] > 0.7:
        confidence -= 0.05

    return {
        "score_hint": int(round(score)),
        "confidence_hint": round(max(0.55, min(0.97, confidence)), 2),
        "tendency": tendency,
        "score_reasons": reasons[:4],
    }


def summarize_for_prompt(features, events, heuristic):
    return {
        "features": features,
        "detected_events": events,
        "heuristic": heuristic,
    }
