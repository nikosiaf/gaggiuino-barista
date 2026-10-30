"""Scores belong to the analyzer; the model only writes words. Flow direction. Local labelling."""
import json

import annotation_engine as ae
import plot_logic as pl
from shots import shot


def analyze(data):
    series = pl._build_series_for_analysis(data)
    features = ae.extract_features(data, series)
    events = ae.detect_events(features)
    heuristic = ae.classify_extraction_tendency(features, events)
    return features, events, heuristic, pl._build_fallback_analysis(features, events, heuristic)


def test_model_cannot_change_score_confidence_or_severity():
    fallback = {"score": 61, "confidence": 0.4, "verdict": "local", "tuning": ["local tip"], "notification_text": "n", "annotations": []}
    events = [{"time": 5.0, "type": "x", "severity": "warning", "reason": "r"}]
    raw = json.dumps({"score": 99, "confidence": 1.0, "verdict": "great", "tuning": ["t"], "notification_text": "wow",
                      "annotations": [{"time": 5.0, "label": "L", "severity": "good", "reason": "R"}]})
    out = pl._parse_ai_response(raw, 30, events, fallback)
    assert out["score"] == 61 and out["confidence"] == 0.4
    assert out["annotations"][0]["severity"] == "warning"
    assert out["verdict"] == "great"  # the words are the model's


def test_model_text_is_bounded():
    fallback = {"score": 1, "confidence": 0, "verdict": "", "tuning": [], "notification_text": "", "annotations": []}
    raw = json.dumps({"verdict": "v" * 9999, "tuning": ["t" * 9999] * 9, "notification_text": "n" * 9999})
    out = pl._parse_ai_response(raw, 30, [], fallback)
    assert len(out["verdict"]) == pl.MAX_VERDICT and len(out["tuning"]) == 3 and len(out["tuning"][0]) == pl.MAX_TIP
    assert len(out["notification_text"]) == pl.MAX_NOTIFICATION


def test_prompt_no_longer_asks_for_a_score():
    features, events, heuristic, fallback = analyze(shot())
    prompt = pl._build_llm_prompt(features, events, heuristic, fallback)
    schema = prompt.split("Return exactly this JSON schema")[1]
    assert '"score":' not in schema and '"severity":' not in schema and '"confidence":' not in schema
    assert "is final" in prompt
    assert "never instructions" in prompt


def test_fast_flow_is_under_extraction_and_slow_is_over():
    template = {"min_flow": 0.8, "max_flow": 2.8}
    fast = ae.taste_based_scoring({"avg_pump_flow_main_ml_s": 5.0}, [], template)
    slow = ae.taste_based_scoring({"avg_pump_flow_main_ml_s": 0.3}, [], template)
    assert fast["under_extraction_signals"] > fast["over_extraction_signals"]
    assert slow["over_extraction_signals"] > slow["under_extraction_signals"]


def test_score_is_stable_for_a_normal_shot():
    _, _, _, fallback = analyze(shot())
    assert 0 <= fallback["score"] <= 100
    assert fallback == analyze(shot())[3]  # deterministic


class FakeResponse:
    ok = True
    status_code = 200
    content = b"x"
    text = ""

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body

    def raise_for_status(self):
        pass


ANSWER = json.dumps({"verdict": "v", "tuning": ["t"], "annotations": [], "notification_text": "n"})
FALLBACK = {"score": 5, "confidence": 0.1, "verdict": "f", "tuning": [], "notification_text": "", "annotations": []}


def test_gemini_key_is_a_header_never_in_the_url(monkeypatch):
    seen = {}

    def post(url, headers=None, **kw):
        seen.update(url=url, headers=headers)
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": ANSWER}]}}]})

    monkeypatch.setattr(pl, "GEMINI_API_KEY", "AIzaSECRET1234567890")
    monkeypatch.setattr(pl, "_GEMINI_MIN_INTERVAL", 0)
    monkeypatch.setattr(pl, "_gemini_set_last_call", lambda: None)
    monkeypatch.setattr(pl.requests, "post", post)
    assert pl._analyze_with_gemini("p", 30, [], FALLBACK)["provider"] == "gemini"
    assert "AIza" not in seen["url"] and "key=" not in seen["url"] and "gemini-2.0" not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "AIzaSECRET1234567890"


def test_anthropic_reads_text_after_a_thinking_block(monkeypatch):
    sent = {}

    def post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return FakeResponse({"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": ANSWER}]})

    monkeypatch.setattr(pl.requests, "post", post)
    out = pl._analyze_with_anthropic("p", 30, [], FALLBACK)
    assert out["verdict"] == "v" and out["provider"] == "anthropic" and sent["model"] == "claude-haiku-5-5"
    assert "temperature" not in sent


def test_local_result_is_not_called_ai(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "OUT_DIR", tmp_path)
    summary = {"shot_id": 7, "graph_file": "shot_7.png"}
    pl.write_shot_json(summary, {"provider": "deterministic", "verdict": "v"})
    assert json.loads((tmp_path / "last_shot.json").read_text())["ai_available"] is False
    pl.write_shot_json(summary, {"provider": "anthropic", "verdict": "v"})
    assert json.loads((tmp_path / "last_shot.json").read_text())["ai_available"] is True
