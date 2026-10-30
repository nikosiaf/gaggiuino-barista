"""The endpoint's protection, what the worker receives, and starting without the dashboard parts."""
import json
import subprocess

import pytest

import redact as rd
import server as sv

TOKEN = "correct-token"
KEY = "AIzaSECRET1234567890"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(sv, "API_TOKEN", TOKEN)
    monkeypatch.setattr(sv, "get_machine_status", lambda: {"profile": "p"})
    monkeypatch.setattr(sv, "get_latest_shot_id", lambda: 7)
    monkeypatch.setattr(sv, "send_notification", lambda *a: None)
    return sv.app.test_client()


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def done(stdout="", code=0, stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_get_is_refused_and_starts_nothing(client, monkeypatch):
    ran = []
    monkeypatch.setattr(sv, "_run_analysis", lambda i: ran.append(i))
    assert client.get("/plot/latest").status_code == 405
    assert client.get("/plot/latest", headers=AUTH).status_code == 405
    assert not ran


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_post_needs_the_token(client, monkeypatch, headers):
    monkeypatch.setattr(sv, "_run_analysis", lambda i: pytest.fail("must not run"))
    assert client.post("/plot/latest", headers=headers).status_code == 401


def test_without_a_configured_token_the_endpoint_is_off(client, monkeypatch):
    monkeypatch.setattr(sv, "API_TOKEN", "")
    monkeypatch.setattr(sv, "_run_analysis", lambda i: pytest.fail("must not run"))
    assert client.post("/plot/latest", headers={"Authorization": "Bearer "}).status_code == 403


def test_success_returns_a_short_result_not_process_output(client, monkeypatch):
    out = "WARNING: leak " + KEY + "\nSUMMARY:" + json.dumps({"shot_id": 7, "score": 80, "profile": "p"})
    asked = []
    monkeypatch.setattr(sv, "_run_analysis", lambda i: asked.append(i) or done(out))
    r = client.post("/plot/latest", headers=AUTH)
    assert r.get_json() == {"ok": True, "shot_id": 7, "score": 80, "profile": "p"} and asked == [7]
    assert KEY not in r.get_data(as_text=True)


def test_failure_doesnt_echo_stderr_and_logs_are_redacted(client, monkeypatch, capsys):
    monkeypatch.setattr(sv, "_run_analysis", lambda i: done(code=1, stderr=f"HTTPError for url ...?key={KEY}"))
    r = client.post("/plot/latest", headers=AUTH)
    assert r.status_code == 500 and KEY not in r.get_data(as_text=True)
    assert KEY not in capsys.readouterr().out


def test_one_analysis_at_a_time(client, monkeypatch):
    monkeypatch.setattr(sv, "_run_analysis", lambda i: pytest.fail("must not run"))
    sv._analysis_lock.acquire()
    try:
        assert client.post("/plot/latest", headers=AUTH).status_code == 409
    finally:
        sv._analysis_lock.release()


def test_watcher_path_passes_its_shot_id_including_on_retry(monkeypatch):
    calls = []
    results = [done(code=1, stderr="ConnectTimeout"), done("SUMMARY:{}")]
    monkeypatch.setattr(sv, "_run_analysis", lambda i: calls.append(i) or results.pop(0))
    monkeypatch.setattr(sv, "send_notification", lambda *a: None)
    monkeypatch.setattr(sv, "POST_SHOT_DELAY", 0)
    monkeypatch.setattr(sv.time, "sleep", lambda s: None)
    sv.run_plot(42, 30)
    assert calls == [42, 42]


def test_worker_command_and_environment(monkeypatch):
    seen = {}
    monkeypatch.setenv("SUPERVISOR_TOKEN", "super-secret")
    monkeypatch.setenv("API_TOKEN", TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
    monkeypatch.setattr(sv.subprocess, "run", lambda cmd, **kw: seen.update(cmd=cmd, env=kw["env"]) or done())
    sv._run_analysis(42)
    assert seen["cmd"][-2:] == ["--shot-id", "42"]
    assert "SUPERVISOR_TOKEN" not in seen["env"] and "API_TOKEN" not in seen["env"]
    assert seen["env"]["ANTHROPIC_API_KEY"] == "sk-ant-abc"


def test_notification_opens_this_shots_graph(monkeypatch):
    sent = {}

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {}

    monkeypatch.setattr(sv, "HA_TOKEN", "t")
    monkeypatch.setattr(sv.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(sv.requests, "post", lambda url, headers=None, json=None, timeout=None: sent.update(json) or R())
    sv.send_notification({"graph_file": "shot_7.png", "score": 80}, {"score": 80, "tuning": []})
    assert sent["data"]["image"] == "/local/gaggiuino-barista/shot_7.png"
    assert sent["data"]["url"].endswith("shot_7.png")


def test_missing_rest_sensor_doesnt_stop_the_addon(monkeypatch, tmp_path, capsys):
    class R:
        status_code = 404

    started = []
    monkeypatch.setattr(sv.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(sv, "get_machine_status", lambda: None)
    monkeypatch.setattr(sv.threading, "Thread", lambda target, daemon: type("T", (), {"start": lambda self: started.append(1)})())
    monkeypatch.setattr(sv.pathlib, "Path", lambda p: tmp_path / "www", raising=False)
    sv.start()
    assert started and "WARNING" in capsys.readouterr().out


def test_status_is_redacted(client, monkeypatch):
    monkeypatch.setitem(sv.state, "last_error", f"failed ?key={KEY}")
    assert KEY not in client.get("/status").get_data(as_text=True)


@pytest.mark.parametrize("text", [f"x?key={KEY}&a=1", "x-api-key sk-ant-abc123", "Authorization: Bearer abc.def"])
def test_redact_patterns(text):
    out = rd.redact(text)
    assert KEY not in out and "sk-ant-abc123" not in out and "abc.def" not in out and "[redacted]" in out


def test_redact_removes_configured_values(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "plain-value")
    assert "plain-value" not in rd.redact("token plain-value here")
