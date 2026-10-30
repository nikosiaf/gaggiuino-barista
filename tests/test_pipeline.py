"""One shot stays one shot: its data, graph, JSON, history entry and notification."""
import importlib.util
import json
from pathlib import Path

import pytest

import plot_logic as pl
from shots import shot


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "OUT_DIR", tmp_path)
    monkeypatch.setattr(pl, "LAST_FILE", tmp_path / "last_shot.png")
    monkeypatch.setattr(pl, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(pl, "GEMINI_API_KEY", "")
    return tmp_path


def machine(monkeypatch, shots):
    asked = []

    def get_json(url):
        asked.append(url)
        if url.endswith("/latest"):
            return {"lastShotId": max(shots)}
        return shots[int(url.rsplit("/", 1)[1])]

    monkeypatch.setattr(pl, "get_json", get_json)
    return asked


def test_the_requested_shot_is_used_even_when_a_newer_exists(out, monkeypatch, capsys):
    asked = machine(monkeypatch, {7: shot(7), 8: shot(8, profile="Newer")})
    pl.main(7)
    assert not any(u.endswith("/latest") for u in asked) and asked[-1].endswith("/api/shots/7")
    summary = json.loads(capsys.readouterr().out.split("SUMMARY:")[1])
    assert summary["shot_id"] == 7 and summary["graph_file"] == "shot_7.png"
    assert (out / "shot_7.png").stat().st_size > 1000 and (out / "last_shot.png").exists()
    assert json.loads((out / "last_shot.json").read_text())["graph_url"] == "/local/gaggiuino-barista/shot_7.png"
    assert not list(out.glob("*.tmp*"))


def test_regenerating_replaces_the_shots_own_graph(out, monkeypatch):
    machine(monkeypatch, {7: shot(7)})
    (out / "shot_2026-01-01_10-00-00_id7.png").write_bytes(b"old")
    pl.main(7)
    pl.main(7)
    assert [p.name for p in out.glob("shot_*.png")] == ["shot_7.png"]
    assert len(json.loads((out / "shot_history.json").read_text())) == 1


def test_a_failed_render_leaves_latest_alone(out, monkeypatch):
    machine(monkeypatch, {7: shot(7)})
    (out / "last_shot.png").write_bytes(b"previous")
    monkeypatch.setattr(pl.plt.Figure, "savefig", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        pl.main(7)
    assert (out / "last_shot.png").read_bytes() == b"previous"


def load_history(base):
    path = Path(__file__).parent.parent / "addon" / "gaggiuino_barista_history.py"
    spec = importlib.util.spec_from_file_location("history", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BASE_DIR, mod.HISTORY_JSON = base, base / "shot_history.json"
    return mod


def test_history_pairs_graphs_by_shot_id_not_position(tmp_path, capsys):
    mod = load_history(tmp_path)
    (tmp_path / "shot_history.json").write_text(json.dumps([{"shot_id": 9}, {"shot_id": 8}, {"shot_id": 7}]))
    (tmp_path / "shot_9.png").write_bytes(b"x")
    (tmp_path / "shot_2026-01-01_10-00-00_id7.png").write_bytes(b"x")  # an old name, and shot 8 has none
    mod.main()
    out = json.loads(capsys.readouterr().out)
    assert [g["shot_id"] for g in out["graphs"]] == [9, 8, 7]
    assert out["graphs"][0]["url"].endswith("shot_9.png")
    assert out["graphs"][1]["url"] == ""  # the gap keeps the others in place
    assert out["graphs"][2]["url"].endswith("_id7.png")
