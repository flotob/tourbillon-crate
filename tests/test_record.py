from pathlib import Path

from crate.record import describe, record_name


def test_record_name():
    assert record_name("Nata Alma (Club Smash Hit Version)") == "nataalma"
    assert record_name("Strings of Life [Transmat]") == "stringsoflife"
    assert record_name("A Very Long Title That Goes On") == "averylongtitleth"
    assert record_name("808 Bass") == "r808bass"
    assert record_name("(Intro)") == "intro"
    assert record_name("???") == "record"


def test_describe_from_file_name(monkeypatch):
    monkeypatch.setattr("crate.record.read_tags", lambda path: {})
    assert describe(Path("01 Nata Alma (Club Smash Hit Version).m4a")) == ("Nata Alma (Club Smash Hit Version)", None)
    assert describe(Path("Rhythim Is Rhythim - Strings of Life.mp3")) == ("Strings of Life", "Rhythim Is Rhythim")
    assert describe(Path("808.wav")) == ("808", None)


def test_describe_prefers_tags(monkeypatch):
    monkeypatch.setattr("crate.record.read_tags", lambda path: {"title": "Nata Alma", "artist": "Someone"})
    assert describe(Path("01 track.mp3")) == ("Nata Alma", "Someone")
