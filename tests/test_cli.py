import csv
from wakeword_lab.cli import _project_init


def test_init_creates_full_metadata_template_and_config(tmp_path):
    project = tmp_path / "sample"
    _project_init(project, "Hey Nova")
    with (project / "data/raw/metadata.csv").open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert header == ["path", "speaker_id", "group_id", "split", "locale", "phrase_start_s", "phrase_end_s"]
    assert (project / "wakeword.yaml").is_file()
    assert (project / "data/raw/positive").is_dir()


def test_synthesize_negatives_cli_requires_a_phrase(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr("sys.argv", ["wakeword-lab", "synthesize-negatives", str(tmp_path)])
    from wakeword_lab.cli import main
    with pytest.raises(SystemExit):
        main()


def test_record_cli_uses_env_user_and_passes_options(tmp_path, monkeypatch):
    calls = {}
    from wakeword_lab import record

    def fake_session(project, **kwargs):
        calls["project"] = project
        calls.update(kwargs)

    monkeypatch.setattr(record, "run_recording_session", fake_session)
    monkeypatch.setattr("sys.argv", [
        "wakeword-lab", "record", str(tmp_path), "--kind", "negative",
        "--split", "validation", "--count", "3",
    ])
    monkeypatch.setenv("USER", "tester")
    from wakeword_lab.cli import main
    main()
    assert calls["kind"] == "negative" and calls["split"] == "validation"
    assert calls["count"] == 3 and calls["speaker"] == "tester"


def test_fetch_negatives_cli_rejects_unknown_source(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr("sys.argv", [
        "wakeword-lab", "fetch-negatives", str(tmp_path), "--source", "other-dataset",
    ])
    from wakeword_lab.cli import main
    with pytest.raises(SystemExit):
        main()
