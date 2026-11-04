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
