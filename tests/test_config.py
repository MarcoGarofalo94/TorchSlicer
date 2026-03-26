from torchslicer.config import RunConfig


def test_run_config_load_applies_worker_tag_filter_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "discovery:\n"
        "  tag_filter:\n"
        "    - yaml-tag\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKER_TAG_FILTER", "env-a, env-b")

    cfg = RunConfig.load(str(config_path))

    assert cfg.discovery.tag_filter == ["env-a", "env-b"]
