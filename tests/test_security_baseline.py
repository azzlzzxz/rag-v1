from pathlib import Path


def test_settings_source_contains_no_literal_secret_key():
    source = Path("config/settings.py").read_text(encoding="utf-8")
    assert "sk-" not in source


def test_env_example_contains_only_empty_key_placeholders():
    content = Path(".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=\n" in content
    assert "SILICONFLOW_API_KEY=\n" in content
