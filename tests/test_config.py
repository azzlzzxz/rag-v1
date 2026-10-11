from pathlib import Path

from config.settings import Settings


def test_settings_support_current_and_legacy_environment_names(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        BASE_DIR=tmp_path,
        DATA_DIR=tmp_path / "data",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        MODELS_DIR=tmp_path / "models",
        UPLOAD_DIR=tmp_path / "upload",
        VERSION="2.1.0",
        OLLAMA_MODEL="qwen-test",
        OPENAI_MODEL="openai-test",
        SILICONFLOW_MODEL="siliconflow-test",
        SILICONFLOW_API_KEY="secret-value",
    )

    assert settings.PROJECT_VERSION == "2.1.0"
    assert settings.OLLAMA_MODEL_NAME == "qwen-test"
    assert settings.OPENAI_MODEL_NAME == "openai-test"
    assert settings.SILICONFLOW_MODEL_NAME == "siliconflow-test"
    assert "secret-value" not in repr(settings)


def test_active_model_name_matches_provider():
    settings = Settings(_env_file=None, LLM_PROVIDER="siliconflow")
    assert settings.active_model_name == settings.SILICONFLOW_MODEL_NAME


def test_cors_defaults_are_local_only():
    settings = Settings(_env_file=None)
    assert settings.ALLOWED_ORIGINS
    assert all(origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1") for origin in settings.ALLOWED_ORIGINS)
