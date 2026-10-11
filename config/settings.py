"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 放在 Settings 类外部，避免 Pydantic 把路径常量误当成需要从环境变量解析的字段。
BASE_DIR = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Central configuration for the API and the RAG runtime."""

    # Pydantic Settings 2.x 使用 model_config；旧版内部 class Config 已不再推荐。
    # `.env` 只在本地读取，文件本身由 .gitignore 排除，避免密钥进入版本库。
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    PROJECT_NAME: str = "企业级RAG智能问答系统"
    # AliasChoices 兼容旧 `.env` 中的 VERSION，允许迁移期间不立即修改个人配置。
    PROJECT_VERSION: str = Field(
        default="1.0.0",
        validation_alias=AliasChoices("PROJECT_VERSION", "VERSION"),
    )
    BASE_DIR: Path = BASE_DIR

    DATA_DIR: Path = BASE_DIR / "data"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    MODELS_DIR: Path = BASE_DIR / "models"
    UPLOAD_DIR: Path = BASE_DIR / "upload"
    INGESTION_STORAGE_DIR: Path = DATA_DIR / "ingestion"

    VECTOR_STORE_TYPE: Literal["faiss", "chroma"] = "chroma"
    CHROMA_ANONYMIZED_TELEMETRY: bool = False
    EMBEDDING_MODEL_NAME: str = "bge_small_zh"
    EMBEDDING_DEVICE: str = "cpu"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    CHROMA_COLLECTION_NAME: str = "langchain"
    EXCEL_ROWS_PER_DOCUMENT: int = 30

    SEARCH_TOP_K: int = 5
    RETRIEVAL_CANDIDATE_K: int = 20
    RETRIEVAL_MIN_FUSION_SCORE: float = 0.0
    RETRIEVAL_MIN_RERANK_SCORE: float = 0.2
    USE_RERANKER: bool = True
    RERANKER_MODEL_NAME: str = "bge_reranker_base"

    LLM_PROVIDER: Literal["openai", "siliconflow", "ollama"] = "siliconflow"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = Field(
        default="qwen2.5:7b",
        validation_alias=AliasChoices("OLLAMA_MODEL_NAME", "OLLAMA_MODEL"),
    )

    # SecretStr 可以防止配置对象被打印或记录日志时意外暴露完整密钥。
    OPENAI_API_KEY: SecretStr = SecretStr("")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL_NAME: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("OPENAI_MODEL_NAME", "OPENAI_MODEL"),
    )

    SILICONFLOW_API_KEY: SecretStr = SecretStr("")
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
    SILICONFLOW_MODEL_NAME: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        validation_alias=AliasChoices("SILICONFLOW_MODEL_NAME", "SILICONFLOW_MODEL"),
    )

    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = BASE_DIR / "app.log"
    LOG_TO_FILE: bool = True

    # 对外只暴露当前 provider 对应的模型名，避免 API 层重复并写错 provider 分支。
    @computed_field
    @property
    def active_model_name(self) -> str:
        if self.LLM_PROVIDER == "ollama":
            return self.OLLAMA_MODEL_NAME
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_MODEL_NAME
        return self.SILICONFLOW_MODEL_NAME

    def ensure_directories(self) -> None:
        """启动前创建必要目录，使全新环境无需手工准备数据目录。"""
        for directory in (
            self.DATA_DIR,
            self.VECTOR_DB_DIR,
            self.MODELS_DIR,
            self.UPLOAD_DIR,
            self.INGESTION_STORAGE_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
