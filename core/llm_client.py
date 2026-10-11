"""
大模型客户端 —— 统一封装 ollama OpenAI SiliconFlow 等大模型客户端
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """大模型客户端"""
    def __init__(self):
        self._llm = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化大模型客户端"""
        if settings.LLM_PROVIDER == 'ollama':
            self._llm = ChatOllama(model=settings.OLLAMA_MODEL_NAME, base_url=settings.OLLAMA_BASE_URL, temperature=0.1,verbose=True)
            logger.info(f"OLLAMA模型初始化完成，模型: {settings.OLLAMA_MODEL_NAME}")
        elif settings.LLM_PROVIDER == 'openai':
            # SecretStr 仅在创建 SDK 客户端的边界解包，其他业务层不会接触明文密钥。
            self._llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME, api_key=settings.OPENAI_API_KEY.get_secret_value(), base_url=settings.OPENAI_BASE_URL, temperature=0.1)
            logger.info(f"OpenAI模型初始化完成，模型: {settings.OPENAI_MODEL_NAME}")
        elif settings.LLM_PROVIDER == 'siliconflow':
            # SiliconFlow 提供 OpenAI 兼容协议，因此继续复用 ChatOpenAI 客户端。
            self._llm = ChatOpenAI(model=settings.SILICONFLOW_MODEL_NAME, api_key=settings.SILICONFLOW_API_KEY.get_secret_value(), base_url=settings.SILICONFLOW_BASE_URL, temperature=0.1)
            logger.info(f"SiliconFlow模型初始化完成，模型: {settings.SILICONFLOW_MODEL_NAME}")
        else:
            raise ValueError(f"不支持的大模型提供商: {settings.LLM_PROVIDER}")
        
    def get_llm(self):
        """获取大模型客户端"""
        return self._llm

    def change_provider(self, provider: str, **kwargs):
        """切换大模型提供商"""
        settings.LLM_PROVIDER = provider
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self._initialize_client()
        logger.info(f"大模型提供商已切换为: {provider}")
        
_llm_client = None

def get_llm():
    """获取大模型客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client.get_llm()
