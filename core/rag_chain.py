"""
RAG问答链模块——使用LCEL语法实现
整合 检索、历史感知 意图识别 LLM生成等模块，支持多轮对话和来源追溯
"""
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from core.llm_client import get_llm
from core.retriever import get_rag_retriever
from core.intent_recognizer import IntentRecognizer
from core.memory_manager import get_memory_manager
from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "default"

class RAGChain:
    """RAG问答链"""
    def __init__(self):
        self.llm = get_llm()
        self.retriever = get_rag_retriever()
        self.intent_recognizer = IntentRecognizer()
        self.memory_manager = get_memory_manager()
        self._rag_chain = self._create_rag_chain()
        logger.info("RAGChain initialized")
    
    def _create_rag_chain(self):
        """创建RAG链"""
        # 1. 改写用户提示词
        # 用户第一次问：推荐几款牙膏？   第二次：那个便宜？
        # 改写后：推荐几款牙膏？  第二次：上一次推荐的牙膏哪个便宜？
        contextualize_q_system_prompt = """你是一个专业的query改写专家。你的任务是根据历史对话，将用户的追问改写成一个独立、完整的query。
            如果用户的提问已经足够独立，请直接返回原问题，不要添加任何额外内容。
        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 2. 创建历史感知的检索器
        """create_history_aware_retriever 做的事情
        聊天历史
          +
        当前问题
          ↓
        LLM
          ↓
        改写后的完整问题
          ↓
        Retriever
          ↓
        检索相关文档

        Args:
            llm: 大语言模型: 理解历史 + 当前问题，然后改写查询
            retriever: 检索器: 根据改写后的查询检索相关文档
            prompt: 提示词模板: 定义如何改写查询
        """
        history_aware_retriever = create_history_aware_retriever(
            self.llm, 
            self.retriever.get_compresstion_retriever(search_kwargs={"k": settings.SEARCH_TOP_K}), 
            contextualize_q_prompt
        )

        logger.info("RAG 历史感知检索器创建成功")

        qa_system_prompt = """你是一个专业的客服助手。请根据提供的上下文信息回答用户问题。

        回答要求：
        1. 仅基于提供的上下文信息回答，不要编造额外信息
        2. 如果上下文信息不足，请诚实地说明
        3. 回答要简洁、准确、友好
        4. 如有相关产品或服务信息，请提供详细说明

        上下文信息：
        {context}

        用户问题：{input}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        logger.info("RAG 问答链创建成功")
        
        # 3. 创建问题回答链
        """create_stuff_documents_chain 做的事情: 拿着检索出来的 Documents，让 LLM 根据这些 Documents 回答问题。
        改写后的完整问题
          +
        检索到的文档
          ↓
        LLM
          ↓
        最终回答
        
        Args:
            llm: 大语言模型: 理解改写后的查询和检索到的文档，然后生成最终回答
            prompt: 提示词模板: 定义如何生成最终回答
        """
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        logger.info("RAG 问题回答链创建成功")

        # 4. 创建最终的RAG链
        """
        历史感知检索器
          +
        问题回答链
          ↓
        最终回答
        """
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        logger.info("RAG 最终链创建成功")
        
        return rag_chain

    def ask(self, question: str, session_id: str = DEFAULT_SESSION_ID) -> dict:
        """
        处理用户问题
        :param question: 用户问题
        :param session_id: 会话ID
        :return: 回答结果
        """
        if not question.strip():
            return {"answer": "问题不能为空", "sources": [], "intent": "invalid"}

        # 意图识别
        intent = self.intent_recognizer.recognize(question)
        logger.info(f"意图识别结果: {intent}")

        # 获取对话历史
        try:
            chat_history = self.memory_manager.get_chat_history(session_id)
            chat_input = {"input": question, "chat_history": chat_history}
            result = self._rag_chain.invoke(chat_input)
            answer = result["answer"]
            sources_doc = result.get("context", [])
            sources = []
            for doc in sources_doc:
                sources.append({
                    "source": doc.metadata.get("file_name", doc.metadata.get("source", "未知")),
                    "page": doc.metadata.get("page", 1),
                    "content": doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })
            # 更新对话历史
            self.memory_manager.add_exchange(session_id, question, answer)
            return {"answer": answer, "sources": sources, "intent": intent}
        except Exception as e:
            logger.error(f"处理问题时出错: {e}")
            return {"answer": "抱歉，处理您的问题时出错了", "sources": [], "intent": "error"}
        
        
    def get_chat_history(self, session_id: str = DEFAULT_SESSION_ID) -> list:
        """
        获取对话历史
        :param session_id: 会话ID
        :return: 对话历史
        """
        messages = self.memory_manager.get_chat_history(session_id)
        return [
            {"role": "human", "content": msg.content} if isinstance(msg, HumanMessage) else
            {"role": "assistant", "content": msg.content}
            for msg in messages
        ]

    def clear_session(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        """
        清空会话
        :param session_id: 会话ID
        """
        self.memory_manager.clear_session(session_id)
        logger.info(f"会话 {session_id} 已清空")


_rag_chain_instance = None

def get_rag_chain() -> RAGChain:
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = RAGChain()
    return _rag_chain_instance