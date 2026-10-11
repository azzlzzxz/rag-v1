"""
多轮对话记忆管理模块——支持多会话隔离
"""

import uuid
import logging
from datetime import datetime, timedelta
# 迁移过渡：LangChain 1.x 已将旧 Memory API 移至 langchain-classic。
# 阶段 4 会改用 LangGraph checkpoint 后删除此依赖。
from langchain_classic.memory import ConversationBufferWindowMemory

logger = logging.getLogger(__name__)

class ConversationSession:
  """
  单个对话会话的记忆管理
  """
  def __init__(self, session_id:str, window_size: int = 5):
    self.session_id = session_id
    self.created_at = datetime.now()
    self.last_active = datetime.now()
    self.memory = ConversationBufferWindowMemory(
        k=window_size, # 保留最近5轮对话
        memory_key="chat_history",  # 记忆键名
        return_messages=True,  # 返回消息对象而不是字符串
        output_key="answer"  # 输出键名
    )
    self.metadata: dict = {}

  def add_user_message(self, message: str):
    """添加用户消息到记忆"""
    self.memory.chat_memory.add_user_message(message)
    self.last_active = datetime.now()

  def add_ai_message(self, message: str):
    """添加AI消息到记忆"""
    self.memory.chat_memory.add_ai_message(message)
    self.last_active = datetime.now()

  def get_memory(self) -> list:
    """获取对话历史"""
    return self.memory.chat_memory.messages

  def clear(self):
    """清空当前会话的记忆"""
    self.memory.chat_memory.clear()
    self.last_active = datetime.now()

  def to_dict(self) -> dict:
    """序列化会话数据"""
    messages = []
    for message in self.memory.chat_memory.messages:
      messages.append({
        "type": message.type,
        "content": message.content
      })
    return {
        "session_id": self.session_id,
        "created_at": self.created_at.isoformat(),
        "last_active": self.last_active.isoformat(),
        "messages": messages,
        "metadata": self.metadata
    }


class MemoryManager:
  """
  多回话记忆管理
  """
  def __init__(self, window_size: int = 5, session_ttl_minutes: int = 60):
    self.window_size = window_size
    self.session_ttl = timedelta(minutes=session_ttl_minutes) # 会话过期时间
    self.sessions: dict[str, ConversationSession] = {} # 存储所有会话
    logger.info("记忆管理器初始化完成，window_size=%d, session_ttl=%s", window_size, session_ttl_minutes)
    
  def create_session(self, session_id: str = None) -> str:
    """创建新会话"""
    if session_id is None:
      session_id = str(uuid.uuid4())
    
    # 检查是否已存在该会话
    if session_id in self.sessions:
      logger.warning("会话已存在: %s", session_id)
      return session_id
    
    self.sessions[session_id] = ConversationSession(session_id, self.window_size)
    logger.info("创建新会话: %s", session_id)
    return session_id

  def get_session(self, session_id: str) -> ConversationSession:
    """获取会话"""
    if session_id not in self.sessions:
      logger.warning("会话不存在: %s", session_id)
      return None
    
    # 检查会话是否过期
    if datetime.now() - self.sessions[session_id].last_active > self.session_ttl:
      logger.info("会话已过期: %s", session_id)
      del self.sessions[session_id]
      return None
    
    self.sessions[session_id].last_active = datetime.now()
    return self.sessions[session_id]

  def get_or_create_session(self, session_id: str) -> ConversationSession:
    """获取或创建会话"""
    if session_id and session_id in self.sessions:
      return self.get_session(session_id)
    new_session_id = self.create_session(session_id)
    return self.get_session(new_session_id)

  def add_exchange(self, session_id: str, query: str, answer: str):
    """添加对话交换"""
    session = self.get_or_create_session(session_id)
    # session.add_exchange({"query": query, "answer": answer})
    session.add_user_message(query)
    session.add_ai_message(answer)
    logger.info(f"添加对话记录：会话 {session_id}，用户问题: {query}, AI回答: {answer}")

  def get_chat_history(self, session_id: str) -> list:
    """获取聊天历史"""
    session = self.get_session(session_id)
    if session is None:
      return []
    return session.get_memory()

  def clear_session(self, session_id: str):
    """清空会话"""
    if session_id in self.sessions:
      self.sessions[session_id].clear()
      logger.info(f"清空会话: {session_id}")

  def delete_session(self, session_id: str):
    """删除会话"""
    if session_id in self.sessions:
      del self.sessions[session_id]
      logger.info(f"删除会话: {session_id}")

  def cleanup_old_sessions(self):
    """清理过期会话"""
    current_time = datetime.now()
    expired_sessions = [
      session_id for session_id, session in self.sessions.items()
      if current_time - session.last_active > self.session_ttl
    ]
    for session_id in expired_sessions:
      del self.sessions[session_id]
      logger.info(f"清理过期会话: {session_id}")
    if expired_sessions:
      logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
    return len(expired_sessions)

  def list_sessions(self) -> list:
    """列出所有会话"""
    return [session.to_dict() for session in self.sessions.values()]

  def get_default_session(self) -> ConversationSession:
    """获取默认会话"""
    return self.get_or_create_session("default").memory

_memory_manager_instance = None

def get_memory_manager() -> MemoryManager:
    """获取内存管理器单例"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance
