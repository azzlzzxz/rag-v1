"""
文档加载器模块——支持 pdf word excel txt markdown 等多种格式的文档加载
步骤：
1. 加载文档：使用合适的加载器，可以选择是否添加元数据。 
    什么是元数据？——就是描述数据的数据，例如：文件名、文件类型、作者等等等
    添加元数据，优点就是能更好的管理数据。
2. 分割文档：将文档分割成多个片段，以便于后续的处理。
"""
from pathlib import Path
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentLoader:
    """
    文档加载器类
    """

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".markdown": UnstructuredMarkdownLoader,
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len,
        )

    
    def load_file(self, file_path: str):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.LOADER_MAP:
            raise ValueError(f"不支持的文件格式: {suffix}")

        try:
            loader_class = self.LOADER_MAP[suffix]
            if suffix == '.txt':
                loader = loader_class(str(file_path), encoding='utf-8')
            else:
                loader = loader_class(str(file_path))
            documents = loader.load()
            logger.info(f"加载文档: {file_path}, 共 {len(documents)} 个文档块")
            # 添加元数据
            for doc in documents:
                doc.metadata["file_path"] = str(file_path)
                doc.metadata["file_name"] = file_path.name
                doc.metadata["file_type"] = suffix[1:]

            # 分割文档
            documents = self.text_splitter.split_documents(documents)
            logger.info(f"分割文档: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            raise e
            return []
        
        
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> list:
        """
        加载目录中的所有文件
        :param directory_path: 目录路径
        :param recursive: 是否递归加载子目录
        :return: 文档列表
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        documents = []
        pattern = "**/*" if recursive else "*"  # 递归匹配所有文件
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.LOADER_MAP:
                documents.extend(self.load_file(str(file_path)))
        logger.info(f"加载目录: {directory_path}, 共 {len(documents)} 个文档块")
        return documents
    