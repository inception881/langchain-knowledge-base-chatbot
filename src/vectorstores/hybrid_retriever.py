"""
Hybrid Retriever - Combining FAISS vector search with BM25 keyword search

This module provides a hybrid retrieval approach that combines the strengths of:
1. FAISS vector search (semantic similarity)
2. BM25 keyword search (lexical/keyword matching)

The hybrid approach improves retrieval quality by capturing both semantic relationships
and keyword matches, especially useful for queries that might be missed by embeddings alone.
"""
from typing import List, Optional, Dict, Any, Union, Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnableLambda

from src.config import Config
# from src.embedding import get_embeddings_singleton
from src.utils.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

def get_all_documents_from_faiss(
    vector_store: FAISS,
    filter_metadata: Optional[dict] = None
) -> List[Document]:
    """
    从 FAISS 向量库获取所有 Document，支持元数据过滤
    为每个 Document 添加 id 属性（值为 idx）
    
    Args:
        vector_store: FAISS 向量存储对象
        filter_metadata: 可选的元数据过滤条件，例如 {"source": "pdf"}
        
    Returns:
        list: Document 对象列表（每个 doc 带有 id 属性）
    """
    all_docs = []
    failed_ids = []
    
    # 遍历所有索引
    for idx, doc_id in vector_store.index_to_docstore_id.items():
        try:
            doc = vector_store.docstore.search(doc_id)
            
            # 为 doc 添加 id 属性
            doc.id = idx
            
            # 应用元数据过滤
            if filter_metadata:
                if all(
                    doc.metadata.get(k) == v 
                    for k, v in filter_metadata.items()
                ):
                    all_docs.append(doc)
            else:
                all_docs.append(doc)
                
        except Exception as e:
            failed_ids.append((doc_id, str(e)))
            continue
    
    if failed_ids:
        logger.warning(f"⚠️ 警告: {len(failed_ids)} 个文档检索失败")
        for doc_id, error in failed_ids[:5]:  # 只记录前5个错误，避免日志过长
            logger.warning(f"   - {doc_id}: {error}")
    
    return all_docs

def combine_results(results: dict) -> List[Document]:
    """
    融合向量检索和BM25检索的结果
    
    Args:
        results: 包含 "vector" 和 "bm25" 两个键的字典，值为检索结果
        
    Returns:
        融合后的文档列表
    """
    vector_res = results["vector"]
    bm25_res = results["bm25"]
    seen = set()
    combined = []
    
    # 先添加向量检索结果
    for doc in vector_res:
        # 使用内容和ID作为唯一标识
        key = (doc.page_content)
        if key not in seen:
            combined.append(doc)
            seen.add(key)
    
    # 再添加BM25检索结果
    for doc in bm25_res:
        key = (doc.page_content)
        if key not in seen:
            combined.append(doc)
            seen.add(key)
    
    # 返回前K个结果
    return combined[:Config.TOP_K*2]

def create_hybrid_retriever(vector_store: FAISS, k: int = None) -> BaseRetriever:
    """
    创建混合检索器，结合FAISS向量检索和BM25关键词检索
    
    Args:
        vector_store: FAISS向量存储
        k: 检索文档数量
        
    Returns:
        混合检索器
    """
    # 获取所有文档用于BM25
    all_docs = get_all_documents_from_faiss(vector_store)
    
    if not all_docs:
        logger.warning("⚠️ No documents found in FAISS, falling back to vector retriever only")
        search_kwargs = {"k": k or Config.TOP_K}
        return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    
    # 创建BM25检索器
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k or Config.TOP_K
    
    # 创建向量检索器
    vector_retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": k or Config.TOP_K}
    )
    
    # 创建混合检索器
    hybrid_retriever = (
        RunnableParallel(vector=vector_retriever, bm25=bm25_retriever)
        | RunnableLambda(combine_results)
    )
    
    logger.info(f"✅ Created hybrid retriever with {len(all_docs)} documents")
    return hybrid_retriever
