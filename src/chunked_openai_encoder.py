from __future__ import annotations

from typing import List, Literal
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import tiktoken
import time


class ChunkedOpenAIEncoder:
    """
    청크 기반 OpenAI 임베딩 with 배치 처리 최적화
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 50,  # API 배치 크기
        max_retries: int = 3,
        max_tokens: int = 8000,
        chunk_overlap: int = 100,
        aggregation: Literal["mean", "sum", "max", "weighted"] = "mean",
    ):
        """
        Args:
            batch_size: API 요청당 청크 개수 (최대 2048)
                - 추천: 100 (안정적)
                - 최대: 2048 (빠르지만 rate limit 위험)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.aggregation = aggregation
        
        # OpenAI 클라이언트
        self.client = OpenAI(api_key=api_key)
        
        # Tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # 임베딩 차원
        self.embedding_dim = self._get_embedding_dim(model_name)
        
        print(f"[ChunkedOpenAIEncoder] Model: {model_name}")
        print(f"[ChunkedOpenAIEncoder] Embedding dim: {self.embedding_dim}")
        print(f"[ChunkedOpenAIEncoder] Max tokens per chunk: {max_tokens}")
        print(f"[ChunkedOpenAIEncoder] Chunk overlap: {chunk_overlap}")
        print(f"[ChunkedOpenAIEncoder] Aggregation: {aggregation}")
        print(f"[ChunkedOpenAIEncoder] Batch size: {batch_size}")
    
    def _get_embedding_dim(self, model_name: str) -> int:
        """모델별 임베딩 차원"""
        if "large" in model_name:
            return 3072
        else:
            return 1536
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text) // 4
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """텍스트를 토큰 기준 청크로 분할"""
        if not text or not text.strip():
            return [""]
        
        try:
            tokens = self.tokenizer.encode(text)
            
            if len(tokens) <= self.max_tokens:
                return [text]
            
            chunks = []
            stride = self.max_tokens - self.chunk_overlap
            
            for i in range(0, len(tokens), stride):
                chunk_tokens = tokens[i:i + self.max_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                if i + self.max_tokens >= len(tokens):
                    break
            
            return chunks if chunks else [text]
            
        except Exception as e:
            print(f"[WARN] Chunking failed: {e}")
            max_chars = self.max_tokens * 4
            chunks = []
            for i in range(0, len(text), max_chars):
                chunk = text[i:i + max_chars]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks if chunks else [text]
    
    def _encode_batch(self, texts: List[str], retry: int = 0) -> List[np.ndarray]:
        """
        배치 텍스트를 한 번에 임베딩
        핵심 최적화: 여러 청크를 한 번에 API 호출!
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,  # 리스트로 한 번에!
                encoding_format="float"
            )
            
            # 입력 순서대로 임베딩 반환
            embeddings = [
                np.array(item.embedding, dtype=np.float32) 
                for item in response.data
            ]
            
            return embeddings
            
        except Exception as e:
            error_msg = str(e)
            
            # Rate limit 처리
            if "rate_limit" in error_msg.lower():
                if retry < self.max_retries:
                    wait_time = 2 ** retry
                    print(f"[WARN] Rate limit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    return self._encode_batch(texts, retry + 1)
                else:
                    print(f"[ERROR] Rate limit exceeded")
                    return [np.zeros(self.embedding_dim, dtype=np.float32)] * len(texts)
            
            # 기타 에러
            print(f"[ERROR] Batch encoding failed: {error_msg[:100]}")
            return [np.zeros(self.embedding_dim, dtype=np.float32)] * len(texts)
    
    def _aggregate_chunk_embeddings(
        self, 
        chunk_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """청크 임베딩 집계"""
        if len(chunk_embeddings) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        if len(chunk_embeddings) == 1:
            return chunk_embeddings[0]
        
        stacked = np.stack(chunk_embeddings)
        
        if self.aggregation == "mean":
            return np.mean(stacked, axis=0)
        elif self.aggregation == "sum":
            return np.sum(stacked, axis=0)
        elif self.aggregation == "max":
            return np.max(stacked, axis=0)
        elif self.aggregation == "weighted":
            num_chunks = len(chunk_embeddings)
            weights = self._compute_chunk_weights(num_chunks)
            return np.sum(stacked * weights[:, np.newaxis], axis=0)
        else:
            return np.mean(stacked, axis=0)
    
    def _compute_chunk_weights(self, num_chunks: int) -> np.ndarray:
        """가우시안 가중치"""
        if num_chunks == 1:
            return np.array([1.0])
        
        positions = np.arange(num_chunks)
        center = (num_chunks - 1) / 2
        sigma = num_chunks / 4
        weights = np.exp(-((positions - center) ** 2) / (2 * sigma ** 2))
        return weights / np.sum(weights)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환 (배치 최적화)
        
        핵심 최적화:
        1. 모든 텍스트를 청크로 분할
        2. 모든 청크를 배치로 한 번에 임베딩
        3. 각 문서별로 청크 임베딩을 집계
        """
        if not texts:
            return np.array([])
        
        print(f"[INFO] Preparing chunks for {len(texts)} texts...")
        
        # 1단계: 모든 텍스트를 청크로 분할
        all_chunks = []  # 모든 청크 (평탄화)
        doc_chunk_map = []  # 각 문서의 청크 인덱스 범위
        
        chunk_start = 0
        for text in tqdm(texts, desc="Splitting into chunks", leave=False):
            if not text or not text.strip():
                chunks = [""]
            else:
                chunks = self._split_into_chunks(text)
            
            chunk_count = len(chunks)
            all_chunks.extend(chunks)
            doc_chunk_map.append((chunk_start, chunk_start + chunk_count))
            chunk_start += chunk_count
        
        print(f"[INFO] Total chunks: {len(all_chunks)} (avg {len(all_chunks)/len(texts):.1f} per text)")
        
        # 2단계: 모든 청크를 배치로 임베딩
        print(f"[INFO] Encoding chunks in batches of {self.batch_size}...")
        
        all_chunk_embeddings = []
        num_batches = (len(all_chunks) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(all_chunks), self.batch_size), 
                     desc="Encoding chunks", 
                     total=num_batches):
            batch_chunks = all_chunks[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_chunks)
            all_chunk_embeddings.extend(batch_embeddings)
            
            # Rate limit 방지
            if i % (self.batch_size * 10) == 0:
                time.sleep(0.1)
        
        # 3단계: 각 문서별로 청크 임베딩 집계
        print(f"[INFO] Aggregating chunks per document...")
        
        doc_embeddings = []
        for start_idx, end_idx in tqdm(doc_chunk_map, desc="Aggregating", leave=False):
            chunk_embs = all_chunk_embeddings[start_idx:end_idx]
            
            # zero 벡터가 아닌 것만 필터링
            valid_embs = [emb for emb in chunk_embs if np.any(emb)]
            
            if valid_embs:
                doc_emb = self._aggregate_chunk_embeddings(valid_embs)
            else:
                doc_emb = np.zeros(self.embedding_dim, dtype=np.float32)
            
            doc_embeddings.append(doc_emb)
        
        return np.stack(doc_embeddings)
    
    def get_embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        return self.embedding_dim


# 하위 호환성
# OpenAIEncoder = ChunkedOpenAIEncoder