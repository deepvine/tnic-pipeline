# sbert_kiwi_encoder.py

from __future__ import annotations

from typing import List, Optional
import warnings

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi

warnings.filterwarnings("ignore")


def kiwi_sentence_split(kiwi: Kiwi, text: str, min_len: int = 2) -> List[str]:
    """
    kiwipiepy(Kiwi)를 이용한 한국어 문장 분리기

    - kiwi.split_into_sents(text) 결과에서 s.text를 추출
    - 너무 짧은 문장 제거
    """
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    try:
        # Kiwi 문장 분리
        sents = kiwi.split_into_sents(text)
        sentences = [s.text.strip() for s in sents if s.text and s.text.strip()]
    except Exception:
        # 혹시 Kiwi가 예외를 던지면, 최소한의 fallback (문장부호 기반)
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s and s.strip()]

    # 너무 짧은 문장 제거
    sentences = [s for s in sentences if len(s) >= min_len]
    return sentences


class KiwiSbertEncoder:
    """
    SBERT + Kiwi 문장 분리 기반 문서 임베딩 인코더

    - 문서 → Kiwi로 문장 분리
    - 각 문장을 SBERT로 임베딩
    - 문장 임베딩들을 평균(Sentence Mean Pooling)해서 문서 임베딩 생성
    - 텍스트 리스트를 (num_docs, dim) 배열로 변환하는 encode_texts 지원
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,   # "cuda", "cpu" 등
        batch_size: int = 32,           # SentenceTransformer 내부 batch_size
        min_sentence_len: int = 2,      # 너무 짧은 문장 필터링 기준
        num_workers: int = 0,           # SentenceTransformer encode에 전달(지원 버전에 따라 무시될 수 있음)
    ):
        """
        model_name 예시
        - 한국어 위주: "jhgan/ko-sroberta-multitask"
        - 범용: "sentence-transformers/all-mpnet-base-v2"
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.min_sentence_len = min_sentence_len
        self.num_workers = num_workers

        # Kiwi는 한 번만 생성해서 재사용
        self.kiwi = Kiwi()

        print(f"[KiwiSbertEncoder] Loading SBERT model: {model_name}")

        if device is not None:
            self.model = SentenceTransformer(model_name, device=device)
        else:
            self.model = SentenceTransformer(model_name)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[KiwiSbertEncoder] Embedding dim: {self.embedding_dim}")
        print(f"[KiwiSbertEncoder] Batch size: {self.batch_size}")

    def _sanitize_text(self, text: str) -> str:
        """
        텍스트 정제 (기본적인 제어 문자 제거 정도만 수행)
        """
        if not text:
            return ""

        # NULL, REPLACEMENT CHARACTER 제거
        text = text.replace("\x00", "").replace("\ufffd", "")

        # 간단한 공백 정리
        text = " ".join(text.split())
        return text.strip()

    def encode_document(self, text: str) -> np.ndarray:
        """
        단일 텍스트(문서)를 문서 임베딩으로 변환
        - Kiwi로 문장 분리
        - SBERT로 각 문장 임베딩
        - 문장 임베딩 평균
        """
        text = self._sanitize_text(text)
        if not text or len(text) < 2:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        sentences = kiwi_sentence_split(self.kiwi, text, min_len=self.min_sentence_len)
        if not sentences:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            # 문장 리스트를 SBERT로 임베딩
            # sentence-transformers 버전에 따라 num_workers를 지원하지 않을 수 있어 안전하게 처리
            encode_kwargs = dict(
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if self.num_workers and self.num_workers > 0:
                encode_kwargs["num_workers"] = self.num_workers

            sent_embs = self.model.encode(sentences, **encode_kwargs)

            if sent_embs.ndim == 1:
                doc_emb = sent_embs.astype(np.float32)
            else:
                doc_emb = sent_embs.mean(axis=0).astype(np.float32)

            if np.isnan(doc_emb).any() or np.isinf(doc_emb).any():
                return np.zeros(self.embedding_dim, dtype=np.float32)

            return doc_emb

        except Exception as e:
            print(f"[ERROR] encode_document failed: {str(e)[:200]}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 배열로 변환
        - 입력: List[str] (문서 리스트)
        - 출력: np.ndarray, shape = (num_texts, embedding_dim)
        """
        embeddings: List[np.ndarray] = []
        failed_count = 0

        print(f"[INFO] Encoding {len(texts)} documents with KiwiSbertEncoder...")

        for i, text in enumerate(tqdm(texts, desc="Encoding documents", unit="doc")):
            try:
                emb = self.encode_document(text)
                embeddings.append(emb)
            except Exception as e:
                print(f"[ERROR] Failed on document {i}: {str(e)[:200]}")
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                failed_count += 1

        if failed_count > 0:
            print(f"[WARN] {failed_count}/{len(texts)} documents failed to encode")

        if not embeddings:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        return np.stack(embeddings, axis=0)

    def get_embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        return self.embedding_dim


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 벡터 사이의 코사인 유사도
    """
    denom = float(norm(vec_a) * norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


# 사용 예시
if __name__ == "__main__":
    encoder = KiwiSbertEncoder(
        # 한국어 문서 위주면 ko-SBERT 계열 추천
        # model_name="jhgan/ko-sroberta-multitask",
        model_name="sentence-transformers/all-mpnet-base-v2",
        batch_size=32,
        # device="cuda",  # 필요하면 명시
    )

    docs = [
        "첫 번째 사업보고서 내용입니다. 태양광과 풍력 발전 설비에 투자했습니다.",
        "두 번째 문서입니다. 2024년에는 신재생에너지와 친환경 설비 투자를 확대했습니다.",
        "세 번째 문서입니다. 기존 화학 사업부 생산 설비 교체에 집중했습니다.",
    ]

    doc_embs = encoder.encode_texts(docs)
    print("Embeddings shape:", doc_embs.shape)

    sim_01 = cosine_similarity(doc_embs[0], doc_embs[1])
    sim_02 = cosine_similarity(doc_embs[0], doc_embs[2])

    print(f"doc0 vs doc1 similarity: {sim_01:.4f}")
    print(f"doc0 vs doc2 similarity: {sim_02:.4f}")
