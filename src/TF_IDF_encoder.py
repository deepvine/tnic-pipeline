# src/tfidf_encoder.py

from __future__ import annotations

from typing import List, Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class TfidfEncoder:
    """
    한국어 TF-IDF 기반 문서 임베딩 생성기.

    - fit()으로 코퍼스를 한 번 학습한 후
    - encode_texts()로 개별 텍스트 리스트를 벡터화.

    반환:
      - shape: (len(texts), vocab_size) 의 numpy 배열 (dense) 또는 sparse matrix
    """

    def __init__(
        self,
        vectorizer: Optional[TfidfVectorizer] = None,
        max_features: int = 30000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.9,
        use_idf: bool = True,
        norm: str = "l2",
    ) -> None:
        # 이미 학습된 vectorizer를 넘겨줄 수도 있고,
        # 아니면 여기서 새로 만들 수도 있음.
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                use_idf=use_idf,
                norm=norm,
            )
            self._fitted = False
        else:
            self.vectorizer = vectorizer
            self._fitted = True

    def fit(self, texts: Iterable[str]) -> "TfidfEncoder":
        """
        전체 코퍼스를 넣고 TF-IDF vocabulary 학습.
        (BERT의 pretrain에 해당하는 단계라고 보면 됨.)
        """
        clean_texts = [
            t if isinstance(t, str) and t.strip() else ""
            for t in texts
        ]
        self.vectorizer.fit(clean_texts)
        self._fitted = True
        return self

    def encode_texts(
        self,
        texts: List[str],
        dense: bool = True,
    ):
        """
        입력 텍스트 리스트를 TF-IDF 벡터로 변환.

        dense=True  -> numpy.ndarray 반환 (BERT랑 비슷하게 사용)
        dense=False -> scipy sparse matrix 반환
        """
        if not self._fitted:
            raise RuntimeError(
                "TfidfEncoder가 아직 fit되지 않았습니다. "
                "먼저 encoder.fit(corpus) 를 호출하거나, "
                "미리 학습된 vectorizer를 넘겨주세요."
            )

        clean_texts = [
            t if isinstance(t, str) and t.strip() else ""
            for t in texts
        ]
        X = self.vectorizer.transform(clean_texts)  # sparse matrix

        if dense:
            return X.toarray().astype(np.float32)
        return X

    # --------- 저장 / 로드 유틸 ---------

    def save(self, path: str) -> None:
        """
        학습된 vectorizer를 joblib으로 저장.
        """
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "TfidfEncoder":
        """
        저장된 vectorizer를 불러와서 TfidfEncoder 인스턴스 생성.
        """
        vec = joblib.load(path)
        return cls(vectorizer=vec)
