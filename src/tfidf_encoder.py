# src/tfidf_encoder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfEncoderConfig:
    """
    TF-IDF 인코더 설정값을 담는 데이터 클래스.
    """

    min_df: int = 2          # 너무 희귀한 토큰 제거
    max_df: float = 0.9      # 너무 자주 등장하는 토큰 제거 (문서 비율)
    # 필요시 ngram_range, max_features 등 옵션을 추가할 수 있음.


class TfidfEncoder:
    """
    간단한 TF-IDF 인코더 래퍼 클래스.

    - texts: 토큰들이 공백으로 join 된 문자열 리스트
      예) "탄산 가스 사업 CO2 액화" 이런 형식

    사용 예:
        encoder = TfidfEncoder(min_df=2, max_df=0.9)
        embeddings = encoder.encode_texts(clean_texts)  # (num_docs, dim) ndarray
    """

    def __init__(self, min_df: int = 2, max_df: float = 0.9) -> None:
        self.config = TfidfEncoderConfig(min_df=min_df, max_df=max_df)

        # tokenizer는 따로 쓰지 않고, build_year_filtered_texts 결과(공백-separated)를 그대로 사용
        self.vectorizer = TfidfVectorizer(
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            # 이미 토큰이 공백 기준으로 잘라진 상태라면 analyzer="word" 기본값으로 충분
            # token_pattern 기본값 사용
        )

        self._fitted: bool = False

    # --------------------------------------------------
    # 기본 fit / transform API
    # --------------------------------------------------
    def fit(self, texts: List[str]) -> "TfidfEncoder":
        """
        전체 텍스트 리스트에 대해 TF-IDF vocabulary를 학습한다.
        """
        if not texts:
            # 빈 리스트 들어온 경우는 그냥 넘어감
            self._fitted = False
            return self

        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        이미 fit 된 vocabulary를 사용하여 TF-IDF 벡터로 변환한다.
        """
        if not texts:
            # 문서가 없는 경우 (0, 0) shape 반환
            return np.zeros((0, 0), dtype=np.float32)

        if not self._fitted:
            raise RuntimeError(
                "TfidfEncoder is not fitted yet. Call `fit(texts)` "
                "or use `encode_texts(texts)` which does fit+transform 한 번에 수행합니다."
            )

        X = self.vectorizer.transform(texts)  # scipy CSR matrix
        return X.astype(np.float32).toarray()

    # --------------------------------------------------
    # 편의용 one-shot API (연도별로 fit+transform)
    # --------------------------------------------------
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        주어진 texts 리스트에 대해 fit + transform을 한 번에 수행한다.

        main_tfidf.py 에서는 연도별(year)로 호출되므로,
        각 연도마다 독립적인 TF-IDF 스케일을 쓰게 된다.
        (전 연도 공통 vocab이 필요하면, process_all_years 단계에서
         한 번만 fit()을 호출하고, 연도별로 transform()만 호출하도록
         로직을 변경하면 된다.)
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        X = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return X.astype(np.float32).toarray()
