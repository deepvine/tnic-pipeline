# src/similarity.py

from __future__ import annotations

import numpy as np


def l2_normalize_rows_dense(X: np.ndarray) -> np.ndarray:
    """
    dense numpy 배열에 대해 행 단위 L2 정규화.

    각 행 x_i에 대해:
      x_i <- x_i / ||x_i||
    """
    # 제곱합: (n_rows, 1)
    squared_sum = np.sum(X ** 2, axis=1, keepdims=True)
    norms = np.sqrt(squared_sum)
    norms[norms == 0] = 1.0  # 0 나눗셈 방지

    X_norm = X / norms
    return X_norm


def compute_cosine_similarity_matrix_dense(X_norm: np.ndarray) -> np.ndarray:
    """
    단위벡터로 정규화된 행렬 X_norm에 대해
    코사인 유사도 행렬 S = X_norm @ X_norm.T 계산.

    X_norm: (num_firms, hidden_size)
    반환:
      S: (num_firms, num_firms)
    """
    S = X_norm @ X_norm.T
    # 수치 에러 방지를 위해 대각선은 1로 보정
    np.fill_diagonal(S, 1.0)
    return S


def median_center_similarity(S: np.ndarray) -> np.ndarray:
    """
    median centering:

    기업 i에 대해:
      med_i = median_j (j != i) S_ij
      S'_ij = S_ij - med_i
    """
    N = S.shape[0]
    S_centered = S.copy()

    for i in range(N):
        row = S[i, :]
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        med_i = np.median(row[mask])
        S_centered[i, :] = row - med_i

    return S_centered
