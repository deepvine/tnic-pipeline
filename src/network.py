# src/network.py

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def build_tnic_edges(
    S: np.ndarray,
    firm_ids: List[str],
    year: int,
    threshold: float = 0.2132,
) -> List[Dict[str, Any]]:
    """
    TNIC 방식으로 네트워크 엣지 생성.

    S: 코사인 유사도 행렬 (num_firms x num_firms)
    firm_ids: 행 인덱스에 해당하는 firm_id 리스트
    threshold: 코사인 유사도 임계값 (0~1)
    """
    N = S.shape[0]
    edges: List[Dict[str, Any]] = []

    for i in range(N):
        for j in range(i + 1, N):
            sim = float(S[i, j])
            if sim > threshold:
                edges.append(
                    {
                        "year": year,
                        "firm_i": firm_ids[i],
                        "firm_j": firm_ids[j],
                        "similarity": sim,
                    }
                )
    return edges
