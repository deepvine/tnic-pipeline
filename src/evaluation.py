# src/evaluation.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _percentiles(x: np.ndarray, ps: List[float]) -> Dict[str, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x, ps).tolist()
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _tri_upper_values(mat: np.ndarray, k: int = 1) -> np.ndarray:
    n = mat.shape[0]
    if n <= 1:
        return np.array([], dtype=np.float32)
    iu = np.triu_indices(n, k=k)
    return mat[iu]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def component_sizes(self) -> List[int]:
        roots: Dict[int, int] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            roots[r] = roots.get(r, 0) + 1
        return sorted(roots.values(), reverse=True)


def evaluate_year(
    *,
    year: int,
    S: np.ndarray,
    S_centered: np.ndarray,
    edges: List[Dict[str, Any]],
    firm_ids: List[str],
    firm_names: List[str],
    threshold: float,
    output_dir: Path,
    backend: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    연도별 산출물 평가 요약을 JSON으로 저장합니다.
    - 유사도 분포(상삼각) 통계
    - centered 유사도 분포 통계
    - 엣지/그래프 통계(밀도, 평균 차수, 컴포넌트 크기 등)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n = int(S.shape[0])
    tri_S = _tri_upper_values(S, k=1)
    tri_C = _tri_upper_values(S_centered, k=1)

    # degree 계산
    id_to_idx = {fid: i for i, fid in enumerate(firm_ids)}
    deg = np.zeros(n, dtype=np.int64)

    uf = UnionFind(n)
    for e in edges:
        a = str(e.get("src", e.get("source", e.get("firm_a", ""))))
        b = str(e.get("dst", e.get("target", e.get("firm_b", ""))))
        if a in id_to_idx and b in id_to_idx:
            ia, ib = id_to_idx[a], id_to_idx[b]
            if ia == ib:
                continue
            deg[ia] += 1
            deg[ib] += 1
            uf.union(ia, ib)

    m = int(len(edges))
    density = float(m / (n * (n - 1) / 2)) if n >= 2 else 0.0
    avg_degree = float(deg.mean()) if n > 0 else 0.0
    comp_sizes = uf.component_sizes() if n > 0 else []

    # 상위 차수 노드(최대 20개)
    topk = min(20, n)
    top_idx = np.argsort(-deg)[:topk].tolist()
    top_nodes = []
    for i in top_idx:
        top_nodes.append(
            {
                "firm_id": firm_ids[i],
                "firm_name": firm_names[i] if i < len(firm_names) else "",
                "degree": int(deg[i]),
            }
        )

    report: Dict[str, Any] = {
        "year": int(year),
        "backend": backend,
        "n_firms": n,
        "n_edges": m,
        "threshold": float(threshold),
        "density": density,
        "avg_degree": avg_degree,
        "degree_percentiles": _percentiles(deg.astype(np.float32), [50, 75, 90, 95, 99]),
        "components": {
            "n_components": int(len(comp_sizes)),
            "largest_component_size": int(comp_sizes[0]) if comp_sizes else 0,
            "top_component_sizes": [int(x) for x in comp_sizes[:10]],
        },
        "similarity_raw": {
            "min": float(np.min(tri_S)) if tri_S.size else float("nan"),
            "max": float(np.max(tri_S)) if tri_S.size else float("nan"),
            "mean": float(np.mean(tri_S)) if tri_S.size else float("nan"),
            "std": float(np.std(tri_S)) if tri_S.size else float("nan"),
            "percentiles": _percentiles(tri_S, [1, 5, 25, 50, 75, 95, 99]),
        },
        "similarity_centered": {
            "min": float(np.min(tri_C)) if tri_C.size else float("nan"),
            "max": float(np.max(tri_C)) if tri_C.size else float("nan"),
            "mean": float(np.mean(tri_C)) if tri_C.size else float("nan"),
            "std": float(np.std(tri_C)) if tri_C.size else float("nan"),
            "percentiles": _percentiles(tri_C, [1, 5, 25, 50, 75, 95, 99]),
        },
        "top_degree_nodes": top_nodes,
    }

    if extra:
        report["extra"] = extra

    out_path = output_dir / f"evaluation_{year}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def save_eval_summary(
    *,
    all_reports: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    여러 연도 평가를 모아서 JSONL과 JSON 요약을 저장합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # jsonl
    jsonl_path = output_dir / "evaluation_all_years.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # json (간단 요약)
    summary = {
        "n_years": int(len(all_reports)),
        "years": [int(r["year"]) for r in all_reports],
        "by_year": {str(r["year"]): r for r in all_reports},
    }
    (output_dir / "evaluation_all_years.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
