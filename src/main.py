# src/main.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .preprocess import load_jsonl, group_by_year, GEO_STOPWORDS
from .vectorize import build_year_vocab, build_presence_matrix
from .similarity import l2_normalize_rows, compute_cosine_similarity_matrix, median_center_similarity
from .network import build_tnic_edges


def process_year(
    year: int,
    year_records: List[Dict[str, Any]],
    output_dir: Path,
    geo_stopwords: set[str],
    similarity_threshold: float,
    max_doc_freq_ratio: float = 0.25,
) -> None:
    """
    단일 연도에 대해 전체 파이프라인 수행:

      1) vocabulary 생성
      2) 기업-단어 이진 행렬 생성
      3) 단위벡터화
      4) 코사인 유사도 행렬 계산
      5) 중앙값 보정
      6) TNIC 엣지 생성
      7) 결과 저장 (npy, jsonl)
    """
    print(f"===== Processing year {year} (num_records={len(year_records)}) =====")

    # 1. vocab 생성 (지리 단어 제거 + 25% 이상 공통 단어 제거)
    vocab, df_counts = build_year_vocab(
        year_records,
        geo_stopwords=geo_stopwords,
        max_doc_freq_ratio=max_doc_freq_ratio,
    )
    print(f"- vocab size (after filters): {len(vocab)}")

    if len(vocab) == 0:
        print(f"  [WARN] year {year}: vocab is empty, skipping.")
        return

    # 2. 기업-단어 이진 행렬
    X, firm_ids = build_presence_matrix(year_records, vocab)
    print(f"- presence matrix shape: {X.shape} (firms x words)")

    # 3. 단위벡터화
    X_norm = l2_normalize_rows(X)

    # 4. 코사인 유사도 행렬
    S = compute_cosine_similarity_matrix(X_norm)
    print(f"- similarity matrix shape: {S.shape}")

    # 5. 중앙값 보정
    S_centered = median_center_similarity(S)

    # 6. TNIC 엣지 생성 (원시 유사도 S 기준)
    edges = build_tnic_edges(
        S,
        firm_ids,
        year=year,
        threshold=similarity_threshold,
    )
    print(f"- TNIC edges (sim > {similarity_threshold}): {len(edges)}")

    # 7. 저장
    year_prefix = f"{year}"

    # similarity 행렬 저장
    np.save(output_dir / f"similarity_{year_prefix}.npy", S)
    np.save(output_dir / f"similarity_centered_{year_prefix}.npy", S_centered)

    # vocab, df_counts 저장 (옵션이지만 재현에 유용)
    with (output_dir / f"vocab_{year_prefix}.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    with (output_dir / f"df_counts_{year_prefix}.json").open("w", encoding="utf-8") as f:
        json.dump(df_counts, f, ensure_ascii=False)

    # edges 저장
    edges_path = output_dir / f"tnic_edges_{year_prefix}.jsonl"
    with edges_path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")

    print(f"- saved similarity matrices, vocab, and edges for year {year}")


def process_all_years(
    jsonl_path: str | Path,
    output_dir: str | Path,
    similarity_threshold: float = 0.2132,
    max_doc_freq_ratio: float = 0.25,
) -> None:
    """
    전체 jsonl 파일을 읽고 연도별로 분리한 뒤,
    각 연도에 대해 process_year를 실행.
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(jsonl_path)
    by_year = group_by_year(records)

    # 기본 지리 단어 세트 사용 (원하면 외부에서 다른 세트를 넘겨도 됨)
    geo_stopwords = GEO_STOPWORDS

    for year, year_records in sorted(by_year.items()):
        process_year(
            year=year,
            year_records=year_records,
            output_dir=output_dir,
            geo_stopwords=geo_stopwords,
            similarity_threshold=similarity_threshold,
            max_doc_freq_ratio=max_doc_freq_ratio,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TNIC 파이프라인: SEC 10-K 텍스트 기반 기업 유사도 네트워크 생성"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="입력 jsonl 파일 경로 (firm_id, year, tokens 필드를 포함)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="결과 파일을 저장할 디렉터리 경로",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2132,
        help="TNIC 코사인 유사도 임계값 (기본 0.2132 ≒ 21.32%)",
    )
    parser.add_argument(
        "--max_df_ratio",
        type=float,
        default=0.25,
        help="전체 기업의 몇 퍼센트 이상이 사용하는 단어를 제거할지 비율 (기본 0.25)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_all_years(
        jsonl_path=args.input,
        output_dir=args.output_dir,
        similarity_threshold=args.threshold,
        max_doc_freq_ratio=args.max_df_ratio,
    )


if __name__ == "__main__":
    main()
