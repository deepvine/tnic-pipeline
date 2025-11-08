# src/main.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .preprocess import load_jsonl, group_by_year, GEO_STOPWORDS
from .vectorize import build_year_filtered_texts
from .bert_encoder import BertEncoder
from .similarity import (
    l2_normalize_rows_dense,
    compute_cosine_similarity_matrix_dense,
    median_center_similarity,
)
from .network import build_tnic_edges


def process_year(
    year: int,
    year_records: List[Dict[str, Any]],
    output_dir: Path,
    encoder: BertEncoder,
    similarity_threshold: float,
    max_doc_freq_ratio: float,
) -> None:
    """
    단일 연도에 대해 전체 파이프라인 수행:

      1) 한국어 텍스트에서 토큰 추출 후
         - 25% 이상 기업이 사용하는 토큰 제거
         - 지리 관련 토큰 제거
      2) 남은 토큰을 공백으로 join → clean_text
      3) BERT 임베딩 생성 (기업별 벡터)
      4) L2 정규화
      5) 코사인 유사도 행렬 계산
      6) 중앙값 보정
      7) TNIC 엣지 생성
      8) 결과 저장
    """
    print(f"===== Processing year {year} (num_records={len(year_records)}) =====")

    # 1. clean_text 생성
    clean_texts, firm_ids, df_counts, high_freq_tokens = build_year_filtered_texts(
        year_records,
        geo_stopwords=GEO_STOPWORDS,
        max_doc_freq_ratio=max_doc_freq_ratio,
    )

    print(f"- num firms: {len(firm_ids)}")
    print(f"- num high-freq tokens (> {max_doc_freq_ratio*100:.1f}% firms): {len(high_freq_tokens)}")

    if len(firm_ids) == 0:
        print(f"  [WARN] year {year}: no firms, skipping.")
        return

    # 2. BERT 임베딩 생성
    print(f"- encoding texts with BERT: model={encoder.model_name}")
    embeddings = encoder.encode_texts(clean_texts)  # (num_firms, hidden_size)
    print(f"- embeddings shape: {embeddings.shape}")

    # 3. L2 정규화
    X_norm = l2_normalize_rows_dense(embeddings)

    # 4. 코사인 유사도 행렬
    S = compute_cosine_similarity_matrix_dense(X_norm)
    print(f"- similarity matrix shape: {S.shape}")

    # 5. 중앙값 보정
    S_centered = median_center_similarity(S)

    # 6. TNIC 엣지 생성
    edges = build_tnic_edges(
        S,
        firm_ids,
        year=year,
        threshold=similarity_threshold,
    )
    print(f"- TNIC edges (sim > {similarity_threshold}): {len(edges)}")

    # 7. 저장
    year_prefix = f"{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 유사도 행렬 저장
    np.save(output_dir / f"similarity_{year_prefix}.npy", S)
    np.save(output_dir / f"similarity_centered_{year_prefix}.npy", S_centered)

    # df_counts, high_freq_tokens 저장 (분석 재현용)
    with (output_dir / f"df_counts_{year_prefix}.json").open("w", encoding="utf-8") as f:
        json.dump(df_counts, f, ensure_ascii=False)

    with (output_dir / f"high_freq_tokens_{year_prefix}.json").open("w", encoding="utf-8") as f:
        json.dump(high_freq_tokens, f, ensure_ascii=False)

    # 엣지 저장
    edges_path = output_dir / f"tnic_edges_{year_prefix}.jsonl"
    with edges_path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")

    print(f"- saved similarity matrices and edges for year {year}")


def process_all_years(
    jsonl_path: str | Path,
    output_dir: str | Path,
    encoder: BertEncoder,
    similarity_threshold: float = 0.2132,
    max_doc_freq_ratio: float = 0.25,
) -> None:
    """
    전체 jsonl 파일을 읽고 연도별로 분리한 뒤,
    각 연도에 대해 process_year 실행.
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(jsonl_path)
    by_year = group_by_year(records)

    for year, year_records in sorted(by_year.items()):
        process_year(
            year=year,
            year_records=year_records,
            output_dir=output_dir,
            encoder=encoder,
            similarity_threshold=similarity_threshold,
            max_doc_freq_ratio=max_doc_freq_ratio,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TNIC 파이프라인 (한국어 BERT 기반 기업 유사도 네트워크 생성)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="입력 jsonl 파일 경로 (firm_id, year, text 또는 tokens 필드를 포함)",
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
        help="전체 기업의 몇 퍼센트 이상이 사용하는 토큰을 제거할지 비율 (기본 0.25)",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="klue/bert-base",
        help="사용할 한국어 BERT 모델 이름 (예: klue/bert-base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="모델을 올릴 디바이스 (예: cpu, cuda, cuda:0 등)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="BERT 토크나이저 max_length (기본 512)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="BERT 인퍼런스 배치 크기 (기본 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    encoder = BertEncoder(
        model_name=args.bert_model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    process_all_years(
        jsonl_path=args.input,
        output_dir=args.output_dir,
        encoder=encoder,
        similarity_threshold=args.threshold,
        max_doc_freq_ratio=args.max_df_ratio,
    )


if __name__ == "__main__":
    main()
