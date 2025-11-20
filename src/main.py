# src/main.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .preprocess import load_jsonl, group_by_year, GEO_STOPWORDS
from .vectorize import build_year_filtered_texts
from .bert_encoder import BertEncoder   # ← 새로 추가
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
         - max_doc_freq_ratio 이상 기업이 사용하는 토큰 제거
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
    print(f"- encoding texts with encoder: {type(encoder).__name__}")
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


def _load_all_records(
    input_path: str | Path | None,
    input_dir: str | Path | None,
) -> List[Dict[str, Any]]:
    """
    단일 jsonl 파일 또는 디렉터리(아래의 모든 .jsonl)를 읽어 전체 records 리스트로 반환.
    """
    records: List[Dict[str, Any]] = []

    # 2019년 사업보고서만 대상
    TARGET_YEAR = 2019

    if input_dir is not None:
        base = Path(input_dir)

        # 예시 구조:
        # input_dir/
        #   ├─ 삼성전자/
        #   │    ├─ 사업보고서(2018).jsonl
        #   │    └─ 사업보고서(2019).jsonl
        #   ├─ 현대자동차/
        #   │    └─ 사업보고서(2019).jsonl
        #   └─ ...
        #
        # 위와 같은 구조에서 2019년 파일만 찾기
        pattern = f"*{TARGET_YEAR}*.jsonl"
        jsonl_files = sorted(base.glob(pattern))

        if not jsonl_files:
            raise FileNotFoundError(
                f"input_dir={input_dir} 아래에서 {TARGET_YEAR}년 사업보고서 패턴({pattern})에 맞는 .jsonl 파일이 없습니다."
            )

        print(
            f"[INFO] input_dir={input_dir} 에서 "
            f"{TARGET_YEAR}년 사업보고서 jsonl 파일 {len(jsonl_files)}개를 로딩합니다."
        )

        # 회사별 2019년 사업보고서 파일 이터레이션
        for p in jsonl_files:
            company_name = p.parent.name  # 상위 디렉터리명을 회사 이름으로 활용
            print(f"  - 회사: {company_name}, 파일: {p.name}, 경로: {p}")

            rs = load_jsonl(p)
            records.extend(rs)

    elif input_path is not None:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {p}")
        print(f"[INFO] input file={p} 로딩")
        records = load_jsonl(p)
    else:
        raise ValueError("input 또는 input_dir 중 하나는 반드시 지정해야 합니다.")

    print(f"[INFO] 총 records 수: {len(records)}")
    return records



def process_all_years(
    input_path: str | Path | None,
    input_dir: str | Path | None,
    output_dir: str | Path,
    encoder: BertEncoder,
    similarity_threshold: float = 0.2132,
    max_doc_freq_ratio: float = 0.25,
) -> None:
    """
    전체 jsonl 파일(또는 디렉터리)을 읽고 연도별로 분리한 뒤,
    각 연도에 대해 BERT 임베딩 기반 TNIC 네트워크를 생성한다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 전체 records 로드
    records = _load_all_records(input_path=input_path, input_dir=input_dir)
    by_year = group_by_year(records)

    # 2) BERT는 별도의 fit 과정 없이 바로 encode_texts를 사용
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
        description="TNIC 파이프라인 (한국어 BERT 임베딩 기반 기업 유사도 네트워크 생성)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        help="입력 jsonl 파일 경로 (firm_id, year, text 또는 tokens 필드를 포함)",
    )
    group.add_argument(
        "--input_dir",
        help="여러 jsonl 파일이 들어 있는 디렉터리 경로 (하위 *.jsonl 전체 사용)",
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

    # BERT 관련 옵션
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="skt/kobert-base-v1",
        help="HuggingFace BERT 계열 모델 이름 (기본 skt/kobert-base-v1)",
    )
    parser.add_argument(
        "--bert_device",
        type=str,
        default=None,
        help="모델 디바이스 설정 (예: cpu, cuda, cuda:0). 기본값은 자동 선택.",
    )
    parser.add_argument(
        "--bert_batch_size",
        type=int,
        default=16,
        help="BERT 인코딩 배치 크기 (기본 16)",
    )
    parser.add_argument(
        "--bert_max_length",
        type=int,
        default=256,
        help="BERT 토크나이저 max_length (기본 256)",
    )
    parser.add_argument(
        "--bert_pooler",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="문장 임베딩 풀링 방식: cls 또는 mean (기본 cls)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # BERT 인코더 생성
    encoder = BertEncoder(
        model_name=args.bert_model_name,
        device=args.bert_device,
        batch_size=args.bert_batch_size,
        max_length=args.bert_max_length,
        pooler=args.bert_pooler,
    )

    process_all_years(
        input_path=args.input,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        encoder=encoder,
        similarity_threshold=args.threshold,
        max_doc_freq_ratio=args.max_df_ratio,
    )


if __name__ == "__main__":
    main()
