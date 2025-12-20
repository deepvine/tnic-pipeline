# src/main_embed.py
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from .preprocess import load_jsonl, GEO_STOPWORDS
from .sbert_encoder import KiwiSbertEncoder
from .chunked_openai_encoder import ChunkedOpenAIEncoder
from .tfidf_encoder import TfidfEncoder
from .vectorize import build_year_filtered_texts
from .similarity import (
    l2_normalize_rows_dense,
    compute_cosine_similarity_matrix_dense,
    median_center_similarity,
)
from .network import build_tnic_edges
from .evaluation import evaluate_year, save_eval_summary


MAX_CLEAN_INPUT = 1_000_000
MAX_RAW_CHARS = 1_000_000


def clean_text_for_relatedness(text: str) -> str:
    financial_patterns = [
        r"\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:백만원|억원|천원|원|조원|만원)",
        r"\d{1,3}(?:,\d{3})*(?:\.\d+)?%",
        r"\(단위\s*:\s*[^)]+\)",
        r"(?:전기|당기|전년|금년)(?:대비|말|초)?",
        r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*)?(?:주|좌)",
    ]
    date_patterns = [
        r"\d{4}[년./-]\s*\d{1,2}[월./-]\s*\d{1,2}일?",
        r"\d{4}[년.]?\s*\d{1,2}[월.]?",
        r"제\s*\d+\s*기",
        r"\d{4}년도?",
        r"(?:상반기|하반기|[1-4]분기|반기|사업연도)",
    ]
    legal_patterns = [
        r"상법|자본시장법|금융위원회|금융감독원|한국거래소",
        r"공시규정|공정거래법|독점규제법|증권거래법",
        r"주주총회|이사회|감사위원회|사외이사",
        r"정관|내부회계관리제도|내부통제",
    ]
    accounting_patterns = [
        r"외부감사|내부감사|감사보고서|감사인",
        r"재무제표|연결재무제표|별도재무제표",
        r"한국채택국제회계기준|K-?IFRS|기업회계기준",
        r"회계처리기준|회계정책|중요한 회계정책",
        r"자산총계|부채총계|자본총계|자기자본",
        r"유동자산|비유동자산|유동부채|비유동부채",
        r"영업이익|당기순이익|매출총이익|법인세비용",
        r"감가상각|손상차손|대손충당금",
    ]
    format_patterns = [
        r"해당\s*사항\s*없음|해당사항\s*없음|해당없음",
        r"주석\s*참조|주석참조|별도\s*기재",
        r"기재\s*생략|생략함|상기\s*참조",
        r"상세\s*내용은?\s*[^.]*참조",
        r"-{2,}|={2,}|_{2,}",
        r"※|▶|▷|●|○|◎|■|□",
    ]
    address_patterns = [
        r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
        r"(?:특별시|광역시|도|특별자치시|특별자치도)?"
        r"(?:\s*[가-힣]+(?:시|군|구|읍|면|동|리|로|길))+",
        r"\d{2,3}-\d{3,4}-\d{4}",
        r"(?:전화|팩스|FAX|TEL|HP)\s*:?\s*\d",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        r"(?:http|www)[^\s]+",
    ]
    stopword_patterns = [
        r"당사|본사|폐사|귀사",
        r"본\s*보고서|동\s*보고서|상기|하기|전기|후기",
        r"참고로|아울러|한편|또한|그리고|따라서|그러나|하지만",
        r"있습니다|없습니다|됩니다|합니다|입니다|습니다",
        r"것으로|바와\s*같이|대하여|관하여|의하여",
    ]
    hr_patterns = [
        r"대표이사|사내이사|사외이사|감사|상무|전무|이사",
        r"정규직|비정규직|계약직|임원|직원수?",
        r"남|여|명|인",
    ]

    all_patterns = (
        financial_patterns
        + date_patterns
        + legal_patterns
        + accounting_patterns
        + format_patterns
        + address_patterns
        + stopword_patterns
        + hr_patterns
    )

    if len(text) > MAX_CLEAN_INPUT:
        text = text[:MAX_CLEAN_INPUT]

    cleaned = text
    for pattern in all_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\b[\d,]+\.?\d*\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\b[가-힣a-zA-Z]\b", "", cleaned)
    return cleaned.strip()


def remove_template_mean(embeddings: np.ndarray) -> np.ndarray:
    template = embeddings.mean(axis=0, keepdims=True)
    debiased = embeddings - template
    norms = np.linalg.norm(debiased, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return debiased / norms


def build_full_texts_from_records(
    year_records: List[Dict[str, Any]],
    max_raw_chars: int = MAX_RAW_CHARS,
) -> Tuple[List[str], List[str], List[str]]:
    full_texts: List[str] = []
    corp_codes: List[str] = []
    corp_names: List[str] = []

    print(f"  → Extracting full texts from {len(year_records)} records...")

    for idx, r in enumerate(tqdm(year_records, desc="    Processing records", unit="record", leave=False)):
        t0 = time.perf_counter()

        corp_code = str(r.get("stock_code", ""))
        corp_name = str(r.get("corp_name", ""))

        text = r.get("parsed_business_content", "")
        if not text or not text.strip():
            continue

        raw_len = len(text)
        if raw_len > max_raw_chars:
            print(f"\n[DROP] too long idx={idx} code={corp_code} name={corp_name} raw_len={raw_len}")
            continue

        if raw_len > 300_000:
            print(f"\n[WARN] very long text idx={idx} code={corp_code} name={corp_name} raw_len={raw_len}")

        try:
            text = clean_text_for_relatedness(text)
        except Exception as e:
            print(f"\n[ERROR] clean_text failed idx={idx} code={corp_code} name={corp_name} raw_len={raw_len}: {e}")
            raise

        dt = time.perf_counter() - t0
        if dt > 5.0:
            head = text[:120].replace("\n", " ")
            print(
                f"[SLOW] idx={idx} code={corp_code} name={corp_name} "
                f"raw_len={raw_len} cleaned_len={len(text)} took={dt:.2f}s head={head!r}"
            )

        full_texts.append(text)
        corp_codes.append(corp_code)
        corp_names.append(corp_name)

    print(f"  → Extracted {len(full_texts)} firms with valid text")
    return full_texts, corp_codes, corp_names


def _load_all_records(
    input_path: str | Path | None,
    input_dir: str | Path | None,
) -> Dict[str, List[Dict[str, Any]]]:
    records_by_year: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if input_dir is not None:
        base = Path(input_dir)
        jsonl_files = sorted(base.rglob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"input_dir={input_dir} 아래에서 .jsonl 파일이 없습니다.")
        print(f"[INFO] input_dir={input_dir} 에서 jsonl 파일 {len(jsonl_files)}개를 로딩합니다.")

        for p in jsonl_files:
            m = re.search(r"\((\d{4})", p.name)
            file_year: Optional[str] = m.group(1) if m else None

            company_name = p.parent.name
            print(f"  - 회사: {company_name}, 파일: {p.name}, 연도: {file_year}")

            rs = load_jsonl(p)
            for rec in rs:
                if len(rec.keys()) == 1:
                    inner_key = list(rec.keys())[0]
                    rec = rec[inner_key]

                if file_year is not None:
                    rec["year"] = int(file_year)

                records_by_year[file_year].append(rec)

    elif input_path is not None:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {p}")

        print(f"[INFO] input_file={p} 로딩")

        m = re.search(r"\((\d{4})", p.name)
        file_year: Optional[str] = m.group(1) if m else None

        rs = load_jsonl(p)
        for rec in rs:
            if len(rec.keys()) == 1:
                inner_key = list(rec.keys())[0]
                if isinstance(rec[inner_key], dict):
                    rec = rec[inner_key]
            if file_year is not None:
                rec["year"] = int(file_year)
            records_by_year[file_year].append(rec)
    else:
        raise ValueError("input 또는 input_dir 중 하나는 반드시 지정해야 합니다.")

    total = sum(len(v) for v in records_by_year.values())
    print(f"[INFO] 총 records 수: {total}")
    year_dist = {y: len(v) for y, v in sorted(records_by_year.items())}
    print(f"[INFO] 연도별 분포: {year_dist}")

    return dict(records_by_year)


def _parse_years(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a.strip()), int(b.strip())
            if a_i <= b_i:
                out.extend(list(range(a_i, b_i + 1)))
            else:
                out.extend(list(range(b_i, a_i + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def _save_outputs_basic(
    output_dir: Path,
    year: int,
    S: np.ndarray,
    S_centered: np.ndarray,
    edges: List[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"similarity_{year}.npy", S)
    np.save(output_dir / f"similarity_centered_{year}.npy", S_centered)
    edges_path = output_dir / f"tnic_edges_{year}.jsonl"
    with edges_path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def process_year_sbert_or_openai(
    *,
    year: int,
    year_records: List[Dict[str, Any]],
    output_dir: Path,
    encoder: Any,  # KiwiSbertEncoder or ChunkedOpenAIEncoder
    similarity_threshold: float,
    max_raw_chars: int,
) -> None:
    print("\n" + "=" * 60)
    print(f"Processing year {year} (records={len(year_records)})")
    print("=" * 60 + "\n")

    print("[1/6] Extracting full texts...")
    full_texts, corp_codes, corp_names = build_full_texts_from_records(
        year_records, max_raw_chars=max_raw_chars
    )

    print("\n[DEBUG] Sample full_texts:")
    for i in range(min(3, len(full_texts))):
        print(f"  [{i}] {full_texts[i][:200]}...")
        print(f"      Length: {len(full_texts[i])} chars")
        print(f"      Words: {len(full_texts[i].split())} tokens")

    print(f"- num firms: {len(corp_codes)}")
    if len(corp_codes) == 0:
        print(f"[WARN] year {year}: no firms, skipping.")
        return

    lengths = [len(t) for t in full_texts]
    print("\n[텍스트 통계]")
    print(f"  기업 수: {len(corp_codes)}")
    print(f"  최소 길이: {min(lengths):,} chars")
    print(f"  최대 길이: {max(lengths):,} chars")
    print(f"  평균 길이: {np.mean(lengths):,.0f} chars")
    print(f"  중앙값: {np.median(lengths):,.0f} chars")

    print("\n[2/6] Encoding texts...")
    embeddings = encoder.encode_texts(full_texts)
    print(f"  embeddings shape: {embeddings.shape}")

    X_norm = remove_template_mean(embeddings)

    norms = np.linalg.norm(X_norm, axis=1)
    print(f"Norm mean: {norms.mean():.4f}")
    print(f"Norm std: {norms.std():.4f}")
    print(f"All norms ≈ 1.0? {np.allclose(norms, 1.0, atol=1e-6)}")

    print("\n[4/6] Computing cosine similarity matrix...")
    S = compute_cosine_similarity_matrix_dense(X_norm)
    print(f"  similarity matrix shape: {S.shape}")

    print("\n[5/6] Median centering similarity matrix...")
    S_centered = median_center_similarity(S)
    S_centered = (S_centered + S_centered.T) / 2

    print("\n[6/6] Building TNIC edges...")
    edges = build_tnic_edges(
        S_centered,
        corp_codes,
        corp_names,
        year=year,
        threshold=similarity_threshold,
    )
    print(f"  TNIC edges (sim > {similarity_threshold}): {len(edges)}")

    print("\n[7/6] Saving results...")
    _save_outputs_basic(output_dir=output_dir, year=year, S=S, S_centered=S_centered, edges=edges)

    print("\n[8/6] Evaluating outputs...")
    evaluate_year(
        year=year,
        S=S,
        S_centered=S_centered,
        edges=edges,
        firm_ids=corp_codes,
        firm_names=corp_names,
        threshold=similarity_threshold,
        output_dir=output_dir,
        backend="sbert_or_openai",
        extra=extra_eval,
    )

    print(f"✓ Completed year {year}")


def process_year_tfidf(
    *,
    year: int,
    year_records: List[Dict[str, Any]],
    output_dir: Path,
    encoder: TfidfEncoder,
    similarity_threshold: float,
    max_doc_freq_ratio: float,
) -> None:
    """
    TF-IDF 파이프라인:
      1) build_year_filtered_texts로 clean_texts/firm_ids/firm_names 생성
      2) TF-IDF 임베딩
      3) L2 정규화
      4) cosine similarity
      5) median center
      6) edges
      7) similarity/centered/df_counts/high_freq_tokens/edges 저장
    """
    print("\n" + "=" * 60)
    print(f"[TF-IDF] Processing year {year} (records={len(year_records)})")
    print("=" * 60 + "\n")

    print("[1/7] Building filtered texts...")
    clean_texts, firm_ids, firm_names, df_counts, high_freq_tokens = build_year_filtered_texts(
        year_records,
        geo_stopwords=GEO_STOPWORDS,
        max_doc_freq_ratio=max_doc_freq_ratio,
    )

    print(f"- num firms: {len(firm_ids)}")
    print(f"- num high-freq tokens (> {max_doc_freq_ratio*100:.1f}% firms): {len(high_freq_tokens)}")
    if len(firm_ids) == 0:
        print(f"[WARN] year {year}: no firms, skipping.")
        return

    print("[2/7] Encoding texts with TF-IDF...")
    embeddings = encoder.encode_texts(clean_texts)
    print(f"- embeddings shape: {embeddings.shape}")

    print("[3/7] L2 normalizing embeddings...")
    X_norm = l2_normalize_rows_dense(embeddings)

    print("[4/7] Computing cosine similarity matrix...")
    S = compute_cosine_similarity_matrix_dense(X_norm)
    print(f"- similarity matrix shape: {S.shape}")

    print("[5/7] Median centering similarity matrix...")
    S_centered = median_center_similarity(S)

    print("[6/7] Building TNIC edges...")
    edges = build_tnic_edges(
        S,
        firm_ids,
        firm_names,
        year=year,
        threshold=similarity_threshold,
    )
    print(f"- TNIC edges (sim > {similarity_threshold}): {len(edges)}")

    print("[7/7] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"similarity_{year}.npy", S)
    np.save(output_dir / f"similarity_centered_{year}.npy", S_centered)

    with (output_dir / f"df_counts_{year}.json").open("w", encoding="utf-8") as f:
        json.dump(df_counts, f, ensure_ascii=False)

    with (output_dir / f"high_freq_tokens_{year}.json").open("w", encoding="utf-8") as f:
        json.dump(high_freq_tokens, f, ensure_ascii=False)

    edges_path = output_dir / f"tnic_edges_{year}.jsonl"
    with edges_path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("[8/7] Evaluating outputs...")
    evaluate_year(
        year=year,
        S=S,
        S_centered=S_centered,
        edges=edges,
        firm_ids=firm_ids,
        firm_names=firm_names,
        threshold=similarity_threshold,
        output_dir=output_dir,
        backend="tfidf",
        extra={
            "text_mode": "filtered",
            "max_doc_freq_ratio": float(max_doc_freq_ratio),
            "n_high_freq_tokens": int(len(high_freq_tokens)),
            "tfidf_vocab_size": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        },
    )

    print(f"✓ [TF-IDF] Completed year {year}")


def process_all_years(
    *,
    input_path: str | Path | None,
    input_dir: str | Path | None,
    output_dir: str | Path,
    backend: str,
    encoder: Any,
    similarity_threshold: float,
    years: Optional[List[int]],
    max_raw_chars: int,
    max_df_ratio: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_all_records(input_path=input_path, input_dir=input_dir)

    for year_str, year_records in sorted(records.items()):
        if year_str is None:
            continue
        year = int(year_str)

        if years is not None and year not in years:
            continue

        if backend == "tfidf":
            process_year_tfidf(
                year=year,
                year_records=year_records,
                output_dir=output_dir,
                encoder=encoder,
                similarity_threshold=similarity_threshold,
                max_doc_freq_ratio=max_df_ratio,
            )
        else:
            process_year_sbert_or_openai(
                year=year,
                year_records=year_records,
                output_dir=output_dir,
                encoder=encoder,
                similarity_threshold=similarity_threshold,
                max_raw_chars=max_raw_chars,
            )
            
    eval_files = sorted(output_dir.glob("evaluation_*.json"))
    merged: List[Dict[str, Any]] = []
    for p in eval_files:
        try:
            merged.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if merged:
        save_eval_summary(all_reports=merged, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TNIC 파이프라인 (sbert/openai/tfidf 선택 실행)")

    parser.add_argument(
        "--backend",
        required=True,
        choices=["sbert", "openai", "tfidf"],
        help="임베딩 백엔드 선택: sbert, openai, tfidf",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="입력 jsonl 파일 경로")
    group.add_argument("--input_dir", help="여러 jsonl 파일이 들어 있는 디렉터리 경로")

    parser.add_argument("--output_dir", required=True, help="결과 파일을 저장할 디렉터리 경로")
    parser.add_argument("--threshold", type=float, default=0.2132, help="TNIC 코사인 유사도 임계값")

    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help='처리할 연도. 예: "2024" / "2020,2021,2024" / "2010-2012,2024". 미지정 시 전체',
    )

    # 공통 안전장치 (sbert/openai만 사용)
    parser.add_argument(
        "--max_raw_chars",
        type=int,
        default=MAX_RAW_CHARS,
        help=f"(sbert/openai) 원문 텍스트 최대 길이(초과시 DROP). 기본 {MAX_RAW_CHARS}",
    )

    # -------- SBERT 옵션 --------
    parser.add_argument(
        "--sbert_model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SBERT 모델 이름",
    )
    parser.add_argument("--bert_device", type=str, default=None, help="(호환용) 디바이스 옵션")
    parser.add_argument("--bert_batch_size", type=int, default=16, help="SBERT 배치 크기")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "sum", "max", "weighted"],
        help="(호환/공통) 집계 방식. openai에서 실제 사용, sbert는 호환용",
    )

    # -------- OpenAI 옵션 --------
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API 키")
    parser.add_argument(
        "--openai_model",
        type=str,
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        help="OpenAI 임베딩 모델",
    )
    parser.add_argument("--max_tokens", type=int, default=8000, help="OpenAI 청크 max tokens")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="OpenAI 청크 overlap")

    # -------- TF-IDF 옵션 --------
    parser.add_argument("--max_df_ratio", type=float, default=0.25, help="vectorize 단계 high-freq 토큰 제거 비율")
    parser.add_argument("--tfidf_min_df", type=int, default=2, help="TF-IDF Vectorizer min_df")
    parser.add_argument("--tfidf_max_df", type=float, default=0.9, help="TF-IDF Vectorizer max_df")

    return parser.parse_args()


def make_encoder(args: argparse.Namespace):
    if args.backend == "sbert":
        return KiwiSbertEncoder(model_name=args.sbert_model_name, batch_size=args.bert_batch_size)

    if args.backend == "openai":
        return ChunkedOpenAIEncoder(
            model_name=args.openai_model,
            api_key=args.openai_api_key,
            max_tokens=args.max_tokens,
            chunk_overlap=args.chunk_overlap,
            aggregation=args.aggregation,
        )

    if args.backend == "tfidf":
        return TfidfEncoder(min_df=args.tfidf_min_df, max_df=args.tfidf_max_df)

    raise ValueError(f"Unknown backend: {args.backend}")


def main() -> None:
    args = parse_args()
    years = _parse_years(args.years) if args.years else None

    print("=" * 60)
    print("TNIC 네트워크 생성 시작")
    print("=" * 60)
    if years is not None:
        print(f"[INFO] years filter: {years}")

    encoder = make_encoder(args)

    print("\n[설정]")
    print(f"  backend: {args.backend}")
    print(f"  input: {args.input}")
    print(f"  input_dir: {args.input_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  threshold: {args.threshold}")

    if args.backend in ("sbert", "openai"):
        print(f"  max_raw_chars: {args.max_raw_chars}")

    if args.backend == "sbert":
        print(f"  sbert_model_name: {args.sbert_model_name}")
        print(f"  bert_batch_size: {args.bert_batch_size}")
        print(f"  aggregation: {args.aggregation} (compat)")
    elif args.backend == "openai":
        print(f"  openai_model: {args.openai_model}")
        print(f"  max_tokens: {args.max_tokens}")
        print(f"  chunk_overlap: {args.chunk_overlap}")
        print(f"  aggregation: {args.aggregation}")
    else:
        print(f"  max_df_ratio: {args.max_df_ratio}")
        print(f"  tfidf_min_df: {args.tfidf_min_df}")
        print(f"  tfidf_max_df: {args.tfidf_max_df}")

    process_all_years(
        input_path=args.input,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        encoder=encoder,
        similarity_threshold=args.threshold,
        years=years,
        max_raw_chars=args.max_raw_chars,
        max_df_ratio=args.max_df_ratio,
    )

    print("\n" + "=" * 60)
    print("모든 처리 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
