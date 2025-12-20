# src/vectorize.py
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Tuple, Literal, Optional

from tqdm import tqdm

from .preprocess import record_to_tokens


def build_year_filtered_texts(
    year_records: List[Dict[str, Any]],
    geo_stopwords: set[str],
    max_doc_freq_ratio: float = 0.25,
    *,
    text_field: str = "parsed_business_content",
    max_raw_chars: int = 1_000_000,
    slow_sec: float = 5.0,
    on_error: Literal["skip", "raise"] = "skip",
) -> Tuple[List[str], List[str], List[str], Dict[str, int], List[str]]:
    """
    한 해의 레코드(기업-연도)를 받아서 다음을 수행:

      1) 각 레코드에서 token 리스트 추출 (길이 가드/슬로우 로그 포함)
      2) 토큰을 소문자로 normalize (영어 대비용)
      3) 기업별 고유 토큰 집합으로 문서 빈도(df) 계산
      4) 전체 기업의 max_doc_freq_ratio 초과하는 토큰을 흔한 단어로 정의
      5) 각 기업별로:
         - 흔한 토큰 제거
         - 지리 관련 토큰 제거
         - 남은 토큰을 공백으로 join → clean_text
    """
    corp_codes: List[str] = []
    corp_names: List[str] = []
    firm_tokens_list: List[List[str]] = []
    df_counter: Counter[str] = Counter()

    dropped_too_long = 0
    dropped_empty = 0
    dropped_error = 0

    print(f"  → Step 1/2: Extracting tokens from {len(year_records)} records...")
    for idx, r in enumerate(tqdm(year_records, desc="    Processing records", unit="record", leave=False)):
        corp_code = str(r.get("stock_code", ""))
        corp_name = str(r.get("corp_name", ""))

        text = r.get(text_field, "")
        if not text or not str(text).strip():
            dropped_empty += 1
            continue

        raw_len = len(text)
        if raw_len > max_raw_chars:
            print(f"\n[DROP] too long idx={idx} code={corp_code} name={corp_name} raw_len={raw_len}")
            dropped_too_long += 1
            continue

        t0 = time.perf_counter()
        try:
            tokens_raw = record_to_tokens(text)
        except Exception as e:
            print(f"\n[ERROR] record_to_tokens failed idx={idx} code={corp_code} name={corp_name} raw_len={raw_len}: {e}")
            dropped_error += 1
            if on_error == "raise":
                raise
            continue
        dt = time.perf_counter() - t0

        if dt > slow_sec:
            head = str(text)[:120].replace("\n", " ")
            print(
                f"\n[SLOW] idx={idx} code={corp_code} name={corp_name} raw_len={raw_len} "
                f"tokens={len(tokens_raw)} took={dt:.2f}s head={head!r}"
            )

        tokens_norm = [t.lower() for t in tokens_raw if t]

        corp_codes.append(corp_code)
        corp_names.append(corp_name)
        firm_tokens_list.append(tokens_norm)

        for w in set(tokens_norm):
            df_counter[w] += 1

    print(
        f"  → Step 1/2 done. kept={len(corp_codes)} "
        f"dropped_empty={dropped_empty} dropped_too_long={dropped_too_long} dropped_error={dropped_error}"
    )

    num_firms = len(corp_codes)
    if num_firms == 0:
        # 빈 결과 방어
        return [], [], [], {}, []

    max_df = max_doc_freq_ratio * num_firms
    # 아래 줄은 원래 코드에 있었는데, 강제로 max_df=5로 덮어쓰고 있음.
    # 의도된 동작이면 유지, 아니면 주석 처리 권장.
    # max_df = 5

    df_counts: Dict[str, int] = {}
    high_freq_tokens: List[str] = []
    for w, df in df_counter.items():
        df_counts[w] = df
        if df > max_df:
            high_freq_tokens.append(w)

    high_freq_set = set(high_freq_tokens)
    geo_set = set(geo_stopwords)

    clean_texts: List[str] = []
    filtered_corp_codes: List[str] = []
    filtered_corp_names: List[str] = []

    print(f"  → Step 2/2: Filtering tokens and building clean texts...")
    for corp_code, corp_name, tokens in tqdm(
        zip(corp_codes, corp_names, firm_tokens_list),
        desc="    Filtering firms",
        unit="firm",
        total=len(corp_codes),
        leave=False,
    ):
        filtered_tokens: List[str] = []
        for t in tokens:
            if not t:
                continue
            if t in high_freq_set:
                continue
            if t in geo_set:
                continue
            filtered_tokens.append(t)

        clean_text = " ".join(filtered_tokens).strip()
        clean_texts.append(clean_text)
        filtered_corp_codes.append(corp_code)
        filtered_corp_names.append(corp_name)

    return clean_texts, filtered_corp_codes, filtered_corp_names, df_counts, high_freq_tokens
