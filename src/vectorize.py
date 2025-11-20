# src/vectorize.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from .preprocess import record_to_tokens


def build_year_filtered_texts(
    year_records: List[Dict[str, Any]],
    geo_stopwords: set[str],
    max_doc_freq_ratio: float = 0.25,
) -> Tuple[List[str], List[str], Dict[str, int], List[str]]:
    """
    한 해의 레코드(기업-연도)를 받아서 다음을 수행:

      1) 각 레코드에서 token 리스트 추출
      2) 토큰을 소문자로 normalize (영어 대비용)
      3) 기업별 고유 토큰 집합으로 문서 빈도(df) 계산
      4) 전체 기업의 max_doc_freq_ratio(기본 25%) 초과하는 토큰을 흔한 단어로 정의
      5) 각 기업별로:
         - 흔한 토큰 제거
         - 지리 관련 토큰 제거
         - 남은 토큰을 공백으로 join → clean_text

    반환:
      - clean_texts: 각 기업에 대해 필터링된 텍스트 문자열 리스트
      - corp_codes: clean_texts와 같은 순서의 corp_code 리스트
      - df_counts: 토큰 -> 해당 연도에서 몇 개 기업이 사용했는지
      - high_freq_tokens: 너무 흔해서 제거된 토큰 리스트
    """
    # 1. 기업별 토큰 및 문서 빈도 계산
    corp_codes: List[str] = []
    firm_tokens_list: List[List[str]] = []
    df_counter: Counter[str] = Counter()

    for r in year_records:
        corp_code = str(r["corp_code"])
        tokens_raw = record_to_tokens(r['parsed_business_content'])
        # 한국어는 대소문자 구분이 크게 없지만, 영어 대비를 위해 소문자 변환
        tokens_norm = [t.lower() for t in tokens_raw]

        corp_codes.append(corp_code)
        firm_tokens_list.append(tokens_norm)

        unique_tokens = set(tokens_norm)
        for w in unique_tokens:
            df_counter[w] += 1

    num_firms = len(corp_codes)
    max_df = max_doc_freq_ratio * num_firms
    max_df = 5
    # 2. 25% 이상 기업이 사용하는 토큰 필터링
    df_counts: Dict[str, int] = {}
    high_freq_tokens: List[str] = []
    for w, df in df_counter.items():
        df_counts[w] = df
        # if df > max_df:
        #     high_freq_tokens.append(w)

    high_freq_set = set(high_freq_tokens)

    # 지리 단어는 영어는 소문자, 한국어는 그대로 들어있음
    geo_set = set(geo_stopwords)

    # 3. 각 기업별 clean_text 생성
    clean_texts: List[str] = []
    filtered_corp_codes: List[str] = []

    for corp_code, tokens in zip(corp_codes, firm_tokens_list):
        filtered_tokens: List[str] = []
        for t in tokens:
            # 빈 토큰 건너뜀
            if not t:
                continue
            # 흔한 토큰 제거
            if t in high_freq_set:
                continue
            # 지리 관련 토큰 제거
            if t in geo_set:
                continue
            filtered_tokens.append(t)

        # 완전히 비어있으면 스킵할 수도 있지만, 여기서는 그대로 두고 빈 문자열로 처리
        clean_text = " ".join(filtered_tokens).strip()
        clean_texts.append(clean_text)
        filtered_corp_codes.append(corp_code)

    return clean_texts, filtered_corp_codes, df_counts, high_freq_tokens