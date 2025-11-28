# src/preprocess.py

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List
# from konlpy.tag import Okt
from kiwipiepy import Kiwi
import re

kiwi = Kiwi()

def extract_nouns(text):
    tokens = kiwi.tokenize(text)

    nouns = [t.form for t in tokens if t.tag.startswith("NN")]

    return nouns

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    jsonl 파일을 읽어서 레코드 리스트로 반환.

    각 줄은 하나의 JSON 객체라고 가정.
    예:
      {"firm_id": "...", "year": 1996, "text": "..."}
      또는 {"firm_id": "...", "year": 1996, "tokens": ["제품", "서비스", ...]}
    """
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

# 시장 위험 섹션 제거
def remove_market_risk_section(text: str) -> str:
    """
    '시장위험' 또는 '시장위험과 위험관리' 섹션을 찾아
    다음 섹션 번호(예: \n8. , \n9)) 전까지 삭제.
    """
    if not text:
        return text

    # 찾을 키워드
    keyword_pattern = r"(시장위험[^\n]*)"   # '시장위험', '시장위험과 위험관리', '시장위험과 위험관리 (연결기준)' 등 포함

    # 다음 섹션 헤더 패턴 (줄바꿈 + 숫자 + . or ) ex) "\n8. ", "\n9) "
    next_section_pattern = r"\n\s*\d{1,2}\s*[\.\)]\s+"

    # loop: 시장위험이 여러 번 등장할 수도 있으므로 반복 제거
    while True:
        m = re.search(keyword_pattern, text)
        if not m:
            break

        start_idx = m.start()

        # start 이후에서 다음 섹션 찾기
        next_header = re.search(next_section_pattern, text[start_idx:])
        if next_header:
            end_idx = start_idx + next_header.start()
            text = text[:start_idx] + text[end_idx:]
        else:
            # 다음 섹션 헤더 없으면 그냥 다 지움
            text = text[:start_idx]
            break

    return text

# 파생상품 섹션 제거 
def remove_market_derivatives_section(text: str) -> str:
    """
    '파생상품' 또는 '시장위험과 위험관리' 섹션을 찾아
    다음 섹션 번호(예: \n8. , \n9)) 전까지 삭제.
    """
    if not text:
        return text

    # 찾을 키워드
    keyword_pattern = r"(파생상품[^\n]*)"   # '시장위험', '시장위험과 위험관리', '시장위험과 위험관리 (연결기준)' 등 포함

    # 다음 섹션 헤더 패턴 (줄바꿈 + 숫자 + . or ) ex) "\n8. ", "\n9) "
    next_section_pattern = r"\n\s*\d{1,2}\s*[\.\)]\s+"

    # loop: 시장위험이 여러 번 등장할 수도 있으므로 반복 제거
    while True:
        m = re.search(keyword_pattern, text)
        if not m:
            break

        start_idx = m.start()

        # start 이후에서 다음 섹션 찾기
        next_header = re.search(next_section_pattern, text[start_idx:])
        if next_header:
            end_idx = start_idx + next_header.start()
            text = text[:start_idx] + text[end_idx:]
        else:
            # 다음 섹션 헤더 없으면 그냥 다 지움
            text = text[:start_idx]
            break

    return text

# 위험관리 섹션제거
def remove_risk_admission_section(text: str) -> str:
    """
    '위험관리' 또는 '위험관리 및 파생거래' 섹션을 찾아
    다음 섹션 번호(예: \n8. , \n9)) 전까지 삭제.
    """
    if not text:
        return text

    # 찾을 키워드
    keyword_pattern = r"(위험관리 및[^\n]*)"   # '시장위험', '시장위험과 위험관리', '시장위험과 위험관리 (연결기준)' 등 포함

    # 다음 섹션 헤더 패턴 (줄바꿈 + 숫자 + . or ) ex) "\n8. ", "\n9) "
    next_section_pattern = r"\n\s*\d{1,2}\s*[\.\)]\s+"

    # loop: 시장위험이 여러 번 등장할 수도 있으므로 반복 제거
    while True:
        m = re.search(keyword_pattern, text)
        if not m:
            break

        start_idx = m.start()

        # start 이후에서 다음 섹션 찾기
        next_header = re.search(next_section_pattern, text[start_idx:])
        if next_header:
            end_idx = start_idx + next_header.start()
            text = text[:start_idx] + text[end_idx:]
        else:
            # 다음 섹션 헤더 없으면 그냥 다 지움
            text = text[:start_idx]
            break

    return text


TERMS_TO_REMOVE = [
    "금액", "비중", "KRW", "USD", "백만원", "억원",
    "단위", "총매출액", "순매출액", "매출액", "년"
]

def remove_specific_terms(text: str) -> str:
    """
    TERMS_TO_REMOVE에 포함된 단어만 제거하고,
    나머지 텍스트는 그대로 유지
    """
    if not text:
        return text

    # 제거할 단어를 OR 정규식 패턴으로 변환
    pattern = r"|".join(map(re.escape, TERMS_TO_REMOVE))

    # 단어 제거 (대소문자 무시)
    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # 중복 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def clean_kor_eng_text(text: str) -> str:
    if not text:
        return text
    # 한글, 영어, 공백 제외 모두 제거
    text = re.sub(r"[^가-힣a-zA-Z\s]", "", text)
    
    # 공백 2칸 이상 → 1칸으로 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text

def group_by_year(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    레코드를 year 기준으로 묶어서 반환.
    """
    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if len(r.keys()) == 1:
            r = r[list(r.keys())[0]]
        year = int(r["year"])
        by_year[year].append(r)
    return by_year


def build_geo_stopwords_ko_en() -> set[str]:
    """
    한국어/영어 지리 관련 단어(stopwords) 집합 생성.

    실제 연구에서는 훨씬 더 풍부한 리스트를 써야 하지만,
    여기서는 예시 수준으로만 구성.
    단어는 모두 소문자로 저장해서 토큰과 비교 시 소문자로 맞춰 사용.
    한국어는 대소문자 구분이 없지만, 영어는 대비를 위해 소문자 처리.
    """
    # 영어 국가/도시
    countries_en = {
        "united states", "usa", "america", "canada", "mexico",
        "japan", "china", "germany", "france", "uk", "united kingdom",
        "korea", "south korea", "italy", "spain", "russia", "brazil",
    }
    cities_en = {
        "new york", "los angeles", "chicago", "houston", "phoenix",
        "philadelphia", "san antonio", "san diego", "dallas", "san jose",
        "seoul", "tokyo", "beijing", "shanghai", "hong kong", "london",
        "paris", "berlin", "madrid", "rome",
    }

    # 한국어 국가/도시/지역 (예시)
    geo_ko = {
        "대한민국", "한국", "미국", "중국", "일본", "유럽", "아시아",
        "서울", "부산", "인천", "대구", "대전", "광주", "울산",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    }

    geo = set()
    for w in countries_en | cities_en:
        geo.add(w.lower())
    # 한국어는 그대로
    for w in geo_ko:
        geo.add(w)
    return geo


GEO_STOPWORDS = build_geo_stopwords_ko_en()

# okt = Okt()

def record_to_tokens(record: Dict[str, Any]) -> List[str]:
    """
    레코드에서 토큰 리스트를 추출.

    우선순위:
      1) tokens 필드가 있으면 그대로 사용
      2) text 필드가 있으면 KoNLPy Okt로 명사만 추출
    """
    if isinstance(record, str):
        # nouns = okt.nouns(record)
        nouns = extract_nouns(record)
        tokens = [t.strip() for t in nouns if len(t.strip()) > 1]  # 1글자 제외
        return tokens
    elif "tokens" in record and record["tokens"]:
        tokens = [str(t).strip() for t in record["tokens"] if str(t).strip()]
        return tokens
    elif "text" in record and record["text"]:
        text = str(record["text"]).strip()
        if not text:
            return []
        # nouns = okt.nouns(text)
        nouns = extract_nouns(record)
        tokens = [t.strip() for t in nouns if len(t.strip()) > 1]  # 1글자 제외
        return tokens
    else:
        return []
