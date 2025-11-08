# src/preprocess.py

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List


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


def group_by_year(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    레코드를 year 기준으로 묶어서 반환.
    """
    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
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


def record_to_tokens(record: Dict[str, Any]) -> List[str]:
    """
    레코드에서 토큰 리스트를 추출.

    우선순위:
      1) tokens 필드가 있으면 그대로 사용
      2) text 필드가 있으면 공백 기준으로 토큰화

    더 정교한 한국어 형태소 분석은 이 단계 위에서 별도 전처리로 넣을 수 있음.
    """
    if "tokens" in record and record["tokens"]:
        tokens = [str(t).strip() for t in record["tokens"] if str(t).strip()]
        return tokens
    elif "text" in record and record["text"]:
        text = str(record["text"])
        # 매우 단순화된 토크나이징: 공백 기준
        tokens = [t.strip() for t in text.split() if t.strip()]
        return tokens
    else:
        return []
