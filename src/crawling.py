import xml.etree.ElementTree as ET

def load_corp_list_from_dart(api_key):
    """
    DART corpCode API에서 전체 기업 목록을 받아서 DataFrame으로 반환.
    corp_code, corp_name, stock_code 포함.
    """
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}

    # corpCode.xml ZIP 다운로드
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"corpCode.xml 요청 실패: HTTP {r.status_code}")

    # ZIP 압축 풀기
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    xml_name = [n for n in zf.namelist() if n.lower().endswith(".xml")][0]
    xml_bytes = zf.read(xml_name)

    # XML 파싱
    root = ET.fromstring(xml_bytes)

    rows = []
    for el in root.findall("list"):
        corp_code = el.findtext("corp_code")
        corp_name = el.findtext("corp_name")
        stock_code = el.findtext("stock_code")

        rows.append({
            "corp_code": corp_code,
            "corp_name": corp_name,
            "stock_code": stock_code,
        })

    df = pd.DataFrame(rows)
    return df


import os
import io
import json
import time
import zipfile
import requests
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================
# 0. 환경 설정
# =========================================
API_KEY = "key"   # 본인 API KEY
OUTPUT_DIR = "/content//matched_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAILED_LIST_FILE = "/content/failed_reports1213.jsonl"

DOC_URL = "https://opendart.fss.or.kr/api/document.xml"
LIST_URL = "https://opendart.fss.or.kr/api/list.json"


# =========================================
# 실패 리스트 기록 함수
# =========================================
def log_failed_report(corp_code, corp_name, rcept_no, error_message):
    failed_report = {
        "corp_code": corp_code,
        "corp_name": corp_name,
        "rcept_no": rcept_no,
        "error_message": error_message
    }

    with open(FAILED_LIST_FILE, "a", encoding="utf-8") as f:
        json.dump(failed_report, f, ensure_ascii=False)
        f.write("\n")


# =========================================
# 1. 사업보고서 목록 조회
# =========================================
def get_business_reports(corp_code):
    all_reports = []
    page_no = 1

    while True:
        params = {
            "crtfc_key": API_KEY,
            "corp_code": corp_code,
            "bgn_de": "20050101",
            "end_de": "20251231",
            "last_reprt_at": "N",
            "pblntf_ty": "A",
            "pblntf_detail_ty": "A001",
            "page_no": page_no,
            "page_count": 100
        }

        res = requests.get(LIST_URL, params=params).json()

        if res.get("status") != "000":
            print(f"corp_code={corp_code} 사업보고서 조회 실패: {res.get('message')}")
            break

        reports = res.get("list", []) or []
        if not reports:
            break

        all_reports.extend(reports)

        if len(reports) < 100:
            break

        page_no += 1
        time.sleep(0.05)  # 최소 대기

    biz_reports = [r for r in all_reports if "사업보고서" in r.get("report_nm", "")]
    biz_reports.sort(key=lambda x: x.get("rcept_dt", ""), reverse=True)

    print(f" 최종 사업보고서 {len(biz_reports)}개 수집 완료")
    return biz_reports


# =========================================
# 2. HTML 태그 제거 (초고속)
# =========================================
TAG_RE = re.compile(r"<[^>]+>")

def strip_html_fast(html):
    return TAG_RE.sub("", html)


# =========================================
# 3. ZIP → XML → 사업의 내용 추출 (고속)
# =========================================
def fetch_business_text_only(rcept_no):

    params = {"crtfc_key": API_KEY, "rcept_no": str(rcept_no)}

    try:
        r = requests.get(DOC_URL, params=params, timeout=20)
    except Exception as e:
        return None, f"HTTP 요청 실패: {e}"

    if r.status_code != 200:
        return None, f"HTTP 상태코드 {r.status_code}"

    # ZIP 읽기
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
    except Exception as e:
        return None, f"ZIP 오류: {e}"

    # zip 내 XML 파일 바로 선택
    xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
    if not xml_files:
        return None, "XML 파일 없음"

    raw = zf.read(xml_files[0])

    # 인코딩 시도
    for enc in ("utf-8", "euc-kr", "cp949"):
        try:
            xml_text = raw.decode(enc)
            break
        except:
            xml_text = None

    if xml_text is None:
        return None, "XML 디코딩 실패"

   # II. 사업의 내용(보험업) 같은 변형까지 모두 잡기
    m_start = re.search(r'<TITLE[^>]*>\s*II\.\s*사업의\s*내용.*?</TITLE>', xml_text)
    m_end   = re.search(r'<TITLE[^>]*>\s*III\.\s*재무에\s*관한\s*사항.*?</TITLE>', xml_text)

    if not m_start or not m_end:
        return None, "사업의 내용/재무 섹션 TITLE 미검출"

    start = m_start.start()
    end = m_end.start()

    if end <= start:
        return None, "섹션 위치 역전(파싱 이상)"

    block = xml_text[start:end]


    # 태그 제거 (초고속)
    text_only = strip_html_fast(block).strip()

    if not text_only:
        return None, "사업의 내용 텍스트 없음"

    return text_only, None


# =========================================
# 4. 저장 함수
# =========================================
def save_business_section(corp_code, corp_name, year, rcept_no, report_nm):

    text, err = fetch_business_text_only(rcept_no)
    if text is None:
        print(f"실패: rcept_no={rcept_no}, 사유: {err}")
        log_failed_report(corp_code, corp_name, rcept_no, err)  # 실패 리스트에 기록
        return False

    fname = f"{corp_code}_{report_nm}_{corp_name}_{rcept_no}"
    fname = re.sub(r'[\\/:*?"<>|]', "_", fname)

    out_path = os.path.join(OUTPUT_DIR, fname + ".jsonl")

    obj = {
        "corp_code": corp_code,
        "corp_name": corp_name,
        "rcept_no": rcept_no,
        "year": year,
        "report_nm": report_nm,
        "parsed_business_content": text
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

    print(f"✔ 저장 완료: {out_path}")
    return True


# =========================================
# 5. 회사별 전체 처리 — 병렬 처리로 크게 가속
# =========================================
def process_corp_code(corp_code, corp_name, idx, total_rows):
    print(f"\n==============================")
    print(f" [{idx}/{total_rows}] corp_code={corp_code}, corp_name={corp_name} 처리 시작")
    print(f"==============================")

    reports = get_business_reports(corp_code)
    if len(reports) == 0:
        print(f"사업보고서 없음: {corp_code}")
        return

    print(f" 사업보고서 {len(reports)}개 발견")

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:   # 병렬 처리
        for r in reports:
            rcept_no = r["rcept_no"]
            report_nm = r["report_nm"]

            m = re.search(r"\((\d{4})\.", report_nm)
            year = m.group(1) if m else "UNKNOWN"

            tasks.append(
                executor.submit(
                    save_business_section,
                    corp_code, corp_name, year, rcept_no, report_nm
                )
            )

        for f in as_completed(tasks):
            pass  # 완료된 작업 체크



# =========================================
# 6. 실행부 - DART에서 corp_code 찾은 뒤 대상 기업만 처리 (정확 일치)
# =========================================

def normalize_corp_code(raw):
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(8) if s.isdigit() else None

#  1) 타겟 회사명 (corp_name과 정확히 같은 문자열이어야 함)
target_names = ["현대자동차"]

print("📥 DART 에서 전체 기업목록 불러오는 중...")
corp_list_df = load_corp_list_from_dart(API_KEY)

# 공백 정리
corp_list_df["corp_name"] = corp_list_df["corp_name"].astype(str).str.strip()

target_rows = []

#  2) 기업명 '정확 일치'로 검색
for name in target_names:
    name_clean = name.strip()
    matched = corp_list_df[corp_list_df["corp_name"] == name_clean]

    if matched.empty:
        print(f"'{name_clean}' 과 정확히 일치하는 기업을 corpCode에서 찾지 못했습니다.")
        continue

    print(f"\n '{name_clean}' 정확 일치 검색 결과:")
    print(matched[["corp_code", "corp_name", "stock_code"]])

    target_rows.append(matched[["corp_code", "corp_name"]])

# 목표 기업 없으면 종료
if not target_rows:
    print("\n 대상 기업을 찾을 수 없어 종료합니다.")
else:
    target_df = pd.concat(target_rows).drop_duplicates().reset_index(drop=True)

    print("\n 최종 대상 기업 목록:")
    print(target_df)

    #  3) 사업보고서 내려받기
    total_rows = len(target_df)
    for idx, row in target_df.iterrows():
        corp_code = normalize_corp_code(row["corp_code"])
        corp_name = row["corp_name"]

        if corp_code is None:
            print(f" corp_code 이상: {row}")
            continue

        print(f"\n 실행: corp_code={corp_code}, corp_name={corp_name}")
        process_corp_code(corp_code, corp_name, idx + 1, total_rows)

    print("\n 선택한 기업들 처리 완료!")
