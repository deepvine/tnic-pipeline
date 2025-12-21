# TNIC-pipeline

기업 사업보고서 텍스트를 기반으로 **TNIC (Text-based Network Industry Classification)** 유사도 네트워크를 생성하는 파이프라인입니다.
TF-IDF, SBERT, OpenAI embedding 등 다양한 벡터라이저를 지원하며, 연도별 기업 간 유사도 행렬과 네트워크 엣지를 산출합니다.

---

## 프로젝트 구조

```
tnic-pipeline/
 ├── src/
 │   ├── preprocess.py              # JSONL 로딩, 토큰화, 정제
 │   ├── tfidf_encoder.py            # TF-IDF 벡터라이저
 │   ├── sbert_encoder.py            # SBERT / Kiwi-SBERT 인코더
 │   ├── chunked_openai_encoder.py   # OpenAI 임베딩 (chunk 기반)
 │   ├── vectorize.py                # 연도별 텍스트 → 벡터 변환
 │   ├── similarity.py               # cosine similarity, centering
 │   ├── main.py                     # 메인 파이프라인 실행
 │   └── network.py                  # TNIC 엣지 생성 로직
 ├── evaluation/                     # 결과 평가 스크립트
 ├── data/
 │   └── 10k_jsonl/                  # 입력 데이터(JSONL)
 ├── output/                         # 미리 생성된 2024년 분석 결과
 │   ├── tfidf/
 │   │   └── tnic_edges_2024.jsonl
 │   ├── sbert/
 │   │   └── tnic_edges_2024.jsonl
 │   └── openai/
 │       └── tnic_edges_2024.jsonl
 ├── visualize_*.ipynb               # 시각화 / 분석 노트북
 └── README.md
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

CUDA 사용 시 PyTorch CUDA 버전이 설치되어 있어야 합니다.

---

## 기본 실행 예시

```bash
python -m src.main \
  --backend 백엔드모델선택(tfidf|sbert|openai) \
  --input_dir 사업보고서자료 \
  --output_dir 출력위치 \
  --threshold 유사도임계값 \
  --years 분석대상년도
```

```bash
# TFIDF 기반 기본 실행 예시
python -m src.main \
  --backend tfidf \
  --input_dir data/final_jsonl_filled_v2 \
  --output_dir output/tfidf \
  --threshold 0.2132 \
  --max_df_ratio 0.25 \
  --tfidf_min_df 2 \
  --tfidf_max_df 0.9 \
  --years 2024
```

### 생성 결과

실행이 완료되면 연도별로 다음 파일들이 생성됩니다.

```
similarity_YYYY.npy              # cosine similarity 행렬
similarity_centered_YYYY.npy     # median-centered similarity
df_counts_YYYY.json             # token document frequency
tnic_edges_YYYY.jsonl           # 최종 TNIC 네트워크 엣지
```

---

## Backend별 실행 방법

### 1. TF-IDF 기반

```bash
python -m src.main \
  --backend tfidf \
  --input_dir data/final_jsonl_filled_v2 \
  --output_dir output/tfidf \
  --threshold 0.2132 \
  --max_df_ratio 0.25 \
  --tfidf_min_df 2 \
  --tfidf_max_df 0.9 \
  --years 2024
```

* `tfidf_min_df` : 최소 document frequency
* `tfidf_max_df` : 최대 document frequency 비율

---

### 2. SBERT 기반 (Document-level)

```bash
python -m src.main \
  --backend sbert \
  --input_dir data/final_jsonl_filled_v2 \
  --output_dir output/bert_document2 \
  --aggregation mean \
  --bert_device cuda \
  --threshold 0.2 \
  --years 2024
```

* `aggregation` : chunk embedding을 문서 단위로 집계하는 방식 (mean / max 등)
* `bert_device` : cuda 또는 cpu

---

### 3. OpenAI Embedding 기반

```bash
python -m src.main \
  --backend openai \
  --input_dir data/final_jsonl_filled_v2 \
  --output_dir output/openai \
  --threshold 0.2 \
  --years 2024
```

* OpenAI API Key가 .env 파일에 환경변수로 설정되어 있어야 합니다.

---

## 연도(years) 옵션 설명

* `--years` 옵션을 **지정하지 않으면**, 입력 데이터에 포함된 **전체 연도에 대해 모두 실행**됩니다.
* 특정 연도만 실행하고 싶을 경우 아래와 같이 지정합니다.

```bash
--years 2024
```

또는 구간 형태로도 평가 시 사용 가능합니다.

```bash
--years 2004-2007
```
---

## 평가만 실행하기

이미 생성된 output 결과를 기반으로 **평가만** 수행할 수 있습니다.

```bash
python -m src.evaluate \
  --output_dir output/openai \
  --backend openai \
  --threshold 0.2 \
  --years "2024-2024"
```

* similarity threshold 기준 통과 엣지 통계
* 연도별 네트워크 특성 분석

---

## 시각화 및 분석

결과 네트워크 및 통계 분석은 아래 노트북을 통해 확인할 수 있습니다.

```
visualize_tfidf.ipynb
visualize_sbert.ipynb
visualize_openai.ipynb
```

각 backend 결과에 맞는 `visualize_*.ipynb` 노트북을 실행하면

* 네트워크 그래프
* 유사도 분포
* 산업군/섹터별 연결 구조

등을 시각적으로 분석할 수 있습니다.

---

## Notes

* threshold 값은 네트워크 밀도에 큰 영향을 미치므로 실험적으로 조정하는 것을 권장합니다.
* 대규모 연도/기업 수 실행 시 메모리 사용량에 주의하세요.
* SBERT / OpenAI backend는 GPU 또는 API rate limit 영향을 받을 수 있습니다.

