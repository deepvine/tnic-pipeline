# TNIC-pipeline
프로젝트 구조
```
tnic-pipeline/
 ├── src/
 │   ├── preprocess.py
 │   ├── vectorize.py
 │   ├── similarity.py
 │   └── network.py
 ├── data/
 │   └── 10k_jsonl/
 ├── output/
 ├── notebooks/
 └── README.md

```

## How to use
```
python -m src.main \
  --input data/korean_10k_business_desc.jsonl \
  --output_dir output/tnic_korean \
  --bert_model klue/bert-base \
  --device cuda:0 \
  --threshold 0.2132 \
  --max_df_ratio 0.25
```
아래와 같이 결과가 생성됩니다.
```
similarity_YYYY.npy
similarity_centered_YYYY.npy
df_counts_YYYY.json
high_freq_tokens_YYYY.json
tnic_edges_YYYY.jsonl
```