import os
import json
import glob
import argparse
import pandas as pd
from tqdm import tqdm

# =========================
# JSONL 로드
# =========================
def load_all_jsonl(jsonl_root, text_key, year_key):
    paths = glob.glob(os.path.join(jsonl_root, "**/*.jsonl"), recursive=True)
    print(f"[INFO] Found jsonl files: {len(paths)}")

    rows = []
    for p in tqdm(paths, desc="Reading jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj["_source_file"] = os.path.basename(p)
                    rows.append(obj)
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(rows)

    if year_key not in df.columns:
        raise KeyError(f"{year_key} not found in columns")
    if text_key not in df.columns:
        raise KeyError(f"{text_key} not found in columns")

    df = df.dropna(subset=[year_key, text_key]).copy()
    df[year_key] = df[year_key].astype(int)
    df["text"] = df[text_key].astype(str)

    return df


# =========================
# 형태소 분석기
# =========================
def load_analyzer():
    try:
        from konlpy.tag import Mecab
        return Mecab()
    except:
        from konlpy.tag import Okt
        return Okt()


def count_nouns(analyzer, text):
    try:
        return len(analyzer.nouns(text))
    except:
        return 0


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_dir",
        default="final_jsonl_filled_v2/final_jsonl_filled_v2",
        help="jsonl root directory"
    )
    parser.add_argument("--text_key", default="parsed_business_content")
    parser.add_argument("--year_key", default="year")
    parser.add_argument("--use_noun", action="store_true")
    parser.add_argument(
        "--sample_per_year",
        type=int,
        default=0,
        help="0이면 전체, >0이면 연도별 샘플링"
    )

    args = parser.parse_args()

    # 1) load
    df = load_all_jsonl(args.jsonl_dir, args.text_key, args.year_key)

    # 2) 샘플링 (선택)
    if args.sample_per_year > 0:
        df = (
            df.groupby(args.year_key, group_keys=False)
              .apply(lambda x: x.sample(min(len(x), args.sample_per_year), random_state=42))
              .reset_index(drop=True)
        )
        print(f"[INFO] Sampled rows: {len(df)}")

    # 3) 길이 계산
    df["char_len"] = df["text"].apply(len)
    df["word_len"] = df["text"].apply(lambda x: len(x.split()))

    # 4) 연도별 문장 길이
    year_char_stats = (
        df.groupby(args.year_key)["char_len"]
          .agg(count="count", mean="mean", median="median", min="min", max="max")
          .reset_index()
          .sort_values(args.year_key)
    )

    year_word_stats = (
        df.groupby(args.year_key)["word_len"]
          .agg(mean="mean", median="median", min="min", max="max")
          .reset_index()
          .sort_values(args.year_key)
    )

    # 5) 명사 개수 (선택)
    year_noun_stats = None
    if args.use_noun:
        analyzer = load_analyzer()
        tqdm.pandas(desc="Counting nouns")
        df["noun_cnt"] = df["text"].progress_apply(
            lambda x: count_nouns(analyzer, x)
        )

        year_noun_stats = (
            df.groupby(args.year_key)["noun_cnt"]
              .agg(mean="mean", median="median", min="min", max="max", sum="sum")
              .reset_index()
              .sort_values(args.year_key)
        )

    # 6) 저장
    os.makedirs("eda_outputs", exist_ok=True)
    year_char_stats.to_csv("eda_outputs/year_char_stats.csv", index=False, encoding="utf-8-sig")
    year_word_stats.to_csv("eda_outputs/year_word_stats.csv", index=False, encoding="utf-8-sig")

    if year_noun_stats is not None:
        year_noun_stats.to_csv(
            "eda_outputs/year_noun_stats.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # 7) 출력
    print("\n=== Yearly Char Length ===")
    print(year_char_stats.head(10).to_string(index=False))

    print("\n=== Yearly Word Length ===")
    print(year_word_stats.head(10).to_string(index=False))

    if year_noun_stats is not None:
        print("\n=== Yearly Noun Count ===")
        print(year_noun_stats.head(10).to_string(index=False))

    print("\n[DONE] Results saved to ./eda_outputs/")


if __name__ == "__main__":
    main()
