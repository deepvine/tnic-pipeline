import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 경로
# =========================
DATA_DIR = "eda_outputs"
FIG_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# =========================
# CSV 로드
# =========================
char_df = pd.read_csv(os.path.join(DATA_DIR, "year_char_stats.csv"))
word_df = pd.read_csv(os.path.join(DATA_DIR, "year_word_stats.csv"))

# 안전: year 정렬
char_df["year"] = pd.to_numeric(char_df["year"], errors="coerce")
word_df["year"] = pd.to_numeric(word_df["year"], errors="coerce")
char_df = char_df.sort_values("year")
word_df = word_df.sort_values("year")

# =========================
# 유틸
# =========================
def save_show(fig_name: str):
    out = os.path.join(FIG_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print("saved:", out)

def minmax(x):
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x * 0

# =========================
# 0) 체크: 컬럼 존재 확인 (count는 char 기준만!)
# =========================
required_char_cols = {"year", "count"}
required_len_cols = {"year", "mean", "max"}

missing_char = required_char_cols - set(char_df.columns)
missing_char_len = required_len_cols - set(char_df.columns)
missing_word_len = required_len_cols - set(word_df.columns)

if missing_char:
    print("⚠️ year_char_stats.csv missing:", missing_char)
if missing_char_len:
    print("⚠️ year_char_stats.csv missing (length cols):", missing_char_len)
if missing_word_len:
    print("⚠️ year_word_stats.csv missing (length cols):", missing_word_len)

# ============================================================
# 1) 연도별 사업보고서 수 시각화 (count)  
# ============================================================
plt.figure(figsize=(10, 4))
plt.bar(char_df["year"], char_df["count"])
plt.title("Number of Business Reports by Year")
plt.xlabel("Year")
plt.ylabel("# of Reports")
plt.grid(axis="y", alpha=0.3)
save_show("01_year_report_count.png")

# ============================================================
# 2) 연도별 사업보고서 길이 시각화 
#   - (2-1) Characters: 평균 & 최대
#   - (2-2) Words: 평균 & 최대
# ============================================================

# (2-1) Characters
plt.figure(figsize=(10, 4))
plt.plot(char_df["year"], char_df["mean"], marker="o", label="Mean (chars)")
plt.plot(char_df["year"], char_df["max"], marker="o", label="Max (chars)")
plt.title("Report Length by Year (Characters): Mean & Max")
plt.xlabel("Year")
plt.ylabel("# of Characters")
plt.legend()
plt.grid(True, alpha=0.3)
save_show("02_year_char_mean_max.png")

# (2-2) Words
plt.figure(figsize=(10, 4))
plt.plot(word_df["year"], word_df["mean"], marker="o", label="Mean (words)")
plt.plot(word_df["year"], word_df["max"], marker="o", label="Max (words)")
plt.title("Report Length by Year (Words): Mean & Max")
plt.xlabel("Year")
plt.ylabel("# of Words")
plt.legend()
plt.grid(True, alpha=0.3)
save_show("03_year_word_mean_max.png")

# ============================================================
# (선택) 길이 비교용: Char vs Word 정규화 평균 트렌드
# ============================================================
df = (
    char_df[["year", "mean", "max"]].rename(columns={"mean": "char_mean", "max": "char_max"})
    .merge(
        word_df[["year", "mean", "max"]].rename(columns={"mean": "word_mean", "max": "word_max"}),
        on="year",
        how="inner",
    )
)

df["char_mean_norm"] = minmax(df["char_mean"])
df["word_mean_norm"] = minmax(df["word_mean"])

plt.figure(figsize=(10, 4))
plt.plot(df["year"], df["char_mean_norm"], marker="o", label="Chars mean (norm)")
plt.plot(df["year"], df["word_mean_norm"], marker="o", label="Words mean (norm)")
plt.title("Yearly Trend Comparison (Normalized Means)")
plt.xlabel("Year")
plt.ylabel("Normalized value (0–1)")
plt.legend()
plt.grid(True, alpha=0.3)
save_show("04_year_mean_trend_compare_norm.png")

print("\n All figures saved to:", FIG_DIR)
