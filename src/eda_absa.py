import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception as exc:
    sns = None
    print(f"Warning: seaborn import failed: {exc}")


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_CHARTS = Path(__file__).resolve().parents[1] / "output" / "charts"
OUTPUT_REPORTS = Path(__file__).resolve().parents[1] / "output" / "reports"

ASPECTS = [
    "GENERAL",
    "CAMERA",
    "BATTERY",
    "SCREEN",
    "PERFORMANCE",
    "DESIGN",
    "PRICE",
    "SER&ACC",
    "FEATURES",
    "STORAGE",
]
SENTIMENTS = ["Positive", "Negative", "Neutral"]


def load_data():
    """Đọc và gộp dữ liệu từ train/dev/test."""
    files = {
        "train": DATA_ROOT / "train.csv",
        "dev": DATA_ROOT / "dev.csv",
        "test": DATA_ROOT / "test.csv",
    }

    frames = {}
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="warn",
        )
        # Normalize tên cột nếu cần
        if "rating" not in df.columns and "n_star" in df.columns:
            df = df.rename(columns={"n_star": "rating"})
        if "date" not in df.columns and "date_time" in df.columns:
            df = df.rename(columns={"date_time": "date"})

        df = df.rename(columns={col: col.strip() for col in df.columns})
        df = df[[col for col in ["comment", "rating", "date", "label"] if col in df.columns]]
        df["split"] = split
        frames[split] = df

    df_all = pd.concat(frames.values(), ignore_index=True)
    return frames["train"], frames["dev"], frames["test"], df_all


def extract_labels(label_text):
    """Tách các cặp Aspect#Sentiment từ một chuỗi label."""
    if not isinstance(label_text, str) or pd.isna(label_text):
        return []

    labels = re.findall(r"\{([^#]+)#([^}]+)\}", label_text)
    return [(aspect.strip(), sentiment.strip()) for aspect, sentiment in labels]


def dataset_statistics(train_df, dev_df, test_df, all_df):
    """Tính thống kê cơ bản cho tập dữ liệu."""
    train_count = len(train_df)
    dev_count = len(dev_df)
    test_count = len(test_df)
    total_count = len(all_df)

    percentages = {
        "train_pct": train_count / total_count * 100 if total_count else 0,
        "dev_pct": dev_count / total_count * 100 if total_count else 0,
        "test_pct": test_count / total_count * 100 if total_count else 0,
    }

    stats = {
        "train_count": train_count,
        "dev_count": dev_count,
        "test_count": test_count,
        "total_count": total_count,
        "train_pct": percentages["train_pct"],
        "dev_pct": percentages["dev_pct"],
        "test_pct": percentages["test_pct"],
    }
    return pd.DataFrame([stats])


def aspect_statistics(df_all):
    """Đếm số lần xuất hiện của từng Aspect và tạo ma trận Aspect×Sentiment."""
    extracted = df_all["label"].apply(extract_labels)
    aspect_list = []
    sentiment_list = []
    matrix_rows = []

    for labels in extracted:
        for aspect, sentiment in labels:
            aspect_list.append(aspect)
            sentiment_list.append(sentiment)
            matrix_rows.append({"aspect": aspect, "sentiment": sentiment})

    aspect_counts = pd.Series(aspect_list).value_counts()
    all_aspects = [aspect for aspect in ASPECTS if aspect in aspect_counts.index]
    extra_aspects = [aspect for aspect in aspect_counts.index if aspect not in ASPECTS]
    ordered_aspects = all_aspects + extra_aspects
    aspect_counts = aspect_counts.reindex(ordered_aspects, fill_value=0)
    aspect_df = aspect_counts.reset_index()
    aspect_df.columns = ["aspect", "count"]
    

    if matrix_rows:
        aspect_sentiment = pd.DataFrame(matrix_rows)
        matrix = pd.crosstab(aspect_sentiment["aspect"], aspect_sentiment["sentiment"]).reindex(index=ordered_aspects, columns=SENTIMENTS, fill_value=0)
    else:
        matrix = pd.DataFrame(0, index=ordered_aspects, columns=SENTIMENTS)

    return aspect_df, matrix


def sentiment_statistics(df_all):
    """Đếm số lần xuất hiện của từng Sentiment."""
    extracted = df_all["label"].apply(extract_labels)
    sentiment_list = [sentiment for labels in extracted for _, sentiment in labels]
    sentiment_counts = pd.Series(sentiment_list).value_counts().reindex(SENTIMENTS, fill_value=0)
    sentiment_df = sentiment_counts.reset_index()
    sentiment_df.columns = ["sentiment", "count"]
    return sentiment_df


def comment_length_statistics(df_all):
    """Tính các thống kê độ dài bình luận theo số từ."""
    lengths = df_all["comment"].fillna("").astype(str).apply(lambda text: len(text.split()))
    stats = {
        "mean_words": lengths.mean(),
        "min_words": lengths.min(),
        "max_words": lengths.max(),
        "std_words": lengths.std(),
    }
    stats_df = pd.DataFrame([stats])
    return lengths, stats_df


def imbalance_analysis(count_df, label_column):
    """Phân tích mất cân bằng dữ liệu cho cột nhãn có chỉ số count."""
    counts = count_df.set_index(label_column)["count"].sort_values(ascending=False)
    if counts.empty:
        return {
            "max_class": None,
            "min_class": None,
            "max_count": 0,
            "min_count": 0,
            "ratio": 0,
            "comment": "Không có dữ liệu để phân tích mất cân bằng.",
        }

    max_class = counts.idxmax()
    min_class = counts.idxmin()
    max_count = int(counts.max())
    min_count = int(counts.min())
    ratio = max_count / min_count if min_count else float("inf")

    if ratio >= 10:
        comment = "Dữ liệu mất cân bằng nghiêm trọng: một số lớp có số lượng lớn gấp 10 lần lớp khác."
    elif ratio >= 5:
        comment = "Dữ liệu mất cân bằng rõ ràng: các lớp lớn hơn nhiều so với lớp nhỏ nhất."
    elif ratio >= 2:
        comment = "Dữ liệu có mất cân bằng vừa phải."
    else:
        comment = "Dữ liệu tương đối cân bằng giữa các lớp."

    return {
        "max_class": max_class,
        "min_class": min_class,
        "max_count": max_count,
        "min_count": min_count,
        "ratio": ratio,
        "comment": comment,
    }


def plot_charts(dataset_stats, aspect_df, sentiment_df, matrix, lengths):
    """Vẽ và lưu biểu đồ vào thư mục output/charts."""
    OUTPUT_CHARTS.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("ggplot")

    # Safer palettes khi seaborn không khả dụng
    if sns is not None:
        bar_palette = None
        pastel_colors = sns.color_palette("pastel")
        aspect_colors = sns.color_palette("tab10", len(aspect_df))
        sentiment_colors = sns.color_palette("Set2", len(sentiment_df))
    else:
        bar_palette = None
        pastel_colors = plt.cm.Pastel1(np.linspace(0, 1, 3))
        aspect_colors = plt.cm.tab10(np.linspace(0, 1, len(aspect_df)))
        sentiment_colors = plt.cm.Set2(np.linspace(0, 1, len(sentiment_df)))

    # 1. Bar Chart: Distribution of Train / Dev / Test
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.barplot(x=["train", "dev", "test"], y=dataset_stats[["train_count", "dev_count", "test_count"]].iloc[0].values)
    else:
        plt.bar(["train", "dev", "test"], dataset_stats[["train_count", "dev_count", "test_count"]].iloc[0].values, color=bar_palette)
    plt.title("Distribution of Train / Dev / Test")
    plt.ylabel("Number of Samples")
    plt.xlabel("Dataset")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "dataset_distribution_bar.png")
    plt.close()

    # 2. Pie Chart: Train / Dev / Test Ratio
    plt.figure(figsize=(7, 7))
    plt.pie(
        dataset_stats[["train_count", "dev_count", "test_count"]].iloc[0].values,
        labels=["train", "dev", "test"],
        autopct="%1.1f%%",
        startangle=140,
        colors=pastel_colors,
    )
    plt.title("Train / Dev / Test Ratio")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "dataset_ratio_pie.png")
    plt.close()

    # 3. Bar Chart: Aspect Distribution
    plt.figure(figsize=(12, 6))
    if sns is not None:
        sns.barplot(x="aspect", y="count", data=aspect_df)
    else:
        plt.bar(aspect_df["aspect"], aspect_df["count"], color=aspect_colors)
    plt.title("Aspect Distribution")
    plt.ylabel("Count")
    plt.xlabel("Aspect")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "aspect_distribution_bar.png")
    plt.close()

    # 4. Pie Chart: Aspect Distribution
    plt.figure(figsize=(8, 8))
    plt.pie(
        aspect_df["count"],
        labels=aspect_df["aspect"],
        autopct="%1.1f%%",
        startangle=140,
        colors=aspect_colors,
    )
    plt.title("Aspect Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "aspect_distribution_pie.png")
    plt.close()

    # 5. Bar Chart: Sentiment Distribution
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.barplot(x="sentiment", y="count", data=sentiment_df)
    else:
        plt.bar(sentiment_df["sentiment"], sentiment_df["count"], color=sentiment_colors)
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "sentiment_distribution_bar.png")
    plt.close()

    # 6. Pie Chart: Sentiment Distribution
    plt.figure(figsize=(7, 7))
    plt.pie(
        sentiment_df["count"],
        labels=sentiment_df["sentiment"],
        autopct="%1.1f%%",
        startangle=140,
        colors=sentiment_colors,
    )
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "sentiment_distribution_pie.png")
    plt.close()

    # 7. Stacked Bar Chart: Aspect vs Sentiment
    plt.figure(figsize=(12, 8))
    matrix_plot = matrix.copy()
    matrix_plot.index.name = "aspect"
    matrix_plot.columns.name = "sentiment"
    matrix_plot.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="tab20")
    plt.title("Aspect vs Sentiment")
    plt.ylabel("Count")
    plt.xlabel("Aspect")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "aspect_sentiment_stacked_bar.png")
    plt.close()

    # 8. Histogram: Comment Length Distribution
    plt.figure(figsize=(10, 6))
    if sns is not None:
        sns.histplot(lengths, bins=30, kde=True, color="steelblue")
    else:
        plt.hist(lengths, bins=30, color="steelblue", edgecolor="black", alpha=0.75)
    plt.title("Comment Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHARTS / "comment_length_histogram.png")
    plt.close()


def save_reports(dataset_stats, aspect_df, sentiment_df, matrix, comment_stats_df):
    """Lưu các báo cáo thống kê thành file CSV."""
    OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)
    dataset_stats.to_csv(OUTPUT_REPORTS / "dataset_statistics.csv", index=False)
    aspect_df.to_csv(OUTPUT_REPORTS / "aspect_distribution.csv", index=False)
    sentiment_df.to_csv(OUTPUT_REPORTS / "sentiment_distribution.csv", index=False)
    matrix.to_csv(OUTPUT_REPORTS / "aspect_sentiment_matrix.csv")
    comment_stats_df.to_csv(OUTPUT_REPORTS / "comment_length_statistics.csv", index=False)


def print_report(dataset_stats, aspect_df, sentiment_df, matrix, comment_stats_df, aspect_imbalance, sentiment_imbalance):
    """In các kết quả ra terminal với định dạng rõ ràng."""
    print("\n" + "=" * 80)
    print("1. Dataset statistics")
    print("=" * 80)
    print(dataset_stats.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. Aspect distribution")
    print("=" * 80)
    print(aspect_df.to_string(index=False))
    print("\nAspect imbalance:")
    print(f"  - Most frequent aspect: {aspect_imbalance['max_class']} ({aspect_imbalance['max_count']})")
    print(f"  - Least frequent aspect: {aspect_imbalance['min_class']} ({aspect_imbalance['min_count']})")
    print(f"  - Ratio max/min: {aspect_imbalance['ratio']:.2f}")
    print(f"  - Comment: {aspect_imbalance['comment']}")

    print("\n" + "=" * 80)
    print("3. Sentiment distribution")
    print("=" * 80)
    print(sentiment_df.to_string(index=False))
    print("\nSentiment imbalance:")
    print(f"  - Most frequent sentiment: {sentiment_imbalance['max_class']} ({sentiment_imbalance['max_count']})")
    print(f"  - Least frequent sentiment: {sentiment_imbalance['min_class']} ({sentiment_imbalance['min_count']})")
    print(f"  - Ratio max/min: {sentiment_imbalance['ratio']:.2f}")
    print(f"  - Comment: {sentiment_imbalance['comment']}")

    print("\n" + "=" * 80)
    print("4. Aspect × Sentiment matrix")
    print("=" * 80)
    print(matrix.to_string())

    print("\n" + "=" * 80)
    print("5. Comment length statistics")
    print("=" * 80)
    print(comment_stats_df.to_string(index=False))
    print("\nKết quả biểu đồ được lưu trong:")
    print(f"  - {OUTPUT_CHARTS}")
    print("Các báo cáo CSV được lưu trong:")
    print(f"  - {OUTPUT_REPORTS}")
    print("=" * 80 + "\n")


def main():
    """Thực thi toàn bộ quy trình EDA."""
    print("Bắt đầu EDA cho bộ dữ liệu UIT-ViSFD...\n")
    train_df, dev_df, test_df, all_df = load_data()

    dataset_stats = dataset_statistics(train_df, dev_df, test_df, all_df)
    aspect_df, aspect_sentiment_matrix = aspect_statistics(all_df)
    sentiment_df = sentiment_statistics(all_df)
    lengths, comment_stats_df = comment_length_statistics(all_df)

    aspect_imbalance = imbalance_analysis(aspect_df, "aspect")
    sentiment_imbalance = imbalance_analysis(sentiment_df, "sentiment")

    save_reports(dataset_stats, aspect_df, sentiment_df, aspect_sentiment_matrix, comment_stats_df)
    plot_charts(dataset_stats, aspect_df, sentiment_df, aspect_sentiment_matrix, lengths)
    print_report(dataset_stats, aspect_df, sentiment_df, aspect_sentiment_matrix, comment_stats_df, aspect_imbalance, sentiment_imbalance)
    print("EDA hoàn tất.")


if __name__ == "__main__":
    main()
