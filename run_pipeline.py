from preprocessing.run_preprocessing_pipeline import run_preprocessing_pipeline
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC

def main():
    short_term_data, long_term_data, news2idx, category2idx = run_preprocessing_pipeline()

    num_news = len(news2idx)
    num_categories = len(category2idx)

    print("Num news:", num_news)
    print("Num categories:", num_categories)

    short_model = ShortTermLTC(
        num_news=num_news,
        num_categories=num_categories,
        hidden_dim=64
    )

    long_model = LongTermLTC(
        num_news=num_news,
        num_categories=num_categories,
        hidden_dim=64
    )

    user_id = next(iter(short_term_data))

    st_vec, _, _ = short_model(short_term_data[user_id])
    lt_vec, _, _ = long_model(long_term_data[user_id])

    print("ST vector shape:", st_vec.shape)
    print("LT vector shape:", lt_vec.shape)

if __name__ == "__main__":
    main()
