import pandas as pd
import hashlib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap


USER_ID = "STU160"
NUMERIC_ID = "160"


def compute_user_hash(user_id: str) -> str:
    h = hashlib.sha256(user_id.encode()).hexdigest()
    return h[:8].upper()


def load_data():
    books = pd.read_csv("books (1).csv")
    reviews = pd.read_csv("reviews.csv")
    return books, reviews


def find_text_column(df: pd.DataFrame):
    candidates = ["text", "review_text", "review", "content", "body"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("No suitable text column found in reviews.csv")


def find_link_column(books: pd.DataFrame, reviews: pd.DataFrame):
    candidates = ["parent_asin", "asin", "book_id", "title"]
    for c in candidates:
        if c in books.columns and c in reviews.columns:
            return c
    raise ValueError("No common link column between books and reviews.")


def find_rating_column(reviews: pd.DataFrame):
    candidates = ["rating", "stars", "star_rating", "review_rating"]
    for c in candidates:
        if c in reviews.columns:
            return c
    return None


def get_suspicious_books(books: pd.DataFrame):
    if "rating_number" not in books.columns or "average_rating" not in books.columns:
        raise ValueError("Books file must have 'rating_number' and 'average_rating' columns.")
    sus = books[(books["rating_number"] == 1234) & (books["average_rating"] == 5.0)]
    return sus


def get_target_book(books, reviews, user_hash, text_col, link_col):
    suspicious_books = get_suspicious_books(books)

    if suspicious_books.empty:
        raise ValueError("No books found with rating_number=1234 and average_rating=5.0.")

    sus_keys = suspicious_books[link_col].dropna().unique()
    reviews_sus = reviews[reviews[link_col].isin(sus_keys)].copy()

    mask = reviews_sus[text_col].astype(str).str.contains(user_hash, case=False, na=False)
    matched_reviews = reviews_sus[mask]

    if matched_reviews.empty:
        raise ValueError(f"No reviews found containing hash {user_hash}.")

    target_review = matched_reviews.iloc[0]
    key_value = target_review[link_col]

    target_book = suspicious_books[suspicious_books[link_col] == key_value]
    if target_book.empty:
        raise ValueError("Could not match review with any suspicious book.")
    target_book = target_book.iloc[0]

    return target_book, target_review, matched_reviews


def compute_flag1_from_title(title: str) -> str:
    no_space = title.replace(" ", "")
    first8 = no_space[:8]
    h = hashlib.sha256(first8.encode()).hexdigest()
    return h


def label_reviews_for_model(df: pd.DataFrame, text_col: str, rating_col: str | None):
    texts = df[text_col].astype(str).fillna("")

    if rating_col is not None and rating_col in df.columns:
        ratings = df[rating_col].fillna(5)
    else:
        ratings = pd.Series([5] * len(df), index=df.index)

    word_counts = texts.apply(lambda t: len(t.split()))

    superlatives = ["amazing", "awesome", "best", "incredible", "perfect", "must-read", "fantastic", "outstanding"]
    domain_words = ["character", "characters", "plot", "story", "writing", "ending", "chapter", "narrative"]

    def has_any(words, text):
        t = text.lower()
        return any(w in t for w in words)

    suspicious = (ratings == 5) & (word_counts < 40) & texts.apply(lambda t: has_any(superlatives, t))
    genuine = (ratings == 5) & (word_counts >= 40) & texts.apply(lambda t: has_any(domain_words, t))

    labeled_df = df.copy()
    labeled_df["label"] = np.nan
    labeled_df.loc[suspicious, "label"] = 1
    labeled_df.loc[genuine, "label"] = 0

    labeled_df = labeled_df.dropna(subset=["label"])
    labeled_df["label"] = labeled_df["label"].astype(int)

    return labeled_df, texts.loc[labeled_df.index]


def train_model_and_get_shap_words(df: pd.DataFrame, text_col: str, rating_col: str | None, numeric_id: str):
    labeled_df, labeled_texts = label_reviews_for_model(df, text_col, rating_col)
    if labeled_df.empty or labeled_df["label"].nunique() < 2:
        raise ValueError("Not enough labeled data to train model.")

    X_text = labeled_texts.tolist()
    y = labeled_df["label"].values

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    suspicion_scores = model.predict_proba(X)[:, 1]
    labeled_df["suspicion_score"] = suspicion_scores

    genuine_mask = (labeled_df["label"] == 0)
    genuine_df = labeled_df[genuine_mask].copy()
    if genuine_df.empty:
        raise ValueError("No genuine reviews found for SHAP analysis.")

    genuine_df = genuine_df.sort_values("suspicion_score").head(50)

    genuine_indices = genuine_df.index
    X_genuine = X[genuine_indices]

    explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
    shap_values = explainer.shap_values(X_genuine)

    shap_values_mean = shap_values.mean(axis=0)
    feature_names = np.array(vectorizer.get_feature_names_out())

    idx_sorted = np.argsort(shap_values_mean)
    top3_idx = idx_sorted[:3]
    top3_words = feature_names[top3_idx]

    concat = "".join(top3_words) + numeric_id
    flag3_hash = hashlib.sha256(concat.encode()).hexdigest()[:10]
    flag3_string = f"FLAG3{{{flag3_hash}}}"

    return flag3_string, top3_words, concat


def main():
    user_hash = compute_user_hash(USER_ID)
    print("User ID:", USER_ID)
    print("Hash (first 8 chars):", user_hash)

    books, reviews = load_data()
    print("Books shape:", books.shape)
    print("Reviews shape:", reviews.shape)

    text_col = find_text_column(reviews)
    link_col = find_link_column(books, reviews)
    rating_col = find_rating_column(reviews)
    print("Using text column:", text_col)
    print("Using link column:", link_col)
    print("Using rating column:", rating_col)

    target_book, target_review, all_matched_reviews = get_target_book(
        books, reviews, user_hash, text_col, link_col
    )

    print("\n=== TARGET BOOK ===")
    print("Title:", target_book["title"])
    if "subtitle" in target_book:
        print("Subtitle:", target_book["subtitle"])
    print("average_rating:", target_book["average_rating"])
    print("rating_number:", target_book["rating_number"])

    flag1 = compute_flag1_from_title(str(target_book["title"]))
    print("\nFLAG1 =", flag1)

    flag2 = f"FLAG2{{{user_hash}}}"
    print("FLAG2 =", flag2)

    book_key_value = target_book[link_col]
    book_reviews = reviews[reviews[link_col] == book_key_value].copy()

    flag3, top_words, concat_string = train_model_and_get_shap_words(
        book_reviews, text_col, rating_col, NUMERIC_ID
    )

    print("\nTop 3 words that reduce suspicion (from SHAP):", list(top_words))
    print("Concatenated string for FLAG3:", concat_string)
    print("FLAG3 =", flag3)

    print("\n\n=== FINAL FLAGS (copy these into flags.txt) ===")
    print("FLAG1 =", flag1)
    print("FLAG2 =", flag2)
    print("FLAG3 =", flag3)


if __name__ == "__main__":
    main()
