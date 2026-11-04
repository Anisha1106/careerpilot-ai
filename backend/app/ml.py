import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(value: str) -> str:
    value = re.sub(r"https?://\S+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()

def text_similarity(left: str, right: str) -> float:
    left, right = clean_text(left), clean_text(right)
    if not left or not right:
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    try:
        matrix = vectorizer.fit_transform([left, right])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0

def rank_texts(query: str, documents: list[str]) -> list[tuple[int, float]]:
    query = clean_text(query)
    clean_documents = [clean_text(document) for document in documents]
    if not query or not any(clean_documents):
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    try:
        matrix = vectorizer.fit_transform([query, *clean_documents])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except ValueError:
        return []
    ranked = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)
    return [(index, score) for index, score in ranked if score > 0]
