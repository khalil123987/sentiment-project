import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('movie_reviews', quiet=True)

# Load in data
documents = []
labels = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(categories=category):
        text = " ".join(movie_reviews.words(fileid))
        documents.append(text)
        labels.append(1 if category == 'pos' else 0)

# Train/Test split 80/20
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    documents, labels, test_size=0.2, random_state=42
)

# TF-IDF vektorisering
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Train and evaluate the baseline model
print("Feature Set 1 - Baseline (Unigram TF-IDF)")
print("-" * 50)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes":         MultinomialNB(),
    "Linear SVM":          LinearSVC(max_iter=2000, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))