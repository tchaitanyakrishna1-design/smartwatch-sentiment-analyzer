import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import pipeline
import joblib
import random

print("Script started...")

# ==== Small built-in smartwatch review dataset ====
data = [
    ("The smartwatch battery life is excellent and lasts two days.", "positive"),
    ("The screen is bright and very responsive.", "positive"),
    ("Love the design and smooth performance of this watch.", "positive"),
    ("Notifications work perfectly and the app is easy to use.", "positive"),
    ("The strap is uncomfortable and feels cheap.", "negative"),
    ("Keeps disconnecting from my phone, very frustrating.", "negative"),
    ("Step tracking is completely inaccurate and useless.", "negative"),
    ("Charging is slow and the watch overheats sometimes.", "negative"),
    ("Heart rate monitor seems reliable and accurate.", "positive"),
    ("Touch response is laggy and the UI is confusing.", "negative"),
]

random.shuffle(data)
texts = [t for t, s in data]
labels = [s for t, s in data]

df = pd.DataFrame({"text": texts, "sentiment": labels})

X = df["text"].values
y = df["sentiment"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ===== Classical ML: TF-IDF + Logistic Regression =====
tfidf = TfidfVectorizer(max_features=1000)
X_train_vec = tfidf.fit_transform(X_train)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

X_test_vec = tfidf.transform(X_test)
y_pred_ml = clf.predict(X_test_vec)
acc_ml = accuracy_score(y_test, y_pred_ml)
print("Classical Model Accuracy:", acc_ml)

joblib.dump(tfidf, "tfidf.joblib")
joblib.dump(clf, "logreg.joblib")

# ===== Transformer: DistilBERT Sentiment Pipeline =====
nlp = pipeline("sentiment-analysis")

def pred_trf(texts):
    results = nlp(list(texts))
    return ["positive" if r["label"].upper().startswith("POS") else "negative" for r in results]

y_pred_trf = pred_trf(X_test)
acc_trf = accuracy_score(y_test, y_pred_trf)
print("Transformer Accuracy:", acc_trf)

print("\n=== Final Comparison ===")
print(f"Classical:   {acc_ml*100:.2f}%")
print(f"Transformer: {acc_trf*100:.2f}%")
