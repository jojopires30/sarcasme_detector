# train_model.py
import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

nltk.download('stopwords')

# Nettoyage
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\!\?\"\'\-]", "", text)
    return text

# Chargement et préparation
df = pd.read_csv(r"/Users/cocojojo/Desktop/train-balanced-sarcasm.csv")[['label', 'comment']].dropna()
df['clean_comment'] = df['comment'].apply(clean_text)
X = df['clean_comment']
y = df['label'].astype(int)

# Séparation entraînement / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words=stopwords.words("english"),
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r'(?u)\b\w[\w\'\-]*\b'
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

# Entraînement
pipeline.fit(X_train, y_train)

# Évaluation
y_pred = pipeline.predict(X_test)
print(" Rapport de classification sur les données de test :\n")
print(classification_report(y_test, y_pred, target_names=["Pas sarcastique", "Sarcastique"]))

# Sauvegarde
joblib.dump(pipeline, "sarcasm_model.joblib")
print("\nModèle entraîné et sauvegardé sous 'sarcasm_model.joblib'")
