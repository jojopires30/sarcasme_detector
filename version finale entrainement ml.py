# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:50:26 2025

@author: theog
"""

# train_model.py
# ----------------
# Script d'entraînement du modèle de sarcasme

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Fonction de nettoyage de texte
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 1) Chargement et nettoyage des données
df = pd.read_csv(r"C:\Users\theog\Desktop\projet_ml_ecole\sarcasm_clean.csv")
df["comment"] = df["comment"].apply(clean_text)

# 2) Séparation en train/test
X = df["comment"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Vectorisation TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000000,
    ngram_range=(1,10),
    min_df=3,
    max_df=0.90
)
X_train_vect = vectorizer.fit_transform(X_train)

# 4) Entraînement du modèle
model = LogisticRegression(
    max_iter=1000,
    solver='liblinear',
    penalty='l2',
    C=1.0
)
model.fit(X_train_vect, y_train)

# 5) Sauvegarde des objets
joblib.dump(vectorizer, r"C:\Users\theog\Desktop\projet_ml_ecole\vectorizer.pkl")
joblib.dump(model,      r"C:\Users\theog\Desktop\projet_ml_ecole\logistic_model.pkl")

print("✅ Entraînement terminé et artefacts sauvegardés.")

