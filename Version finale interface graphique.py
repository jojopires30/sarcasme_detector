# interface_app_ctk.py
import customtkinter as ctk
import re
import joblib

# --- Configuration CustomTkinter ---
ctk.set_appearance_mode("dark")      # Options: "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue", etc.

# --- Chargement des artefacts ---
vectorizer = joblib.load(r"C:\Users\theog\Desktop\projet_ml_ecole\vectorizer.pkl")
model      = joblib.load(r"C:\Users\theog\Desktop\projet_ml_ecole\logistic_model.pkl")

# --- Fonction de nettoyage du texte ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)              # Supprime les URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)          # Garde lettres et espaces
    text = re.sub(r"\s+", " ", text).strip()         # Supprime espaces multiples
    return text

# --- Fonction de prédiction et mise à jour de l'interface ---
def predict():
    user_input = text_input.get("0.0", "end").strip()
    if not user_input:
        result_label.configure(text="⚠️ Veuillez entrer une phrase.")
        return
    cleaned = clean_text(user_input)
    # Vectorisation + prédiction
    vec  = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    # Préparation du libellé
    label      = "😏 Sarcastique" if pred == 1 else "🙂 Non sarcastique"
    confidence = round(prob * 100, 2)
    result_label.configure(text=f"{label} (Confiance : {confidence} %)")

# --- Création de la fenêtre principale ---
app = ctk.CTk()
app.title("🧠 Détecteur de Sarcasme")
app.geometry("600x400")

# --- Widgets ---
title = ctk.CTkLabel(
    app,
    text="Détection de Sarcasme",
    font=ctk.CTkFont(size=24, weight="bold")
)
title.pack(pady=(20, 10))

text_input = ctk.CTkTextbox(
    app,
    width=520,
    height=120,
    font=("Helvetica", 14)
)
text_input.pack(pady=10)

predict_button = ctk.CTkButton(
    app,
    text="Prédire",
    command=predict,
    font=ctk.CTkFont(size=16, weight="bold"),
    width=160
)
predict_button.pack(pady=15)

result_label = ctk.CTkLabel(
    app,
    text="",
    font=ctk.CTkFont(size=18)
)
result_label.pack(pady=(10, 20))

# --- Lancement de l'application ---
app.mainloop()
