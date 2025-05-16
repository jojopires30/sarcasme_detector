
#Groupe M06
# interface.py

import customtkinter as ctk
import re
import joblib

# Configuration générale
ctk.set_appearance_mode("dark")   # "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Tu peux mettre "green", "dark-blue", "blue", etc.

# Chargement du modèle
model = joblib.load("sarcasm_model.joblib")

# Nettoyage du texte
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\!\?\"\'\-]", "", text)
    return text

# Fonction de prédiction
def predict():
    user_input = text_input.get("0.0", "end").strip()
    if not user_input:
        result_label.configure(text="⚠️ Veuillez entrer une phrase.")
        return
    cleaned = clean_text(user_input)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0][1]
    label = "😏 Sarcastique" if pred == 1 else "🙂 Pas sarcastique"
    confidence = round(prob * 100, 2)
    result_label.configure(text=f"{label} (confiance : {confidence}%)")

# Fenêtre principale
app = ctk.CTk()
app.title("Détecteur de Sarcasme")
app.geometry("600x400")

# Widgets
title = ctk.CTkLabel(app, text="🧠 Détection de Sarcasme", font=ctk.CTkFont(size=20, weight="bold"))
title.pack(pady=20)

text_input = ctk.CTkTextbox(app, width=500, height=100, font=("Helvetica", 14))
text_input.pack(pady=10)

predict_button = ctk.CTkButton(app, text="Prédire", command=predict, font=ctk.CTkFont(size=14, weight="bold"))
predict_button.pack(pady=15)

result_label = ctk.CTkLabel(app, text="", font=ctk.CTkFont(size=16))
result_label.pack(pady=20)

# Lancement
app.mainloop()
