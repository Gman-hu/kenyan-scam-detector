import gradio as gr
import joblib
import re
from scipy.sparse import hstack
import os

# 1. Load saved artifacts
model = joblib.load('scam_detector_model.pkl')
word_vec = joblib.load('word_vectorizer.pkl')
char_vec = joblib.load('char_vectorizer.pkl')

# 2. Text preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Prediction Function for Gradio
def analyze_message(message):
    if not message.strip():
        return "Please enter a valid message.", {}

    cleaned = clean_text(message)
    X_word = word_vec.transform([cleaned])
    X_char = char_vec.transform([cleaned])
    X_combined = hstack([X_word, X_char])
    
    probabilities = model.predict_proba(X_combined)[0]
    safe_prob = float(probabilities[0])
    scam_prob = float(probabilities[1])

    confidences = {"Legitimate / Safe": safe_prob, "Fraud / Scam": scam_prob}
    
    if scam_prob >= 0.70:
        verdict = f"🚨 HIGH RISK DETECTED ({scam_prob*100:.1f}% Confidence)\nThis message displays structural patterns typical of financial fraud or psychological manipulation."
    elif scam_prob >= 0.40:
        verdict = f"⚠️ SUSPICIOUS MESSAGE ({scam_prob*100:.1f}% Risk Level)\nExercise caution. Verify the sender independently before taking action."
    else:
        verdict = f"✅ SAFE MESSAGE ({safe_prob*100:.1f}% Confidence)\nNo major fraud or phishing indicators detected."

    return verdict, confidences

# 4. Build Gradio UI Dashboard
theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")

demo = gr.Interface(
    fn=analyze_message,
    inputs=gr.Textbox(lines=4, placeholder="Paste SMS here...", label="📱 Incoming SMS Message"),
    outputs=[gr.Textbox(label="🔍 Risk Analysis Summary", interactive=False), gr.Label(label="📊 Model Confidence Distribution")],
    title="🇰🇪 Kenyan Mobile Money Fraud & Scam Detector",
    description="An AI safety tool powered by Logistic Regression and TF-IDF character/word N-gram analysis, trained on localized Swahili, Sheng, and M-Pesa fraud vectors.",
    examples=[
        ["CONFIRMED. You have received Ksh 2,500.00 from JOHN DOE 0712345678 on 12/8/2026 at 2:15 PM."],
        ["Buda nina shida ya urgent, tuma 500 kwa hii namba mpya nikutext baadaye."],
        ["ATTENTION: Your M-Pesa account has been suspended due to unverified ID details. Click http://bit.ly/fake-mpesa to unlock immediately."],
        ["Aha, Leo tumepanga kuenda supa kununua vitu za home na beshte yangu."]
    ]
)

# 5. Launch the application
if __name__ == "__main__":
    # Get Render's dynamic port or default to 10000
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 so Render can detect the live app
    demo.launch(theme=theme, server_name="0.0.0.0", server_port=port)
