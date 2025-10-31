import os
import pickle
import zipfile
import random
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === Step 1: Load chatbot model and embeddings ===
print("🔄 Loading chatbot model and embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    # Try to load from .pkl
    if os.path.exists("chatbot_with_embeddings.pkl"):
        print("✅ Found chatbot_with_embeddings.pkl — loading locally.")
        with open("chatbot_with_embeddings.pkl", "rb") as f:
            return pickle.load(f)
    
    # Try to load from .zip (with your custom name)
    elif os.path.exists("chatbot_with_embeddings.zip"):
        print("✅ Found chatbot_with_embeddings.zip — loading from zip.")
        with zipfile.ZipFile("chatbot_with_embeddings.zip", "r") as z:
            with z.open("chatbot_with_embeddings - Copy.pkl") as f:
                return pickle.load(f)
    
    else:
        raise FileNotFoundError("❌ No chatbot_with_embeddings.pkl or chatbot_with_embeddings.zip found!")

df = load_data()
print(f"✅ Chatbot data loaded successfully! ({len(df)} rows)")

# Fix embeddings that might be lists
if not isinstance(df["embedding"].iloc[0], np.ndarray):
    print("🔧 Converting embeddings from list -> NumPy arrays …")
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))

# Prestack embeddings for faster similarity search
E = np.vstack(df["embedding"].values)
print("✅ Embeddings matrix ready with shape:", E.shape)


# === Step 2: Define chatbot reply function ===
def chatbot_reply(user_input):
    if not user_input.strip():
        return "🤖 Please say something!"

    query_emb = model.encode([user_input], normalize_embeddings=True)
    similarities = cosine_similarity(query_emb, E)[0]
    best_idx = np.argmax(similarities)

    intent = df.iloc[best_idx]["intent"]

    # ✅ Corrected: 'responses' instead of 'response'
    responses = df.iloc[best_idx]["responses"]

    # Handle if responses are a list or string
    if isinstance(responses, (list, tuple)) and responses:
        response = random.choice(responses)
    else:
        response = str(responses)

    score = round(similarities[best_idx], 3)

    return f"({intent}) — {response}\n\nConfidence: {score}"


# === Step 3: Create Gradio interface ===
demo = gr.ChatInterface(
    fn=lambda message, history: chatbot_reply(message),
    title="🧠 Intelligent Chatbot",
    description="A chatbot that understands user intent using embeddings and retrieval.",
)

# === Step 4: Launch interface ===
if __name__ == "__main__":
    print("🚀 Launching chatbot on http://127.0.0.1:7860")
    demo.launch(share=False, debug=False)

