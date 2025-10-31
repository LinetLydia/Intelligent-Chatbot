---
title: Intelligent Chatbot
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: bigscience-openrail-m
short_description: An intelligent chatbot that understands user intent.
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

# 🧠 Intelligent Chatbot

This project demonstrates the design and deployment of an **AI-powered conversational chatbot** built using **Gradio**, **Sentence Transformers**, and **Hugging Face Spaces**.  
The chatbot understands user messages, detects intent, retrieves relevant responses using text embeddings, and provides meaningful replies in real-time.  

---

## Features
- **Intent Recognition:** Classifies user input to understand what the user wants.  
- **Semantic Retrieval:** Uses embeddings to find the most relevant response rather than relying on simple keyword matching.  
- **Interactive UI:** A simple and elegant interface powered by **Gradio** for real-time conversations.  
- **Deployment Ready:** Fully deployed and accessible via **Hugging Face Spaces**.  
- **Integration Ready:** Can be extended with additional NLP components such as sentiment analysis, Named Entity Recognition (NER), or custom datasets.

---

## Tech Stack
| Component | Purpose |
|------------|----------|
| **Python** | Core language used for data processing and model logic |
| **Gradio** | Builds the interactive chatbot interface |
| **Sentence Transformers** | Generates embeddings for intent understanding |
| **Pandas & NumPy** | Data handling and similarity computation |
| **Hugging Face Spaces** | Free web hosting for AI apps |
| **GitHub** | Version control and project collaboration |

---

## Workflow Overview

1. **Data Preparation:**  
   Chat intents and sample responses are loaded from a dataset (`chatbot_intents_dataset.csv`).  

2. **Text Embedding:**  
   Each intent and response is converted into numerical vector form using **Sentence Transformers**.  

3. **Retrieval & Response Selection:**  
   When a user types a message, the chatbot computes similarity between the input and stored embeddings, then retrieves the best-matching intent and response.  

4. **User Interface:**  
   A clean, responsive **Gradio UI** displays the chatbot and allows users to chat seamlessly.  

5. **Deployment:**  
   The final app is deployed to **Hugging Face Spaces** for free public access.

---

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/LinetLydia/Intelligent-Chatbot.git
   cd Intelligent-Chatbot

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   python app.py

Open the local Gradio link (e.g., http://127.0.0.1:7860) in your browser.

## Live Demo
Try the live chatbot here:
[Intelligent Chatbot on Hugging Face](https://huggingface.co/spaces/linet5/intelligent-chatbot)

## Future Enhancements
Integrate Named Entity Recognition (NER) for context-based responses.

Add sentiment analysis to detect tone or emotion.

Expand dataset for more diverse conversation intents.

Integrate memory to maintain conversation history.

## Author
Linet Lydia 
Data Scientist & AI Enthusiast
📍 Nairobi, Kenya
🔗 GitHub | Hugging Face