# 🌐 Website RAG Chatbot

A Streamlit-based AI chatbot that can crawl any website, embed its contents using OpenAI embeddings, store them in Qdrant vector database, and answer questions based on the site content using GPT-4. Built with LangChain, Qdrant, and OpenAI.

---

## 🚀 Features

- 🌍 Crawl and extract internal pages from a website
- 🧠 Embed content using OpenAI Embeddings (`text-embedding-3-large`)
- 📦 Store embeddings in Qdrant vector database
- 💬 Ask questions in a chat interface using OpenAI GPT-4
- 📊 Real-time progress bars for both crawling and uploading

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/website-rag-chatbot.git
cd website-rag-chatbot
pip install -r requirements.txt
```

---

## 🔐 Setup Environment Variables

Store your keys in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-api-key"
QDRANT_CLOUD_URL = "https://your-qdrant-instance.com"
QDRANT_API_KEY = "your-qdrant-api-key"
```

---

## ▶️ Run the App

```bash
streamlit run main.py
```

---

## 📦 Requirements

```
streamlit
requests
beautifulsoup4
openai
langchain
langchain-community
langchain-openai
qdrant-client
tqdm
```

---

## 📄 License

MIT License © 2025 Manunjay Bhardwaj