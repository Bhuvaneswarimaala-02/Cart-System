# 🛒 Farm2Bag Chat Assistant – AI-Powered E-commerce Support

An intelligent, AI-integrated e-commerce assistant that can:
- Manage shopping carts 🛍️
- Answer user questions 🤖
- Understand natural language queries using LLMs 🧠
- Recommend products 💡
- Support voice input 🎤 and voice response 🔊
- Dynamically respond to queries using RAG (Retrieval-Augmented Generation)

---

## 🚀 Tech Stack

### 💻 Frontend
- **HTML/CSS/JavaScript** – Light UI for chat
- **Speech Recognition API** – Voice input
- **Speech Synthesis API** – Voice output
- **Fetch API** – Sends and receives data to Flask backend

### 🧠 Backend
- **Python + Flask** – REST API & web server
- **Flask-CORS** – Enable cross-origin frontend-backend communication
- **MongoDB** – Store products, cart data, and chat history
- **LangChain** – Orchestrate LLMs, embedding, retrieval, prompts
- **Together AI** – Used for embedding via `TogetherEmbeddings`
- **Groq Cloud + DeepSeek-LLaMA** – Superfast hosted LLM inference
- **FAISS** – Vector store for document retrieval (RAG)
- **BeautifulSoup** – For scraping web content
- **Regex** – Fallback intent recognition

---

## ⚙️ Key Features

### 🤖 AI Assistant
- Uses **Groq-hosted DeepSeek LLM** for generating accurate, fast responses.
- Extracts **intent and entities** using a language model.
- Supports **fallback pattern matching** for better intent coverage.

### 📦 Smart Cart System
- Add, update, remove, view cart items.
- MongoDB stores product and cart info.
- Shows dynamic pricing and totals.

### 🔍 RAG Integration
- Uses **Together Embeddings** to convert content to vectors.
- **FAISS vectorstore** enables semantic search.
- If user’s intent is unknown, triggers RAG-based document Q&A.

### 🌐 Web Scraping
- Scrapes content dynamically from [farm2bag.com](https://www.farm2bag.com/en) using `BeautifulSoup`.

### 🎙️ Voice Support
- Accepts voice input using `webkitSpeechRecognition`.
- Uses `speechSynthesis` to read chatbot responses aloud.

---

## 🧠 NLP + AI Stack

| Task                     | Tool/Model Used                                |
|--------------------------|-------------------------------------------------|
| Intent & entity extraction | `llm.invoke(prompt)` (DeepSeek via Groq)     |
| Embedding generation     | `TogetherEmbeddings`                           |
| Vector search            | `FAISS` (via LangChain)                        |
| LLM answer generation    | `ChatGroq` using `deepseek-r1-distill-llama-70b` |
| Prompt templates         | `ChatPromptTemplate`                           |
| Session memory           | `ChatMessageHistory` (LangChain memory module) |

---

## 🧪 Sample Intents Recognized

- **add_to_cart** → “Add 2 mangoes to my cart”
- **show_cart** → “What’s in my cart?”
- **update_cart** → “Update bananas to 5”
- **remove_from_cart** → “Remove onions”
- **recommend_items** → “Suggest something healthy”
- **get_price** → “How much is 1kg rice?”

If the model fails to understand → triggers **RAG fallback** with vector retrieval.

---

## Local Set Up

- Create a Virtual Environment: python -m venv venv
- Install the requirements: pip install -r requirements.txt
- Set up .env file:
- GROQ_API_KEY=your_groq_api_key
- TOGETHER_API_KEY=your_together_api_key
- OPENAI_API_KEY=optional_if_used
- Move in to directory: cd.\Farm2bag\
- Run the backend: python app1.py
- App runs locally on http://localhost:5000

![image](https://github.com/user-attachments/assets/bb919425-d8e9-49f7-a8df-8fa48c1e9421)

![image](https://github.com/user-attachments/assets/6e8d8197-db45-4b2b-934c-89d5ed5cc16b)

![image](https://github.com/user-attachments/assets/f7b12211-3d42-4b82-bdc7-0a64b7eab3b0)


