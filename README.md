# Internshala_DeepankSingh_Assignment19


---

# Hybrid RAG (Retrieval-Augmented Generation)

Hybrid RAG is a Python application that combines **vector search** and **keyword search** to answer user queries effectively.
It uses **OpenAI’s API** for language understanding and a hybrid search approach for more accurate results.

---

## 🚀 Features

* **Hybrid Retrieval** – Combines semantic search (vector similarity) with keyword search (BM25).
* **RAG Pipeline** – Retrieves relevant documents and uses them with OpenAI’s GPT models for better answers.
* **FastAPI Backend** – Lightweight API for running queries.
* **Configurable** – Easily switch models, embeddings, and retrieval parameters.

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/hybrid_rag.git
cd hybrid_rag
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

1. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

2. Get your API key from [OpenAI](https://platform.openai.com/).

---

## ▶️ Running the App

```bash
uvicorn main:app --reload
```

By default, the app will be available at:

```
http://127.0.0.1:8000
```

---

## 📌 API Usage

**Endpoint:** `POST /query`
**Request Body Example:**

```json
{
  "query": "What is Retrieval-Augmented Generation?"
}
```

**Response Example:**

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique..."
}
```

---

## 🛠 Technologies Used

* **Python 3.10+**
* **FastAPI**
* **OpenAI API**
* **FAISS** for vector search
* **BM25** for keyword search

---

## 📄 License

This project is licensed under the MIT License.

---

