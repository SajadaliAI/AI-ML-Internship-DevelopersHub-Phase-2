# 🤖 Advanced Context-Aware AI Chatbot

An **AI-powered, context-aware chatbot** built with **Streamlit, LangChain, Hugging Face Transformers, Hugging Face Embeddings, and FAISS**.

The chatbot is designed to answer questions related to **Artificial Intelligence, Machine Learning, Deep Learning, NLP, Generative AI, LLMs, RAG, LangChain, LangGraph, Python, Embeddings, Vector Databases, and AI Agents**.

It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from a custom knowledge base and generate answers using a Hugging Face language model.

---

## 🎯 Project Objective

The goal of this project is to build a **domain-specific conversational AI chatbot** that can:

* Answer questions from a custom knowledge base
* Perform semantic search using vector embeddings
* Generate context-aware responses
* Maintain conversation history
* Restrict questions to supported AI-related topics
* Provide an interactive chat interface using Streamlit

---

## 🧠 How It Works

The application follows a RAG-based workflow:

```text
User Question
      ↓
Topic / Scope Check
      ↓
Knowledge Base
      ↓
Text Chunking
      ↓
Hugging Face Embeddings
      ↓
FAISS Vector Store
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
FLAN-T5 Language Model
      ↓
Context-Aware Response
```

---

## 🚀 Features

* 🤖 AI-powered conversational chatbot
* 📚 Custom knowledge base using `knowledge.txt`
* 🔎 Semantic search with FAISS
* 🧠 Hugging Face sentence embeddings
* 💬 Conversational memory
* 🔗 LangChain-based RAG pipeline
* 🎯 AI-domain question filtering
* 🖥️ Streamlit web interface
* ⚡ CPU and GPU detection
* 🗑️ Clear chat functionality
* 📖 Supports AI, ML, NLP, RAG, LLMs, LangChain, Python and related topics

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **LangChain**
* **Hugging Face Transformers**
* **Hugging Face Embeddings**
* **FAISS**
* **PyTorch**
* **Sentence Transformers**
* **FLAN-T5**

---

## 📂 Project Structure

```text
Advanced-Context-Aware-Chatbot/
│
├── app.py
│
├── data/
│   └── knowledge.txt
│
├── requirements.txt
│
└── README.md
```

> The exact filenames can be adjusted according to the files included in the project folder.

---

## 📚 Knowledge Base

The chatbot reads its information from:

```text
data/knowledge.txt
```

The text is divided into smaller chunks before being converted into embeddings. The application uses a chunk size of **500 characters** with an overlap of **50 characters**.

---

## 🔎 Retrieval System

The project uses:

**Embedding Model:**

```text
sentence-transformers/all-MiniLM-L6-v2
```

The knowledge-base chunks are converted into numerical vectors using Hugging Face embeddings and stored in a **FAISS vector store** for similarity-based retrieval.

The retriever is configured to return the **top 3 relevant chunks** for a user query.

---

## 🤖 Language Model

The chatbot uses:

```text
google/flan-t5-base
```

through the Hugging Face Transformers pipeline with the `text2text-generation` task.

The model can run on:

* **GPU**, when CUDA is available
* **CPU**, otherwise

The application automatically detects the available device.

---

## 💬 Conversational Memory

The application uses LangChain's:

```text
ConversationBufferMemory
```

to maintain conversation history and support context-aware interactions across multiple questions.

---

## 🎯 Domain Filtering

The chatbot includes a topic-scope checker that verifies whether a question is related to supported AI and technology topics.

Supported areas include:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* NLP
* Generative AI
* LLMs
* RAG
* LangChain
* LangGraph
* Python
* Embeddings
* Vector Databases
* Hugging Face
* AI Agents

## If a question is outside the supported domain, the chatbot returns a predefined fallback response instead of processing it through the RAG system.

## ⚙️ Installation

Install the required Python packages:

```bash
pip install streamlit torch langchain langchain-huggingface transformers sentence-transformers faiss-cpu
```

Depending on the LangChain version used by the project, additional compatible packages may be required.

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 💡 Example Questions

```text
What is Hugging Face?

What is RAG?

How do embeddings work?

What is FAISS?

What is LangChain?

What is semantic search?

What is a vector database?

What is an AI agent?

What is the difference between ML and Deep Learning?
```

---

## 📈 Skills Demonstrated

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Text Embeddings
* Vector Databases
* FAISS
* LangChain
* Hugging Face Transformers
* Conversational Memory
* Streamlit
* Natural Language Processing
* LLM-based Applications
* AI Chatbot Development

---

## 📝 Project Summary

This project demonstrates how a **domain-specific conversational AI application** can combine **document retrieval, vector embeddings, semantic search, conversational memory, and a language model** to provide context-aware answers through an interactive Streamlit interface.
