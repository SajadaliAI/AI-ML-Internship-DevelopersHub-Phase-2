import streamlit as st
import torch

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from transformers import pipeline


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="Advanced Context-Aware Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Title
# ============================================================

st.title("🤖 Advanced Context-Aware Chatbot")

st.caption(
    "Ask questions related to AI, Machine Learning, Deep Learning, "
    "NLP, RAG, LLMs, Python, LangChain and related technologies."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Chatbot Information")

    st.info(
        "This chatbot is designed to answer questions related to "
        "Artificial Intelligence and related technologies."
    )

    st.markdown(
        """
        **Supported Topics**

        • Artificial Intelligence  
        • Machine Learning  
        • Deep Learning  
        • NLP  
        • Generative AI  
        • LLMs  
        • RAG  
        • LangChain  
        • LangGraph  
        • Chatbots  
        • Python  
        • Embeddings  
        • Vector Databases  
        • Hugging Face  
        • AI Agents
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.history = []

        # Reset conversation memory
        if "memory" in st.session_state:
            st.session_state.memory.clear()

        st.rerun()


# ============================================================
# Device Detection
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.success(f"Running on: {'GPU' if device == 'cuda' else 'CPU'}")


# ============================================================
# Allowed AI Domain Keywords
# ============================================================

ALLOWED_KEYWORDS = [

    # AI
    "artificial intelligence",
    "artificial intelligence",
    "ai",

    # Machine Learning
    "machine learning",
    "ml",

    # Deep Learning
    "deep learning",
    "neural network",
    "cnn",
    "rnn",
    "lstm",

    # NLP
    "nlp",
    "natural language",
    "natural language processing",
    "tokenization",
    "tokenizer",
    "sentiment analysis",

    # Generative AI / LLM
    "generative ai",
    "genai",
    "llm",
    "large language model",
    "language model",
    "gpt",
    "transformer",
    "transformers",
    "bert",
    "flan",
    "hugging face",
    "huggingface",

    # RAG
    "rag",
    "retrieval",
    "retrieval augmented generation",
    "semantic search",
    "vector",
    "vector database",
    "vector store",
    "embedding",
    "embeddings",
    "faiss",
    "chromadb",
    "chroma",

    # LangChain / LangGraph
    "langchain",
    "langgraph",
    "prompt",
    "prompttemplate",
    "agent",
    "agents",
    "agentic ai",
    "tool calling",

    # Chatbots
    "chatbot",
    "chatbots",
    "conversational ai",

    # Python / Programming
    "python",
    "programming",
    "coding",
    "software development",

    # Frameworks
    "streamlit",
    "fastapi",
    "flask",

    # Data Science
    "data science",
    "data preprocessing",
    "feature engineering",
    "classification",
    "regression",
    "clustering"
]


# ============================================================
# Scope Checker
# ============================================================

def is_in_scope(question: str) -> bool:

    question = question.lower().strip()

    return any(
        keyword in question
        for keyword in ALLOWED_KEYWORDS
    )


# ============================================================
# Fallback Response
# ============================================================

FALLBACK_RESPONSE = """
❌ I don't have information about this topic.

🤖 I am designed to answer questions related to:

• Artificial Intelligence
• Machine Learning
• Deep Learning
• NLP
• Generative AI
• LLMs
• RAG
• LangChain
• LangGraph
• Chatbots
• Python
• Embeddings
• Vector Databases
• Hugging Face
• AI Agents

✅ Please ask a question related to these topics.
"""


# ============================================================
# Load Knowledge Base
# ============================================================

@st.cache_resource
def load_knowledge_base():

    try:

        with open(
            "data/knowledge.txt",
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

    except FileNotFoundError:

        st.error(
            "❌ data/knowledge.txt was not found. "
            "Please make sure the data folder exists."
        )

        st.stop()

    # --------------------------------------------------------
    # Text Splitting
    # --------------------------------------------------------

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": device
        }
    )

    # --------------------------------------------------------
    # FAISS Vector Store
    # --------------------------------------------------------

    vectorstore = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vectorstore


# ============================================================
# Load LLM
# ============================================================

@st.cache_resource
def load_llm():

    try:

        # FLAN-T5 is a text-to-text model.
        # Depending on the installed Transformers version,
        # text2text-generation should be supported.

        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            device=0 if device == "cuda" else -1
        )

        llm = HuggingFacePipeline(
            pipeline=hf_pipeline
        )

        return llm

    except Exception as e:

        st.error(
            f"❌ Failed to load Hugging Face model:\n\n{e}"
        )

        st.stop()


# ============================================================
# Initialize Resources
# ============================================================

vectorstore = load_knowledge_base()

llm = load_llm()


# ============================================================
# Conversation Memory
# ============================================================

if "memory" not in st.session_state:

    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


# ============================================================
# RAG Chain
# ============================================================

@st.cache_resource
def create_rag_chain(_vectorstore, _llm):

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        ),
        memory=memory
    )

    return chain


qa_chain = create_rag_chain(
    vectorstore,
    llm
)


# ============================================================
# Chat History
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# Display Previous Messages
# ============================================================

for speaker, message in st.session_state.history:

    if speaker == "User":

        with st.chat_message("user"):

            st.markdown(message)

    else:

        with st.chat_message("assistant"):

            st.markdown(message)


# ============================================================
# User Input
# ============================================================

user_question = st.chat_input(
    "Ask a question about AI, ML, RAG, Python..."
)


# ============================================================
# Process User Question
# ============================================================

if user_question:

    # --------------------------------------------------------
    # Display User Message Immediately
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_question)

    st.session_state.history.append(
        ("User", user_question)
    )

    # --------------------------------------------------------
    # Check Topic
    # --------------------------------------------------------

    if not is_in_scope(user_question):

        response = FALLBACK_RESPONSE

    else:

        # ----------------------------------------------------
        # Generate Response
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                try:

                    response = qa_chain.run(
                        user_question
                    )

                except Exception as e:

                    response = (
                        "❌ Sorry, an error occurred while "
                        "generating the response.\n\n"
                        f"Error: `{e}`"
                    )

            st.markdown(response)

    # --------------------------------------------------------
    # Save Bot Response
    # --------------------------------------------------------

    st.session_state.history.append(
        ("Bot", response)
    )