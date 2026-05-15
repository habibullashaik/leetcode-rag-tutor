# =========================================
# DSA RAG TUTOR
# =========================================

# ========= IMPORTS =========

import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="DSA RAG Tutor",
    page_icon="🧠",
    layout="wide"
)


# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #00FFAA;
    text-align: center;
}

.stTextArea textarea {
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# TITLE
# =========================================

st.title("🧠 DSA Problem Solving Tutor")

st.markdown("""
Paste any LeetCode problem.

The AI will generate:
- Simple Explanation
- Pattern Used
- Step-by-Step Algorithm
""")


# =========================================
# GEMINI API
# =========================================

from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
# =========================================
# LOAD VECTORSTORE
# =========================================

@st.cache_resource

def load_vectorstore():

    # ===== LOAD DOCUMENT =====

    loader = TextLoader(
        "hash.txt",
        encoding="utf-8"
    )

    documents = loader.load()


    # ===== TEXT SPLITTING =====

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(
        documents
    )


    # ===== EMBEDDING MODEL =====

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )


    # ===== VECTOR STORE =====

    vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vectorstore


# =========================================
# VECTORSTORE
# =========================================

vectorstore = load_vectorstore()


# =========================================
# RETRIEVER
# =========================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================================
# GEMINI MODEL
# =========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key,
    temperature=0.3
)


# =========================================
# PROMPT TEMPLATE
# =========================================

prompt = ChatPromptTemplate.from_template(
"""
You are an expert DSA tutor.

Analyze the coding problem carefully.

Generate response ONLY in this format:

# 1. Simple Explanation
Explain the problem in beginner-friendly language.

# 2. Pattern Used
Mention the DSA pattern used.

# 3. Step-by-Step Algorithm
Generate logical solving steps.

DO NOT generate code.

Context:
{context}

Question:
{question}
"""
)


# =========================================
# FORMAT DOCS
# =========================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# =========================================
# OUTPUT PARSER
# =========================================

parser = StrOutputParser()


# =========================================
# LCEL RAG CHAIN
# =========================================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)


# =========================================
# USER INPUT
# =========================================

question = st.text_area(
    "Paste LeetCode Question",
    height=300
)


# =========================================
# BUTTON
# =========================================

if st.button("Generate Explanation & Algorithm"):

    if question:

        with st.spinner("Analyzing Problem..."):

            response = rag_chain.invoke(
                question
            )

            st.markdown(response)