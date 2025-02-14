import os
import streamlit as st
import openai
import requests
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔹 Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# 🔹 Configuración de modelos
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# 🔹 Cargar embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# 🔹 Directorio de datos (solo landmarks)
DATA_DIR = "data/landmark"

# 🔹 Función para cargar archivos de texto
def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts

# 🔹 Cargar solo data de landmarks
documents = load_text_files(DATA_DIR)

# 🔹 Dividir en chunks más pequeños para optimizar embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
flattened_docs = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]

# 🔹 Evitar Rate Limit: procesar embeddings en lotes
BATCH_SIZE = 20
vector_store = FAISS.from_texts(flattened_docs[:BATCH_SIZE], embeddings)
retriever = vector_store.as_retriever()

# 🔹 Prompt actualizado con `query`
prompt_template = PromptTemplate(
    template="""
    You are an expert travel assistant for Puerto Rico.
    Generate an itinerary based on the following query:

    Query: {query}

    Example:
    User: "I have 3 days, I like nature and culture."
    Assistant: "Day 1: Visit El Yunque Rainforest..."
    
    Now generate the best itinerary based on the given information.
    """,
    input_variables=["query"]
)

# 🔹 Ajuste de RetrievalQA con `query` corregido
q
