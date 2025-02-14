import streamlit as st
import os
import openai
import requests
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from datetime import datetime

# Cargar API Keys desde variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Definir modelo optimizado
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Directorio de datos
DATA_DIR = "data/landmarks"

# Función para dividir texto en fragmentos pequeños
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Cargar solo los primeros 30 archivos y limpiar textos
def load_cleaned_texts(directory, max_files=30):
    texts = []
    files = sorted(os.listdir(directory))[:max_files]  # Solo los primeros 30 archivos
    for filename in files:
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            raw_html = file.read()
            soup = BeautifulSoup(raw_html, "html.parser")
            text = soup.get_text().strip()
            texts.extend(chunk_text(text))  # Dividir texto en fragmentos pequeños
    return texts

# Cargar datos solo si el índice no existe
VECTOR_DB_PATH = "vector_store/faiss_index"

if os.path.exists(DATA_DIR):
    landmarks = load_cleaned_texts(DATA_DIR, max_files=30)
else:
    landmarks = []

# Evitar re-procesamiento si ya existe un índice
def get_vector_store():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(model=EMBEDDING_MODEL))
    else:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(landmarks, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)  # Guardar índice localmente
        return vector_store

vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Definir prompt para el chatbot
prompt_template = PromptTemplate(
    template="""
    You are a travel assistant specialized in Puerto Rico tourism.
    User wants to visit places for {days} days.
    Suggest a detailed itinerary based on available landmarks.
    
    Query: {query}
    """,
    input_variables=["days", "query"]
)

# Crear la cadena de consulta con RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    input_key="query",  # 🔹 Corregido para que coincida con la entrada esperada
)

# Obtener datos del clima
def get_weather(location):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days=3"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Could not fetch weather data."}

# Interfaz con Streamlit
st.title("Puerto Rico Travel Planner")

# Entrada del usuario
days = st.number_input("How many days will you travel?", min_value=1, max_value=30, step=1)
interest = st.text_input("Enter your travel interest (e.g., beaches, history, hiking):")

if st.button("Get Itinerary"):
    query = f"I am interested in {interest} and have {days} days."
    itinerary = qa_chain.invoke({"query": query})  # 🔹 Ahora pasamos `query`, que es lo que espera

    st.write("### Suggested Itinerary:")
    st.write(itinerary)

    # Clima para el primer destino del itinerario
    st.write("### Weather Forecast:")
    first_location = itinerary.split("\n")[0] if itinerary else "San Juan"
    weather_data = get_weather(first_location)
    st.json(weather_data)
