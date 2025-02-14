import streamlit as st
import os
import openai
import requests
import json
import re
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
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
BASE_DATA_DIR = "data"
LANDMARK_DIR = os.path.join(BASE_DATA_DIR, "landmark")
MUNICIPALITIES_DIR = os.path.join(BASE_DATA_DIR, "municipalities")

# Función para dividir texto en fragmentos pequeños
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Cargar y limpiar textos de múltiples directorios
def load_cleaned_texts(directories, max_files=30):
    texts = []
    for directory in directories:
        if not os.path.exists(directory):
            continue
        files = sorted(os.listdir(directory))[:max_files]  # Solo los primeros 30 archivos por directorio
        for filename in files:
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                raw_html = file.read()
                soup = BeautifulSoup(raw_html, "html.parser")
                for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    script.extract()
                text = soup.get_text(separator=" ").strip()
                cleaned_text = " ".join(text.split())
                texts.extend(chunk_text(cleaned_text))  # Dividir texto en fragmentos pequeños
    return texts

# Cargar datos de ambas carpetas
landmarks = load_cleaned_texts([LANDMARK_DIR, MUNICIPALITIES_DIR], max_files=30)

# Cargar datos solo si el índice no existe
VECTOR_DB_PATH = "vector_store/faiss_index"

# Evitar re-procesamiento si ya existe un índice
def get_vector_store():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(model=EMBEDDING_MODEL), allow_dangerous_deserialization=True)
    else:
        if not landmarks:
            st.error("No landmark or municipality data found. Please check your data directory.")
            st.stop()
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(landmarks, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)  # Guardar índice localmente
        return vector_store

vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Definir prompt para el chatbot con 'context'
prompt_template = PromptTemplate(
    template="""
    You are a travel assistant specialized in Puerto Rico tourism.
    The user wants to visit places for {days} days.
    Suggest a detailed itinerary based on available landmarks and municipalities.
    
    Based on the following information:
    {context}
    
    Question: {query}
    """,
    input_variables=["days", "query", "context"]
)

# Cargar la cadena de combinación de documentos
combine_documents_chain = load_qa_chain(llm=ChatOpenAI(model=LLM_MODEL), chain_type="stuff")

# Crear la cadena de consulta con RetrievalQA correctamente
qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_documents_chain)

# Obtener datos del clima
def get_weather(location):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days=3"
    response = requests.get(url)
    if response.status_code == 200:
        weather = response.json()
        forecast = weather.get("forecast", {}).get("forecastday", [])[0]
        if forecast:
            return {
                "location": weather.get("location", {}).get("name", "Unknown"),
                "temperature": forecast.get("day", {}).get("avgtemp_c", "N/A"),
                "condition": forecast.get("day", {}).get("condition", {}).get("text", "N/A"),
                "humidity": forecast.get("day", {}).get("avghumidity", "N/A"),
                "wind": forecast.get("day", {}).get("maxwind_kph", "N/A")
            }
    return {"error": "Could not fetch weather data."}

# Extraer todas las ubicaciones del itinerario
def extract_valid_locations(itinerary_text):
    puerto_rico_places = [
        "Adjuntas", "Aguada", "Aguadilla", "Aguas Buenas", "Aibonito", "Añasco", "Arecibo", "Arroyo", "Barceloneta", "Barranquitas", "Bayamón", "Cabo Rojo", "Caguas", "Camuy", "Canóvanas", "Carolina", "Cataño", "Cayey", "Ceiba", "Ciales", "Cidra", "Coamo", "Comerío", "Corozal", "Culebra", "Dorado", "Fajardo", "Florida", "Guánica", "Guayama", "Guayanilla", "Guaynabo", "Gurabo", "Hatillo", "Hormigueros", "Humacao", "Isabela", "Jayuya", "Juana Díaz", "Juncos", "Lajas", "Lares", "Las Marías", "Las Piedras", "Loíza", "Luquillo", "Manatí", "Maricao", "Maunabo", "Mayagüez", "Moca", "Morovis", "Naguabo", "Naranjito", "Orocovis", "Patillas", "Peñuelas", "Ponce", "Quebradillas", "Rincón", "Río Grande", "Sabana Grande", "Salinas", "San Germán", "San Juan", "San Lorenzo", "San Sebastián", "Santa Isabel", "Toa Alta", "Toa Baja", "Trujillo Alto", "Utuado", "Vega Alta", "Vega Baja", "Vieques", "Villalba", "Yabucoa", "Yauco"
    ]
    found_locations = [place for place in puerto_rico_places if place.lower() in itinerary_text.lower()]
    return found_locations if found_locations else ["San Juan"]

# Interfaz con Streamlit
st.title("Puerto Rico Travel Planner")

# Entrada del usuario
days = st.number_input("How many days will you travel?", min_value=1, max_value=30, step=1)
interest = st.text_input("Enter your travel interest (e.g., beaches, history, hiking):")

if st.button("Get Itinerary"):
    query = f"I am interested in {interest} and have {days} days."
    itinerary = qa_chain.invoke({"query": query, "days": days})
    
    if "result" in itinerary:
        st.write("### Suggested Itinerary:")
        st.write(itinerary["result"])
        st.write("### Weather Forecast:")
        locations = extract_valid_locations(itinerary["result"])
        weather_reports = {loc: get_weather(loc) for loc in locations}
        st.json(weather_reports)
    else:
        st.error("No itinerary could be generated. Please try again with different inputs.")
