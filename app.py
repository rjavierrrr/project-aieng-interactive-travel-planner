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
LANDMARKS_DIR = os.path.join(BASE_DATA_DIR, "landmark")
MUNICIPALITIES_DIR = os.path.join(BASE_DATA_DIR, "municipalities")

# Función para dividir texto en fragmentos pequeños
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Cargar y limpiar textos de múltiples directorios
def load_cleaned_texts(directories):
    texts = []
    for directory in directories:
        if not os.path.exists(directory):
            continue
        files = sorted(os.listdir(directory))  # Solo los primeros 30 archivos por directorio
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
landmarks = load_cleaned_texts([LANDMARKS_DIR, MUNICIPALITIES_DIR])

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
    You are a chatbot specialized in Puerto Rico tourism.
    Your job is to help users plan their trip by providing detailed itineraries based on their preferences.
    
    If the user specifies the number of days and an interest (e.g., beaches, history, hiking), create a structured itinerary including suggested locations, activities, and key landmarks.
    
    If the user does not specify these details, provide general information about Puerto Rico's tourist attractions.
    
    Use the following knowledge base:
    {context}
    
    User: {query}
    Assistant:
    """,
    input_variables=["query", "context"]
)

# Cargar la cadena de combinación de documentos
combine_documents_chain = load_qa_chain(llm=ChatOpenAI(model=LLM_MODEL), chain_type="stuff")

# Crear la cadena de consulta con RetrievalQA correctamente
qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_documents_chain)

# Obtener datos del clima
def get_weather(locations, days):
    weather_reports = {}
    for location in locations:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days={days}"
        response = requests.get(url)
        if response.status_code == 200:
            weather = response.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                forecasts = []
                for day in forecast:
                    forecasts.append({
                        "date": day.get("date", "N/A"),
                        "temperature": day.get("day", {}).get("avgtemp_c", "N/A"),
                        "condition": day.get("day", {}).get("condition", {}).get("text", "N/A"),
                        "humidity": day.get("day", {}).get("avghumidity", "N/A"),
                        "wind": day.get("day", {}).get("maxwind_kph", "N/A")
                    })
                weather_reports[location] = forecasts
    return weather_reports
    weather_reports = {}
    for location in locations:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days={days}"
        response = requests.get(url)
        if response.status_code == 200:
            weather = response.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                forecasts = []
                for day in forecast:
                    forecasts.append({
                        "date": day.get("date", "N/A"),
                        "temperature": day.get("day", {}).get("avgtemp_c", "N/A"),
                        "condition": day.get("day", {}).get("condition", {}).get("text", "N/A"),
                        "humidity": day.get("day", {}).get("avghumidity", "N/A"),
                        "wind": day.get("day", {}).get("maxwind_kph", "N/A")
                    })
                weather_reports[location] = forecasts
    return weather_reports
    weather_reports = {}
    for location in locations:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days={days}"
        response = requests.get(url)
        if response.status_code == 200:
            weather = response.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                forecasts = []
                for day in forecast:
                    forecasts.append({
                        "date": day.get("date", "N/A"),
                        "temperature": day.get("day", {}).get("avgtemp_c", "N/A"),
                        "condition": day.get("day", {}).get("condition", {}).get("text", "N/A"),
                        "humidity": day.get("day", {}).get("avghumidity", "N/A"),
                        "wind": day.get("day", {}).get("maxwind_kph", "N/A")
                    })
                weather_reports[location] = forecasts
    return weather_reports
    response = requests.get(url)
    if response.status_code == 200:
        weather = response.json()
        forecast = weather.get("forecast", {}).get("forecastday", [])
        if forecast:
            forecasts = []
            for day in forecast:
                forecasts.append({
                    "date": day.get("date", "N/A"),
                    "temperature": day.get("day", {}).get("avgtemp_c", "N/A"),
                    "condition": day.get("day", {}).get("condition", {}).get("text", "N/A"),
                    "humidity": day.get("day", {}).get("avghumidity", "N/A"),
                    "wind": day.get("day", {}).get("maxwind_kph", "N/A")
                })
            return {"location": weather.get("location", {}).get("name", "Unknown"), "forecast": forecasts}
            return {
                "location": weather.get("location", {}).get("name", "Unknown"),
                "temperature": forecast.get("day", {}).get("avgtemp_c", "N/A"),
                "condition": forecast.get("day", {}).get("condition", {}).get("text", "N/A"),
                "humidity": forecast.get("day", {}).get("avghumidity", "N/A"),
                "wind": forecast.get("day", {}).get("maxwind_kph", "N/A")
            }
    return {"error": "Could not fetch weather data."}

# Interfaz con Streamlit
st.title("Puerto Rico Travel Chatbot")

# Inicializar sesión del chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "assistant", "content": "Hello! How many days do you plan to stay in Puerto Rico?"})
    st.session_state["messages"] = []

# Mostrar mensajes previos
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
user_input = st.chat_input("Ask me about travel destinations in Puerto Rico...")

if user_input:
    # Agregar entrada del usuario al chat
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Obtener respuesta del chatbot
    response = qa_chain.invoke({"query": user_input})
    
    # Mostrar respuesta del chatbot
    bot_response = response.get("result", "I'm not sure how to answer that. Try asking something else!")
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    
    # Agregar respuesta del chatbot al historial
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
