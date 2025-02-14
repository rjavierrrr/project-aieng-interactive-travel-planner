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

# Función para limpiar y cargar los textos de los landmarks y municipios
def load_cleaned_texts(directories):
    texts = []
    for directory in directories:
        if not os.path.exists(directory):
            continue
        files = sorted(os.listdir(directory))
        for filename in files:
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                raw_html = file.read()
                soup = BeautifulSoup(raw_html, "html.parser")
                for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    script.extract()
                text = soup.get_text(separator=" ").strip()
                cleaned_text = " ".join(text.split())
                texts.append(cleaned_text)
    return texts

# Cargar datos de ambas carpetas
locations_data = load_cleaned_texts([LANDMARKS_DIR, MUNICIPALITIES_DIR])

# Cargar datos solo si el índice no existe
VECTOR_DB_PATH = "vector_store/faiss_index"

def get_vector_store():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(model=EMBEDDING_MODEL), allow_dangerous_deserialization=True)
    else:
        if not locations_data:
            st.error("No landmark or municipality data found. Please check your data directory.")
            st.stop()
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts([json.dumps(loc) for loc in locations_data], embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
        return vector_store

vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Definir prompt para el chatbot con estructura de itinerario
prompt_template = PromptTemplate(
    template="""
    You are a chatbot specialized in Puerto Rico tourism.
    Your job is to help users plan their trip by providing detailed itineraries based on their preferences.

    The itinerary **MUST** follow this exact format:

    Day 1:
    - Visit location A
    - Enjoy activity B
    - Stay at location C

    Day 2:
    - Visit location D
    - Try activity E
    - Explore location F

    Use the following knowledge base:
    {context}

    User: I am traveling for {days} days and I am interested in {interest}.
    Assistant:
    """,
    input_variables=["days", "interest", "context"]
)

combine_documents_chain = load_qa_chain(llm=ChatOpenAI(model=LLM_MODEL), chain_type="stuff")
qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_documents_chain)

# Obtener datos del clima con WeatherAPI
def find_weather_forecast(date, location):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days=3"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        for day in weather_data.get("forecast", {}).get("forecastday", []):
            if date == day["date"]:
                return {
                    "date": date,
                    "temperature": day["day"]["avgtemp_c"],
                    "condition": day["day"]["condition"]["text"],
                    "humidity": day["day"]["avghumidity"],
                    "wind": day["day"]["maxwind_kph"]
                }
    return {"error": "Weather data not available"}

# Interfaz con Streamlit
st.title("Puerto Rico Travel Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

start_date = st.date_input("Select your arrival date:")
end_date = st.date_input("Select your departure date:")
interest = st.text_input("What type of trip are you interested in? (e.g., beaches, history, hiking)")

if start_date and end_date and interest:
    num_days = (end_date - start_date).days
    query = f"I am traveling for {num_days} days and I am interested in {interest}."
    response = qa_chain.invoke({"query": query, "days": num_days, "interest": interest})
    itinerary = response.get("result", "I'm not sure how to create an itinerary for that.")
    st.session_state["messages"].append({"role": "assistant", "content": itinerary})

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
