import streamlit as st
import os
import json
import requests
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from geopy.geocoders import Nominatim

# Cargar claves API desde variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Cargar el modelo de embeddings
embeddings = OpenAIEmbeddings()

# Cargar datos desde archivos
DATA_DIR = "data"
landmarks, municipalities, news = [], [], []

# Ubicaciones bloqueadas
locked_locations = []

def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts

if os.path.exists(f"{DATA_DIR}/landmarks"):
    landmarks = load_text_files(f"{DATA_DIR}/landmarks")
if os.path.exists(f"{DATA_DIR}/municipalities"):
    municipalities = load_text_files(f"{DATA_DIR}/municipalities")
if os.path.exists(f"{DATA_DIR}/news"):
    news = load_text_files(f"{DATA_DIR}/news")

# Combinar textos para el vector store
documents = landmarks + municipalities + news
vector_store = FAISS.from_texts(documents, embeddings)
retriever = vector_store.as_retriever()

# Crear el modelo de QA
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=OPENAI_API_KEY), retriever=retriever)

def get_weather(location, date):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Could not fetch weather data."}

def recommend_locations(user_interest):
    return qa_chain.run(user_interest)

def get_coordinates(location):
    geolocator = Nominatim(user_agent="geoapiExercises")
    loc = geolocator.geocode(location)
    return (loc.latitude, loc.longitude) if loc else (None, None)

def generate_itinerary(days, locations):
    itinerary = {}
    for i in range(days):
        if i < len(locations):
            itinerary[f"Day {i+1}"] = {
                "location": locations[i],
                "coordinates": get_coordinates(locations[i])
            }
    return json.dumps(itinerary, indent=4)

# Interfaz de Streamlit
st.title("Puerto Rico Travel Chatbot")

start_date = st.date_input("Select your travel start date:")
days = st.number_input("Number of days traveling:", min_value=1, step=1)
interest = st.text_input("Enter your travel interest (e.g., beaches, history, hiking):")

if st.button("Get Recommendations"):
    recommendations = recommend_locations(interest)
    st.write("Recommended places to visit:")
    st.write(recommendations)
    selected_location = st.selectbox("Lock a location to visit", recommendations.split('\n'))
    
    if st.button("Lock Location"):
        locked_locations.append(selected_location)
        st.write(f"Locked Locations: {locked_locations}")
    
    itinerary = generate_itinerary(days, locked_locations)
    st.write("Suggested Itinerary:")
    st.json(itinerary)
    
    for loc in locked_locations:
        weather = get_weather(loc, start_date)
        if "rain" in str(weather).lower():
            st.warning(f"Warning: {loc} might have bad weather!")
