import streamlit as st
import requests
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from streamlit_folium import folium_static
import folium
import json
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

# 🔹 Cargar variables de entorno
load_dotenv()

# 🔹 Configurar API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Nueva API Key para WeatherAPI

# 🔹 Configurar embeddings de OpenAI
embeddings = OpenAIEmbeddings()

# 🔹 Directorio de datos
DATA_DIR = "data"

# 🔹 Función para cargar archivos de texto y dividir en fragmentos
def load_and_split_texts(directory):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as file:
                        text = file.read().strip()
                        if text:
                            docs = text_splitter.split_text(text)
                            for doc in docs:
                                documents.append(Document(page_content=doc, metadata={"source": filename}))
    
    return documents

# 🔹 Cargar y dividir los textos
documents = load_and_split_texts(DATA_DIR)

# 🔹 Verificar si FAISS ya existe para cargarlo o generarlo
FAISS_INDEX_PATH = "faiss_index"

if os.path.exists(FAISS_INDEX_PATH):
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
else:
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index saved successfully!")

retriever = vector_store.as_retriever()

# 🔹 Definir el prompt para el asistente de viajes
prompt_template = PromptTemplate(
    template="""
    You are a friendly and knowledgeable travel assistant specializing in Puerto Rico tourism. 
    Always respond in an engaging and informative tone.
    
    Example:
    User: "I'm interested in visiting historical places in San Juan."
    Assistant: "San Juan has a rich history! You can visit El Morro, a historic fortress, or explore Old San Juan's colonial architecture. Would you like me to add these to your itinerary?"
    
    Now, answer the following query:
    {question}
    """,
    input_variables=["question"]
)

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever, chain_type_kwargs={"prompt": prompt_template})

def get_weather(location):
    """Fetch weather data for a given location using WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return {
            "location": weather_data["location"]["name"],
            "region": weather_data["location"]["region"],
            "country": weather_data["location"]["country"],
            "temperature": weather_data["current"]["temp_c"],
            "condition": weather_data["current"]["condition"]["text"],
            "humidity": weather_data["current"]["humidity"],
            "wind_kph": weather_data["current"]["wind_kph"],
        }
    return {"error": "Could not fetch weather data."}

def recommend_locations(user_interest):
    """Finds locations based on user interest using LangChain retrieval."""
    result = qa_chain.run(user_interest)
    return result

def get_coordinates(location):
    """Gets latitude and longitude of a given location."""
    geolocator = Nominatim(user_agent="geoapiExercises")
    loc = geolocator.geocode(location)
    return (loc.latitude, loc.longitude) if loc else (None, None)

def generate_itinerary(locations):
    """Creates a structured itinerary JSON based on recommended locations."""
    itinerary = {f"day_{i+1}": {"location": location, "coordinates": get_coordinates(location)} for i, location in enumerate(locations)}
    return json.dumps(itinerary, indent=4)

# 🔹 Configurar la interfaz en Streamlit
st.title("Puerto Rico Travel Planner")

# 🔹 Selección de fechas
start_date = st.date_input("Select your travel start date:")
end_date = st.date_input("Select your travel end date:")

# 🔹 Capturar intereses del usuario
interest = st.text_input("Enter your travel interest (e.g., beaches, history, hiking):")

locked_locations = []

if st.button("Get Recommendations"):
    recommendations = recommend_locations(interest)
    st.write("Recommended places to visit:")
    st.write(recommendations)
    
    # 🔹 Permitir al usuario bloquear ubicaciones
    selected_location = st.selectbox("Lock a location to visit", recommendations.split('\n'))
    if st.button("Lock Location"):
        locked_locations.append(selected_location)
        st.write(f"Locked Locations: {locked_locations}")
    
    # 🔹 Generar itinerario con coordenadas
    itinerary = generate_itinerary(locked_locations)
    st.write("Suggested Itinerary with Coordinates:")
    st.json(itinerary)
    
    # 🔹 Verificar el clima en los lugares seleccionados
    for loc in locked_locations:
        weather = get_weather(loc)
        if "error" not in weather:
            st.write(f"🌦 **Weather in {weather['location']}**")
            st.write(f"📍 {weather['region']}, {weather['country']}")
            st.write(f"🌡 Temperature: {weather['temperature']}°C")
            st.write(f"☁ Condition: {weather['condition']}")
            st.write(f"💨 Wind: {weather['wind_kph']} kph")
            st.write(f"💧 Humidity: {weather['humidity']}%")
        else:
            st.warning(f"Could not retrieve weather for {loc}")
