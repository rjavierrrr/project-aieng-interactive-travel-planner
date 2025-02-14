import streamlit as st
import requests
import os
import openai
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_folium import folium_static
import folium
import json
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from datetime import datetime

# Load OpenAI API Key from GitHub Secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load embedding model (cheaper version)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load data from text files (excluding news)
DATA_DIR = "data"
landmarks, municipalities = [], []

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

# Combine all texts for embedding
documents = landmarks + municipalities

# Split text into smaller chunks to avoid token limits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
split_documents = text_splitter.split_text(" ".join(documents))

# Cache FAISS Vector Store to avoid recomputing embeddings every run
VECTOR_STORE_PATH = "vector_store.pkl"
if os.path.exists(VECTOR_STORE_PATH):
    with open(VECTOR_STORE_PATH, "rb") as f:
        vector_store = pickle.load(f)
else:
    vector_store = FAISS.from_texts(split_documents, embeddings)
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)

retriever = vector_store.as_retriever()

# Define a custom prompt for travel assistant
prompt_template = PromptTemplate(
    template="""
    You are a friendly and knowledgeable travel assistant specializing in Puerto Rico tourism. 
    Provide engaging and informative responses about places based on user preferences.
    
    User's Question: {question}
    """,
    input_variables=["question"]
)

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7), retriever=retriever, chain_type_kwargs={"prompt": prompt_template})

# Weather API Key from GitHub Secrets
API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather(location):
    """Fetch weather data for a given location."""
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
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

def generate_itinerary(locations, days):
    """Creates a structured itinerary JSON based on recommended locations and user-selected days."""
    itinerary = {}
    for i in range(days):
        if i < len(locations):
            itinerary[f"day_{i+1}"] = {
                "location": locations[i],
                "coordinates": get_coordinates(locations[i])
            }
    return json.dumps(itinerary, indent=4)

# Streamlit UI for chatbot interaction
st.title("Puerto Rico Travel Chatbot")

# Ask user for travel dates
days = st.number_input("How many days will you stay in Puerto Rico?", min_value=1, max_value=30, step=1)

# Ask for user interests
interest = st.text_input("What are you interested in? (e.g., beaches, history, hiking):")
if st.button("Get Recommendations"):
    recommendations = recommend_locations(interest)
    st.write("Recommended places to visit:")
    st.write(recommendations)

    # Allow users to lock locations
    selected_location = st.selectbox("Lock a location to visit", recommendations.split('\n'))
    locked_locations = [selected_location]

    if st.button("Confirm Locations"):
        itinerary = generate_itinerary(locked_locations, days)
        st.write("Your suggested itinerary:")
        st.json(itinerary)

        # Check weather for selected locations
        for loc in locked_locations:
            weather = get_weather(loc)
            if "rain" in str(weather).lower():
                st.warning(f"Warning: {loc} might have bad weather!")
