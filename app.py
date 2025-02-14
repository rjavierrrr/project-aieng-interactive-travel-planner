import streamlit as st
import requests
import os
import pickle
import time
import re
import json
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_folium import folium_static
from sentence_transformers import SentenceTransformer
import folium
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from datetime import datetime

# Load API Keys from environment variables (GitHub Secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("WEATHER_API_KEY")

# ✅ Change embeddings model to `bge-small-en`
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Load data from text files (excluding news)
DATA_DIR = "data"
landmarks, municipalities = [], []

def clean_text(text):
    """Cleans text by removing special characters, extra spaces, and short sentences."""
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces, new lines
    text = re.sub(r"[^\w\s.,!?]", "", text)  # Remove special characters (except punctuation)
    text = text.strip()
    return text if len(text) > 30 else ""  # Remove very short texts

def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            content = file.read()
            cleaned_text = clean_text(content)
            if cleaned_text and cleaned_text not in texts:  # Remove duplicates
                texts.append(cleaned_text)
    return texts

if os.path.exists(f"{DATA_DIR}/landmarks"):
    landmarks = load_text_files(f"{DATA_DIR}/landmarks")
if os.path.exists(f"{DATA_DIR}/municipalities"):
    municipalities = load_text_files(f"{DATA_DIR}/municipalities")

# Combine all texts for embedding
documents = landmarks + municipalities

# Split text into smaller chunks to reduce token usage
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
split_documents = text_splitter.split_text(" ".join(documents))

# ✅ Change to SentenceTransformer embeddings
VECTOR_STORE_PATH = "vector_store.pkl"
if os.path.exists(VECTOR_STORE_PATH):
    with open(VECTOR_STORE_PATH, "rb") as f:
        vector_store = pickle.load(f)
else:
    print("Generating new embeddings... This may take time.")
    embeddings = [embedding_model.encode(text) for text in split_documents]
    vector_store = FAISS.from_embeddings(embeddings, split_documents)
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)
    print("Vector store saved successfully.")

retriever = vector_store.as_retriever()

# Define a travel assistant prompt
prompt_template = PromptTemplate(
    template="""
    You are a friendly and knowledgeable travel assistant specializing in Puerto Rico tourism. 
    Provide engaging and informative responses about places based on user preferences.
    
    User's Question: {question}
    """,
    input_variables=["question"]
)

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7), retriever=retriever, chain_type_kwargs={"prompt": prompt_template})

def get_weather(location):
    """Fetch weather data with retry mechanism."""
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}&aqi=no"
    
    retry_attempts = 3
    for attempt in range(retry_attempts):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        print(f"Weather API request failed. Retrying in {2**attempt} seconds...")
        time.sleep(2 ** attempt)
    
    return {"error": "Could not fetch weather data."}

def recommend_locations(user_interest):
    """Finds locations based on user interest using LangChain retrieval."""
    return qa_chain.run(user_interest)

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
