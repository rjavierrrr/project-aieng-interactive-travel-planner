import streamlit as st
import requests
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from streamlit_folium import folium_static
import folium
import json
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from datetime import datetime

# Load OpenAI API Key from environment variable
openai.api_key = os.getenv("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")

# Load embedding model
embeddings = OpenAIEmbeddings()

# Load data from text files
DATA_DIR = "data"
landmarks, municipalities, news = [], [], []

# User-selected locked locations
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

# Scrape additional location data from Wikipedia
def scrape_location_info(place):
    url = f"https://en.wikipedia.org/wiki/{place.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return paragraphs[0].text if paragraphs else "No summary available."
    return "Could not retrieve information."

# Combine all texts for embedding
documents = landmarks + municipalities + news

# Create FAISS vector store
vector_store = FAISS.from_texts(documents, embeddings)
retriever = vector_store.as_retriever()

# Define a custom prompt with tone and few-shot learning
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

# OpenWeather API Key (Replace with your actual API key)
API_KEY = "YOUR_OPENWEATHER_API_KEY"

def get_weather(location):
    """Fetch weather data for a given location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
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

def generate_itinerary(locations):
    """Creates a structured itinerary JSON based on recommended locations."""
    itinerary = {f"day_{i+1}": {"location": location, "coordinates": get_coordinates(location)} for i, location in enumerate(locations)}
    return json.dumps(itinerary, indent=4)

# Streamlit UI
st.title("Puerto Rico Travel Planner")

# Ask user for travel dates with a date picker
start_date = st.date_input("Select your travel start date:")
end_date = st.date_input("Select your travel end date:")

# Ask for user interests
interest = st.text_input("Enter your travel interest (e.g., beaches, history, hiking):")
if st.button("Get Recommendations"):
    recommendations = recommend_locations(interest)
    st.write("Recommended places to visit:")
    st.write(recommendations)
    
    # Allow users to lock locations
    selected_location = st.selectbox("Lock a location to visit", recommendations.split('\n'))
    if st.button("Lock Location"):
        locked_locations.append(selected_location)
        st.write(f"Locked Locations: {locked_locations}")
    
    # Generate itinerary with coordinates
    itinerary = generate_itinerary(locked_locations)
    st.write("Suggested Itinerary with Coordinates:")
    st.json(itinerary)
    
    # Check weather dependency
    for loc in locked_locations:
        weather = get_weather(loc)
        if "rain" in str(weather).lower():
            st.warning(f"Warning: {loc} might be affected by bad weather!")
