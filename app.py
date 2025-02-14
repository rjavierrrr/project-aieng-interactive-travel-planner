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

# Load API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Define model
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Data directories
BASE_DATA_DIR = "data"
LANDMARKS_DIR = os.path.join(BASE_DATA_DIR, "landmark")

# Function to split text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Load and clean texts from directories
def load_cleaned_texts(directory):
    texts = []
    if os.path.exists(directory):
        files = sorted(os.listdir(directory))
        for filename in files:
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                raw_html = file.read()
                soup = BeautifulSoup(raw_html, "html.parser")
                for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    script.extract()
                text = soup.get_text(separator=" ").strip()
                cleaned_text = " ".join(text.split())
                texts.extend(chunk_text(cleaned_text))
    return texts

# Load data
landmarks = load_cleaned_texts(LANDMARKS_DIR)

# Vector database path
VECTOR_DB_PATH = "vector_store/faiss_index"

def get_vector_store():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(model=EMBEDDING_MODEL), allow_dangerous_deserialization=True)
    else:
        if not landmarks:
            st.error("No landmark data found. Please check your data directory.")
            st.stop()
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(landmarks, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
        return vector_store

vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Updated prompt to enforce itinerary structure
prompt_template = PromptTemplate(
    template="""
    You are a chatbot specialized in Puerto Rico tourism.
    Your task is to create a structured travel itinerary based on the user's preferences.
    
    User request: "I am traveling for {days} days and I am interested in {interest}."
    
    Provide a detailed itinerary where each day includes:
    - A main location to visit
    - Suggested activities
    - Recommended places to eat
    - Additional tips or nearby attractions
    
    Example Output:
    Day 1:
    - Morning: Visit {context[0]}
    - Afternoon: Try an activity at {context[1]}
    - Evening: Dinner at {context[2]}
    
    Continue this structure for the entire {days}-day trip.
    
    User: I am traveling for {days} days and I am interested in {interest}.
    Assistant:
    """,
    input_variables=["days", "interest", "context"]
)

combine_documents_chain = load_qa_chain(llm=ChatOpenAI(model=LLM_MODEL), chain_type="stuff")
qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_documents_chain)

# Function to fetch weather data
def get_weather(locations, days):
    weather_reports = {}
    for location in locations:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days={days}"
        response = requests.get(url)
        if response.status_code == 200:
            weather = response.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                forecasts = [{
                    "date": day.get("date", "N/A"),
                    "temperature": day.get("day", {}).get("avgtemp_c", "N/A"),
                    "condition": day.get("day", {}).get("condition", {}).get("text", "N/A"),
                    "humidity": day.get("day", {}).get("avghumidity", "N/A"),
                    "wind": day.get("day", {}).get("maxwind_kph", "N/A")
                } for day in forecast]
                weather_reports[location] = forecasts
    return weather_reports

# Streamlit UI
st.title("Puerto Rico Travel Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

start_date = st.date_input("Select your arrival date:")
end_date = st.date_input("Select your departure date:")
interest = st.text_input("What type of trip are you interested in? (e.g., beaches, history, hiking)")

if start_date and end_date and interest:
    num_days = (end_date - start_date).days
    query = f"I am traveling for {num_days} days and I am interested in {interest}."
    
    # Retrieve relevant context
    retrieved_docs = retriever.get_relevant_documents(query)
    st.write(f"Retrieved Context: {retrieved_docs}")
    
    response = qa_chain.invoke({"query": query, "days": num_days, "interest": interest, "context": retrieved_docs})
    itinerary = response.get("result", "I'm not sure how to create an itinerary for that.")
    st.session_state["messages"].append({"role": "assistant", "content": itinerary})
    
    # Extract destinations for weather forecast
    destinations = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", itinerary)
    weather_report = get_weather(destinations, num_days)
    st.session_state["messages"].append({"role": "assistant", "content": f"Weather forecast: {json.dumps(weather_report, indent=2)}"})

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
