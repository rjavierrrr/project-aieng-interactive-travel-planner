import streamlit as st
import openai
import os
import json
import time
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from geopy.geocoders import Nominatim

# 📌 Cargar variables de entorno desde `.env`
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📌 Configurar modelos más económicos
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
VECTOR_STORE_PATH = "vector_store.pkl"

# 📌 Leer datos desde `data/`
DATA_DIR = "data"
landmarks = []

def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts

if os.path.exists(f"{DATA_DIR}/landmarks"):
    landmarks = load_text_files(f"{DATA_DIR}/landmarks")
# if os.path.exists(f"{DATA_DIR}/municipalities"):
#     municipalities = load_text_files(f"{DATA_DIR}/municipalities")

# 📌 Unir data sin `news`
documents = landmarks

# 📌 Fragmentación (`chunking`) para evitar `RateLimitError`
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
split_docs = [text_splitter.split_text(doc) for doc in documents]

# 📌 Cargar o crear FAISS Vector Store
def save_vector_store(vector_store):
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)

def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return None

vector_store = load_vector_store()
if vector_store is None:
    vector_store = FAISS.from_texts([chunk for sublist in split_docs for chunk in sublist], OpenAIEmbeddings(model=EMBEDDING_MODEL))
    save_vector_store(vector_store)

retriever = vector_store.as_retriever()

# 📌 Prompt optimizado para turismo
prompt_template = PromptTemplate(
    template="""
    You are a helpful travel assistant specialized in Puerto Rico tourism.
    Your task is to generate a detailed itinerary based on the user's requested number of travel days.

    Example:
    User: "I have 3 days to explore beaches and historical sites in San Juan."
    Assistant: "Day 1: Visit El Morro and Old San Juan... Day 2: Explore Condado Beach..."

    Now, generate an itinerary based on the user's input:
    {question}
    """,
    input_variables=["question"]
)

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name=LLM_MODEL), retriever=retriever, chain_type_kwargs={"prompt": prompt_template})

# 📌 Obtener coordenadas de un lugar
def get_coordinates(location):
    geolocator = Nominatim(user_agent="geoapiExercises")
    loc = geolocator.geocode(location)
    return (loc.latitude, loc.longitude) if loc else (None, None)

# 📌 Generar itinerario con reintentos
def generate_itinerary(user_request):
    try:
        return qa_chain.run(user_request)
    except openai.error.RateLimitError:
        time.sleep(5)
        return qa_chain.run(user_request)

# 📌 Interfaz Streamlit
st.title("📍 Puerto Rico Travel Itinerary Planner")

days = st.number_input("How many days will you be traveling?", min_value=1, max_value=14, step=1)
interest = st.text_input("What are your interests? (e.g., beaches, history, hiking)")

if st.button("Generate Itinerary"):
    user_query = f"I have {days} days to explore {interest} in Puerto Rico."
    itinerary = generate_itinerary(user_query)
    st.write("📌 Suggested Itinerary:")
    st.write(itinerary)
