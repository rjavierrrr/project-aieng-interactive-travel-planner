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
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
VECTOR_STORE_PATH = "vector_store.pkl"

# 📌 Leer solo la data de `landmarks`
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

# 📌 Fragmentación (`chunking`) para evitar `RateLimitError`
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
split_docs = [text_splitter.split_text(doc) for doc in landmarks]
flattened_docs = [chunk for sublist in split_docs for chunk in sublist]

# 📌 Cargar o crear FAISS Vector Store con `batching`
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
    batch_size = 50  # Evitar exceder tokens por minuto (TPM)
    vector_store = FAISS()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    for i in range(0, len(flattened_docs), batch_size):
        batch = flattened_docs[i:i+batch_size]
        try:
            vector_store.add_texts(batch, embeddings)
        except openai.error.RateLimitError:
            print("Rate limit reached, waiting 5 seconds...")
            time.sleep(5)
            vector_store.add_texts(batch, embeddings)

    save_vector_store(vector_store)

retriever = vector_store.as_retriever()

# 📌 Prompt optimizado para turismo
prompt_template = PromptTemplate(
    template="""
    You are a helpful travel assistant specialized in Puerto Rico tourism.
    Your task is to generate a detailed itinerary based on the user's requested number of travel days.

    Example:
    User: "I have 3 days to explore historical landmarks in Puerto Rico."
    Assistant: "Day 1: Visit El Morro and Old San Juan... Day 2: Explore Ponce and its colonial architecture..."

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
    user_query = f"I have {days} days to explore landmarks in Puerto Rico."
    itinerary = generate_itinerary(user_query)
    st.write("📌 Suggested Itinerary:")
    st.write(itinerary)
