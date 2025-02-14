import os
import streamlit as st
import openai
import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 🔹 Cargar variables de entorno desde .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# 🔹 Configuración de modelos
EMBEDDING_MODEL = "text-embedding-3-small"  # Cambia a "text-embedding-3-large" si necesitas más precisión
LLM_MODEL = "gpt-3.5-turbo"  # Cambia a "gpt-4-turbo" si el presupuesto lo permite

# 🔹 Cargar embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# 🔹 Directorio de datos
DATA_DIR = "data/landmarks"

# 🔹 Función para cargar archivos de texto
def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts

# 🔹 Cargar datos de landmarks
documents = load_text_files(DATA_DIR)

# 🔹 Optimización: dividir en chunks más pequeños
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
flattened_docs = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]

# 🔹 Evitar Rate Limit: procesar embeddings en lotes
BATCH_SIZE = 20
vector_store = FAISS.from_texts(flattened_docs[:BATCH_SIZE], embeddings)
retriever = vector_store.as_retriever()

# 🔹 Configuración del chatbot
prompt_template = PromptTemplate(
    template="""
    You are an expert travel assistant for Puerto Rico.
    Generate an itinerary based on {days} days of travel.
    
    Example:
    User: "I have 3 days, I like nature and culture."
    Assistant: "Day 1: Visit El Yunque Rainforest..."
    
    Now generate the best itinerary:
    {question}
    """,
    input_variables=["days", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# 🔹 API del clima (WeatherAPI)
def get_weather(location):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"{data['location']['name']}: {data['current']['temp_c']}°C, {data['current']['condition']['text']}"
    return "Weather data unavailable."

# 🔹 Interfaz en Streamlit
st.title("Puerto Rico Travel Itinerary")

# 🔹 Selección de días
days = st.number_input("How many days will you travel?", min_value=1, max_value=14, value=3)

# 🔹 Input de intereses del usuario
interest = st.text_input("Enter your travel interest (e.g., beaches, history, hiking):")

if st.button("Get Itinerary"):
    question = f"I have {days} days and I am interested in {interest}."
    itinerary = qa_chain.run({"days": days, "question": question})
    st.write("### Suggested Itinerary:")
    st.write(itinerary)

    # 🔹 Clima para el destino principal
    st.write("### Weather Forecast:")
    main_location = itinerary.split("\n")[0] if itinerary else "San Juan"
    st.write(get_weather(main_location))
