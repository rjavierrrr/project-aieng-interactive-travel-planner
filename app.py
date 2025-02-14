import os
import streamlit as st
import openai
import requests
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔹 Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# 🔹 Configuración de modelos
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# 🔹 Cargar embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# 🔹 Directorio de datos (solo landmarks)
DATA_DIR = "data/landmark"

# 🔹 Función para cargar archivos de texto
def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts

# 🔹 Cargar solo data de landmarks
documents = load_text_files(DATA_DIR)

# 🔹 Dividir en chunks más pequeños para optimizar embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
flattened_docs = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]

# 🔹 Evitar Rate Limit: procesar embeddings en lotes
BATCH_SIZE = 20
vector_store = FAISS.from_texts(flattened_docs[:BATCH_SIZE], embeddings)
retriever = vector_store.as_retriever()

# 🔹 Prompt actualizado para `context`
prompt_template = PromptTemplate(
    template="""
    You are an expert travel assistant for Puerto Rico.
    Generate an itinerary for {days} days of travel based on the following context:
    
    {context}
    
    Example:
    User: "I have 3 days, I like nature and culture."
    Assistant: "Day 1: Visit El Yunque Rainforest..."
    
    Now generate the best itinerary based on the given information.
    """,
    input_variables=["days", "context"]
)

# 🔹 Ajuste de RetrievalQA con el nuevo Prompt
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
    context = f"I have {days} days and I am interested in {interest}."
    itinerary = qa_chain.invoke({"days": days, "context": context})  # 🔹 Se usa `context` en vez de `query`
    
    st.write("### Suggested Itinerary:")
    st.write(itinerary)

    # 🔹 Clima para el destino principal
    st.write("### Weather Forecast:")
    main_location = itinerary.split("\n")[0] if itinerary else "San Juan"
    st.write(get_weather(main_location))
