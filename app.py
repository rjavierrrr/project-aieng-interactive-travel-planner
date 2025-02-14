# Función para obtener clima usando WeatherAPI

def find_weather_forecast(date, location):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days=3"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        for day in weather_data.get("forecast", {}).get("forecastday", []):
            if date == day["date"]:
                return {
                    "date": date,
                    "temperature": day["day"]["avgtemp_c"],
                    "condition": day["day"]["condition"]["text"],
                    "humidity": day["day"]["avghumidity"],
                    "wind": day["day"]["maxwind_kph"]
                }
    return {"error": "Weather data not available"}

# Función para clasificar lugares recomendados
def rank_appropriate_locations(user_prompt):
    response = qa_chain.invoke({"query": user_prompt})
    return response.get("result", "No suitable locations found.")

# Función para obtener información sobre una ubicación específica
def find_info_on_location(user_prompt, location):
    query = f"{user_prompt} about {location}"
    response = qa_chain.invoke({"query": query})
    return response.get("result", "No information available for this location.")

# Función para agregar una ubicación a la lista de lugares a visitar
def add_location_to_visit_list(visit_list, location):
    if location not in visit_list:
        visit_list.append(location)
    return visit_list

# Función para calcular la distancia entre un lugar y la lista de ubicaciones seleccionadas
def compute_distance_to_list(location_list, new_location):
    distances = []
    for loc in location_list:
        if "coordinates" in loc and "coordinates" in new_location:
            loc_coords = (loc["coordinates"]["latitude"], loc["coordinates"]["longitude"])
            new_loc_coords = (new_location["coordinates"]["latitude"], new_location["coordinates"]["longitude"])
            distance_km = geodesic(loc_coords, new_loc_coords).km
            distances.append({"from": loc["name"], "to": new_location["name"], "distance_km": distance_km})
    return distances

# Cargar datos de ambas carpetas
locations_data = load_cleaned_texts([LANDMARKS_DIR, MUNICIPALITIES_DIR])

# Cargar datos solo si el índice no existe
VECTOR_DB_PATH = "vector_store/faiss_index"

def get_vector_store():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(model=EMBEDDING_MODEL), allow_dangerous_deserialization=True)
    else:
        if not locations_data:
            st.error("No landmark or municipality data found. Please check your data directory.")
            st.stop()
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts([json.dumps(loc) for loc in locations_data], embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
        return vector_store

vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Definir prompt para el chatbot con 'context'
prompt_template = PromptTemplate(
    template="""
    You are a chatbot specialized in Puerto Rico tourism.
    Your job is to help users plan their trip by providing detailed itineraries based on their preferences.
    
    The itinerary **MUST** follow this exact format:
    
    Day 1:
    - Visit location A 
    - Enjoy activity B
    - Stay at location C
    
    Day 2:
    - Visit location D
    - Try activity E
    - Explore location F
    
    Use the following knowledge base:
    {context}
    
    User: I am traveling for {days} days and I am interested in {interest}.
    Assistant:
    """,
    input_variables=["days", "interest", "context"]
)

combine_documents_chain = load_qa_chain(llm=ChatOpenAI(model=LLM_MODEL), chain_type="stuff")
qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_documents_chain)

# Interfaz con Streamlit
st.title("Puerto Rico Travel Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

start_date = st.date_input("Select your arrival date:")
end_date = st.date_input("Select your departure date:")
interest = st.text_input("What type of trip are you interested in? (e.g., beaches, history, hiking)")

if start_date and end_date and interest:
    num_days = (end_date - start_date).days
    query = f"I am traveling for {num_days} days and I am interested in {interest}."
    response = qa_chain.invoke({"query": query, "days": num_days, "interest": interest})
    itinerary = response.get("result", "I'm not sure how to create an itinerary for that.")
    st.session_state["messages"].append({"role": "assistant", "content": itinerary})

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
