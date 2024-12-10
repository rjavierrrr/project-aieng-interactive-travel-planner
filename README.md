![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Final Project | The Hitchiker's Guide to Puerto Rico

## Welcome

Welcome to your final project. This is the culmination of your learning journey during this bootcamp. Congratulations on making it until here!

In this project, you are going to use all the AI Engineering skills you have acquired during the course of the last few weeks to build an interactive travel planner for the beautiful island of Puerto Rico. By the end of this project, you will present a working application that cooperates with a visitor to help them build a travel itinerary suitable to their personal preferences.

There is a minimal set of requirements — the Minimum Viable Product (MVP) — that you will need to implement to complete the project successfully, but the way your project will distinguish itself is by whatever features you add to this minimal skeleton. Over the course of the project statement, you will find several hints and suggestions of improvement over the MVP in collapsible sections like this:

<details>
 <summary>Guide: A collapsible section</summary>
  
 > This section contains additional clarifications, optional improvements, and supplementary information that are not critical to the core project requirements. The fully collapsed version of this page will present the MVP.  

</details>

Remember that the contents of the collapsible sections are *just suggestions*; your own ideas for improvement of the guide make for a much more unique and memorable project and should be prioritized whenever feasible.

---

## Interface and minimal functionality

The guide should be served through an application where the main interface with the user is presented as a chatbot that interacts with a prospective traveler and suggests visiting spots in the Island of Puerto Rico based on the interests of the user. It should be able to provide information on several Puerto Rico landmarks, answer questions about events, and help the user compile a list of points of interest to visit. The chatbot should always maintain a friendly and informative tone while remaining focused on helping the user compile a list of points of interest to visit. The end result of a successful interaction with an interested user should be a list of landmarks of interest for the user. The end result of a successful interaction with a non-interested user should be the termination of the conversation in a graceful mode.

<details>
 <summary>Additional context</summary>
  
 > For additional implemented functionality above the MVP, the above parameters may be relaxed. For example: 
 >
 >1. you might want to consider an additional interface with a map of the island to showcase locations of landmarks, or where the user can select areas of interest. In a case like this, maybe your input requires more structure than simple text for a chatbot.
 >2. you might want to return the result as an itinerary segmented by days of travel, or printed on a map. In a case like this, maybe your output requires more structure than a simple list. 

</details>

--- 

## Data

You will find in the `\data` folder the following repositories of raw data:

1. `\data\landmarks`: contains raw .txt files for several Puerto Rico landmarks (extracted from [Wikipedia](https://en.wikipedia.org/wiki/List_of_Puerto_Rico_landmarks))
2. `\data\municipalities`: contains raw .txt files for several Puerto Rico municipalities (extracted from [Wikipedia](https://en.wikipedia.org/wiki/Municipalities_of_Puerto_Rico))
3. `\data\news`: contains raw .txt fies for several years worth of news from the Puerto Rican newspaper [El Mundo](https://www.elmundo.pr/) (extracted from [The Center for Research Libraries](https://gpa.eastview.com/crl/elmundo/?a=p&p=home&e=-------en-25--1--img-txIN----------))


--- 

## Extraction and summarization

You will need to create a repository of structured information for the several Puerto Rico landmarks and municipalities.
You can use an LLM or data scraping and HTML parsing techniques to retrieve, for each of the above *at least* the following information:
1. Location coordinates
2. A small summary of relevant information about the location

Store this information in an appropriately chosen format. Consider using a vector database for storing semantic information on the summaries. Keep in mind that you will be using the summary to match user preferences to potentially interesting locations, so prompt your summary extractions appropriately for this task.

<details>
 <summary>Additional context</summary>
  
 > You are not limited to these datasources, nor to these datapoints. Maybe you want to additionally scrape information about museums, or stores, or restaurants. Maybe you want to add structured information like opening hours or access price, if available.
 >
 >Consider resources like Zoomato, TripAdvisor, Booking, Skyscanner, etc. You can either scrape static information from their APIs or enrich the capabilities of your system later when we're discussing function calling.

</details>

---

## RAG system for perusing news articles

The historic articles for El Mundo contain a wealth of information that may help your user narrow down sites of interest to visit based on their historical significance.

Devise an appropriate chunking strategy and create an appropriate RAG system that retrieves relevant documents to answer user queries and relate past events to user interests.

Notice that the documents are in Spanish, which means that you will have to either:
- translate the documents OR
- use a multilingual system for your embeddings and retrieval (you can use a pre-trained model).

<details>
 <summary>Additional context</summary>
  
 > You may also add structured information to each document, perhaps aligning each document with relevant landmarks and locations.
 >
 > You may enrich this repository with information from other newspapers that you can find in the [Library of Congress](https://www.loc.gov/collections/chronicling-america/dynamic-list-of-titles/?searchType=advanced&st=table&sb=title_s_asc&location_state=puerto+rico). These datasets are image-based, so you will need to make use of a multi-modal model with vision capabilities.
 >
 > Finally, you can of course train your own embedding model for this end.

</details>

---

## Functions for agentic utilization

Create at least the following functions that your agent should be able to call:

- `find_weather_forecast(date,location)`: function calls the [OpenWeather API](https://openweathermap.org/api) and retrieves the weather forecast `location` in `date`.
- `rank_appropriate_locations(user_prompt)`: function evaluates user-fed prompt and goes through RAG system to find appropriate locations to suggest as visitation destinations to the user. For example, if the user mentions liking the sun, beaches should be highly ranked. If they mention liking history, museums and locations with rich historical events should be highly ranked, etc.
- `find_info_on_location(user_prompt, location)`: function evaluates request for information, finds relevant documents regarding `location` and returns information about `location` relevant to the user query. 
- `add_location_to_visit_list(list, location)`: function receives `list`of locations already selected to visit and adds `location` to that list.
- `compute_distance_to_list(list, new_location)`: function that takes the locations already selected to visit and returns the distance of `location` to the closest location already in `list`.

<details>
 <summary>Additional context</summary>
  
 > Here is where you can add most of the customized functionality to improve your system. Some ideas:
 >
 > - you can allow your agent to plan the shortest circuit between locations to visit using a [scheduling API](https://docs.routific.com/reference/getting-started).
 >
 > - you can allow your agent to suggest activities, restaurants, hotels, etc. that are close to the locations in the list of selected landmarks using one of the many [travel APIs](https://tripadvisor-content-api.readme.io/reference/overview).
 >
 > - you can allow your agent to add a point to a map with the chosen locations so the user can visually understand and plan their trip.
 >
 > -...


</details>

---

## Capabilities of the agentic system

The system on your backend should be able to support the following interaction loop:
1. Ask the user for travel dates.
2. Ask the user for interests.
3. Suggest locations based on interests.
4. Answer questions from users regarding locations and events. Ask if the user wants to lock visiting specific locations.
5. Decide if location enjoyment is weather dependent. If so, check weather at location in travel date and issue a warning to reconfirm lock of location.
6. Add location to list of locked locations.
7. Ask if the user is done and either move to 3. or to 8.
8. Return the list of locked locations to the user.

<details>
 <summary>Additional context</summary>
  
 > You may want to separate the system into an orchestrator that decides which functions to run next and a communicator just responsible for conveying information to the user.
 >
 > If you have multiple disparate or competing functionalities (like balancing prices with itineraries, with days, with weather, etc.), you may want to consider using [a crew](https://www.crewai.com/open-source) to reach consensus. 


</details>

---

## Fine-tuning the response agent

You should select a tone of voice for your chatbot and fine-tune it to its intended function.

1. Utilize a powerful LLM, such as the latest models from Anthropic, OpenAI, Google, or Meta, to generate sample interactions that follow your preferred tone of voice. You can include few-shot examples during this generation. Also, make sure to include examples of interactions where the bot refuses to respond or re-steers the conversation back to its intended loop.
2. Use the sample interactions generated in 1. to tune a smaller LLM for these specific tasks. You can use full fine tuning, LoRA, or any other technique to make your weaker model more specialized.

---

## Evaluation process

You must evaluate your system. 
Below - in ascending order of complexity - are some evaluation options. You must complete at least step 1.

1. Evaluate your system qualitatively - test your system thoroughly and frequently. Log and discuss your findings.

2. Create pairs of Questions-Answers that you consider the ground truth. A way to do this can be leveraging on an LLM application (ChatGPT, Claude, etc) to create 30 to 50 correct answers. After, consider using a metrics like [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) to identify if your generated answer matches the correct answer.

3. *LLM-as-a-Judge*: the concept of using *LLM-as-a-Judge* is to have **another** LLM evaluating if the generated answer was correct, based on:

 	- The user's question/input
	- The generated answer
	- The available information (documents, web sources, etc.)

This *LLM-as-a-Judge* should evaluate each answer with a quantitative score.

---

## Deployment

Your model should be served in a cloud architecture. It does not need to be public, nor does it need to have a pleasant interface, but the backend needs to be running on a remote machine.

 <details>
 <summary>Additional context</summary>
  
 > It does not **need** to be public, and it does not **need** to have a pleasant interface, but it can be a point of improvement.

</details>