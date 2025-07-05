#!/usr/bin/env python
# coding: utf-8

# # Prompt Template

# In[1]:


def normalize_count(value):
    if isinstance(value, list):
        return sum(int(v) for v in value if str(v).isdigit())
    elif isinstance(value, (int, float)):
        return value
    elif value is None:
        return 0
    else:
        try:
            return int(value)
        except:
            return 0

def clean_slot_list(values):
    """
    Removes Python None and string 'None' from a list.
    Returns [] if the result is empty or all values were 'None'.
    """
    if not isinstance(values, list):
        return values  # Leave non-lists unchanged, most likely it is None

    cleaned = [v for v in values if v is not None and str(v).strip().lower() != "none"]
    return cleaned if cleaned else []

def generate_prompt(user_query, slots, rag_chunks=[], max_words=400, country="Singapore"):
    # Assistant persona and context intro
    assistant_context = (
        f"You are a highly experienced travel expert and advisor for {country}. "
        "Your task is to provide a well-structured and practical travel itinerary based on the user's needs."
    )

    query_parts = [f"as a"]

    # Build query parts dynamically
    persona = clean_slot_list(slots.get("persona"))
    if persona is None or len(persona) == 0:
        persona = ['tourist'] # default to tourist if persona is not detected
    if len(persona) > 0:
        query_parts.append(' and '.join(persona))

    query_parts.append(f", the user wants to do {slots.get('intent', 'itinerary planning')}") #default to itinerary planning

    location = clean_slot_list(slots.get("location"))
    if location is None:
        location = [country] # default to country if location is not detected
    if len(location) > 0:
        query_parts.append(f"in {', '.join(location)}.")

    if slots.get("date"):
        query_parts.append(f"The travel dates are {', '.join(slots['date'])}.")

    duration_days = clean_slot_list(slots.get("duration_days"))
    # duration = slots.get("duration_days")
    if len(duration_days) > 0:    
    # if isinstance(duration, list):
        query_parts.append(f"The trip duration is {', '.join(str(d) for d in duration_days)} days.")
    # else:
    #     query_parts.append(f"The trip duration is {duration_days} days.")

    food = clean_slot_list(slots.get("food"))
    if food is not None:
        if len(food) > 0:
            query_parts.append(f"The user prefers {', '.join(food)} cuisine.")

    budget = clean_slot_list(slots.get("budget"))
    if budget is not None:
        if len(budget) > 0:
            query_parts.append(f"Budget is {budget}.")

    transport = clean_slot_list(slots.get("transport"))
    if transport is not None:
        if len(transport) > 0:
            query_parts.append(f"Preferred transport options include {', '.join(transport)}.")

    event = clean_slot_list(slots.get("event"))
    if event is not None:
        if len(event) > 0:
            query_parts.append(f"The user is interested in events like {', '.join(event)}.")

    style = clean_slot_list(slots.get("style"))
    if style is not None:
        if len(style) > 0:
            query_parts.append(f"The travel style is {', '.join(style)}.")

    num_adults = normalize_count(slots.get("num_adults"))
    num_kids = normalize_count(slots.get("num_kids"))
    family_members = []
    if num_adults:
        family_members.append(f"{num_adults} adult{'s' if num_adults != 1 else ''}")
    if num_kids:
        family_members.append(f"{num_kids} kid{'s' if num_kids != 1 else ''}")
    if family_members:
        query_parts.append(f"Traveling group includes {', '.join(family_members)}.")

    special = clean_slot_list(slots.get("special"))
    if special is not None:
        if len(special) > 0:
            query_parts.append(f"Special preferences include {', '.join(special)}.")

    query_template = " ".join(query_parts)

    rag_context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(rag_chunks)]) if len(rag_chunks) > 0 else ""

    guidelines = (
           f"1. Keep the response concise, engaging and aligned with the user's intent and preferences.\n"
            "2. Maintain a helpful, friendly and expert tone.\n"
           f"3. The response must not exceed {max_words} words.\n" 
            # "4. Avoid repeating the provided facts verbatim. Instead, rephrase the meaning naturally into the response.\n"
        )


    # Final prompt
    prompt = (
        f"{assistant_context}\n\n"
        f"Here is the user's original request:\n {user_query}\n\n"
        f"Specifically, {query_template}\n\n"
    )

    if len(rag_context) > 0:
        prompt += f"Additional factual information about {country}. Please integrate this context seamlessly into your advice:\n{rag_context}\n\n"

    prompt += (
        f"Instructions:\n {guidelines}\n\n"
        "Now, please generate an itinerary with one relaxing day to rest in the middle."
    )    

    return prompt


# # Test generating prompt

# ### Sample 1

# In[2]:


slots = {
  "persona": ['Family Traveler', 'Foodie'],
  "intent": "explore culture",
  "location": ["Singapore"],
  "date": ["July 2025"],
  "duration_days":[7],
  "food": ["vegetarian"],
  "budget": [1000],
  "style": ["backpacking"],
  "num_adults": 1,
  "num_kids": 0,
}
rag_chunks = [
    "The National Gallery hosts rotating exhibits on Southeast Asian art.",
    "Hawker centers like Lau Pa Sat offer many vegetarian local options.",
]
user_query = "Hi, I'm planning a 7-day backpacking trip to Singapore in July 2025. I'm traveling alone and really want to explore the local culture, especially museums and art. I'm vegetarian and on a budget of about SGD 1000. Can you suggest cultural places to visit and affordable vegetarian food options?"
prompt_template = generate_prompt(user_query, slots, rag_chunks)
print(f"---------- Prompt Template -------------")
print(prompt_template)
print(f"----------------------------------------")


# ### Sample 2

# In[3]:


sample = 2
slots = {
  "persona": ['family traveller'],
  "intent": "family vacation",
  "location": ["Sentosa", "Singapore"],
  "date": ["December 2025"],
  "duration_days": [5],
  "food": ["halal"],
  "budget": [1500],
  "style": ["relaxing"],
  "num_adults": 2,
  "num_kids": 2,
}

rag_chunks = [
    "Sentosa offers a wide range of family-friendly attractions including Universal Studios Singapore and the S.E.A. Aquarium.",
    "Many restaurants in Sentosa and VivoCity provide halal-certified options, ensuring accessible dining for Muslim families.",
    "Siloso Beach is ideal for kids with its calm waters and beachside activities."
]
user_query = "Hi, we’re planning a relaxing 5-day family trip to Singapore this December. We’ll be spending most of our time around Sentosa. We’re a family of four, and we need halal food options. Can you suggest a fun yet laid-back itinerary that’s suitable for kids? Our budget is about SGD 1500."

prompt_template = generate_prompt(user_query, slots, rag_chunks)
print(f"---------- Prompt Template -------------")
print(prompt_template)
print(f"----------------------------------------")


# ### Sample 3

# In[59]:


sample = 3
slots = {
    # "persona": "nature lover",
    "intent": "adventure travel",
    "location": ["Singapore"],
    "date": ["March 2026"],
    "duration_days": [5],
    "food": ["gluten-free"],
    "budget": "SGD 1500",
    "style": ["backpacking", "outdoor activities"],
    "num_adults": 1,
    "num_kids": 0,
}
rag_chunks = []  # empty
user_query = """
I’m a solo traveler planning a 5-day backpacking trip to Singapore in March 2026. 
I need gluten-free food options.  
I want to focus on outdoor activities and adventure.
My budget is SGD 1500.
Can you suggest an itinerary that fits these preferences?
"""
prompt_template = generate_prompt(user_query, slots, rag_chunks)
print(f"---------- Prompt Template -------------")
print(prompt_template)
print(f"----------------------------------------")


# ## Sample 4

# In[8]:


sample = 4
slots = {
    'intent': 'PlanItinerary',
    'persona': ['Family Traveller'],
    'location': ['singapore'],
    'date': ['2025-06-05'],
    'duration_days': ['5 days', '1 rest day'],
    'food': ['None'],
    'budget': ['None'],
    'transport': ['None'],
    'event': ['interactive science exhibits','nature parks','kid-friendly activities'],
    'style': ['None'],
    'num_kids': [],
    'num_adults': ["6"],
    'special': ['None'],
}
rag_chunks = []  # empty
user_query = "We're a family of 4 with two children aged 6 and 9 visiting Singapore for 5 days. We love interactive science exhibits, nature parks, and kid-friendly activities. Can you suggest an itinerary with one rest day in the middle? Show us relevant attractions with images."
print(f"Sample {sample}\n")
prompt_template = generate_prompt(user_query, slots, rag_chunks)
print(f"---------- Prompt Template -------------")
print(prompt_template)
print(f"----------------------------------------")


# ## Sample 5

# In[9]:


slots = { 
    "intent": "PlanItinerary", 
    "location": [ "singapore" ], 
    "date": [ "next 6 days" ], 
    "duration_days": [ "6 days" ], 
    "food": [], 
    "budget": [], 
    "transport": [], 
    "event": [ "outdoor activities", "hiking trails", "cycling", "night safaris" ], 
    "style": [ "adventure" ], 
    "num_kids": [], 
    "num_adults": [ "6" ], 
    "special": [],
      "persona": [ "Adventure Seeker", "Family Traveler" ] 
}
rag_chunks = []  # empty
user_query = "We're a family of 4 with two children aged 6 and 9 visiting Singapore for 5 days. We love interactive science exhibits, nature parks, and kid-friendly activities. Can you suggest an itinerary with one rest day in the middle? Show us relevant attractions with images."
print(f"Sample {sample}\n")
prompt_template = generate_prompt(user_query, slots, rag_chunks)
print(f"---------- Prompt Template -------------")
print(prompt_template)
print(f"----------------------------------------")


# ## Generate prompt

# In[62]:


print(f"Sample {sample}\n")
prompt_template = generate_prompt(user_query, slots, rag_chunks)
print(f"---------- Prompt Template -------------")
print(prompt_template)
print(f"----------------------------------------")


# # Generate Response (travel advice)

# In[180]:


import json
import requests

def generate_response(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt},
            stream=True,
            timeout=30
        )
    except requests.RequestException as e:
        print("Ollama request failed:", e)
        return {}

    # Stream and collect response chunks
    travel_advice = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                travel_advice += data.get("response", "")
            except json.JSONDecodeError:
                continue  # Ignore malformed lines

    return travel_advice

travel_advice = generate_response(prompt_template)
print(travel_advice)


# # Generate as module
# 
# jupyter nbconvert --to script prompt_gen.ipynb
