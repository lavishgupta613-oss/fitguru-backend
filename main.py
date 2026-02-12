# import os
# import uuid
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate

# import random
# load_dotenv()

# # DEMO_RESPONSES = [
# #     "**Great job**! Keep going, you're/n doing amazing 💪",
# #     "Try increasing your **protein** intake/n today!",
# #     "Let's focus on consistency — **even 10 mins** workout/n matters.",
# #     "Drink plenty of water and stay\n **hydrated**!",
# #     "Nice progress! Want a quick **5-minute** workout\n plan?",
# #     "Remember: Fitness is a journey. You're **improving** every\n day!"
# # ]

# # Load environment variables


# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables")

# class Profile(BaseModel):
#     age: str
#     weight: str
#     height: str
#     goal: str
#     activity: str
#     notes: list
# # Initialize model
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.7,
#     max_output_tokens=64
# )

# # Prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI fitness coach.\n" 
#      "If asked who created you, say you were made by Lavish Gupta.\n"
#      "Give SHORT, concise answers\n "
#      "Maximum 3 bullet points or 3 sentences."
# ),
#     ("placeholder", "{history}"),
#     ("human", "{input}")
# ])

# chain = prompt | llm

# # Memory storage
# store = {}



# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# conversation = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="history"
# )

# # FastAPI app
# app = FastAPI()

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Models
# class ChatRequest(BaseModel):
#     session_id: str
#     message: str

# # ⭐ New Chat Route — Generate Shareable Link
# @app.get("/new-chat")
# def create_new_chat():
#     session_id = str(uuid.uuid4())
#     return {"session_id": session_id}


# # Chat API

#     # reply = random.choice(DEMO_RESPONSES)
#     # return {
#     #     "reply": reply,
#     #     "session_id": body.session_id,
#     #     "debug": "This is a demo mock response. LLM is disabled."
#     # }
    
# @app.get("/")
# def health():
#     return {"status": "FitGuru backend running 🚀"}

    
# @app.post("/chat")
# async def chat_api(body: ChatRequest):
#     # reply = random.choice(DEMO_RESPONSES)
#     # print(reply)
#     # return {
#     #     "reply": reply,
#     #     "session_id": body.session_id,
#     #     "debug": "This is a demo mock response. LLM is disabled."
#     # }
#     response = conversation.invoke(
#         {"input": body.message},
#         config={"configurable": {"session_id": body.session_id}}
#     )
#     print("LLM Response:", response.content)
#     return {"reply": response.content}





import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate

import random
load_dotenv()

# DEMO_RESPONSES = [
#     "**Great job**! Keep going, you're/n doing amazing 💪",
#     "Try increasing your **protein** intake/n today!",
#     "Let's focus on consistency — **even 10 mins** workout/n matters.",
#     "Drink plenty of water and stay\n **hydrated**!",
#     "Nice progress! Want a quick **5-minute** workout\n plan?",
#     "Remember: Fitness is a journey. You're **improving** every\n day!"
# ]

# Load environment variables


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

from typing import List, Dict

class Profile(BaseModel):
    age: int | None = None
    weight: int | None = None
    height: int | None = None
    goal: str | None = None
    activity: str | None = None
    notes: List[Dict] = []

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=64
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI fitness coach.\n"
     "Use the user's profile to personalize advice.\n\n"
     "User Profile:\n"
     "Age: {age}\n"
     "Weight: {weight}\n"
     "Height: {height}\n"
     "Goal: {goal}\n"
     "Activity Level: {activity}\n"
     "Notes: {notes}\n\n"
     "Rules:\n"
     "- Give short, concise answers\n"
     "- Max 3 bullet points or sentences\n"
    ),
    ("placeholder", "{history}"),
    ("human", "{input}")
])


chain = prompt | llm

# Memory storage
store = {}




def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


current_profile = None

# Models
class ChatRequest(BaseModel):
    session_id: str
    message: str

# ⭐ New Chat Route — Generate Shareable Link
@app.get("/new-chat")
def create_new_chat():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


# Chat API

    # reply = random.choice(DEMO_RESPONSES)
    # return {
    #     "reply": reply,
    #     "session_id": body.session_id,
    #     "debug": "This is a demo mock response. LLM is disabled."
    # }
    
@app.get("/")
def health():
    return {"status": "FitGuru backend running 🚀"}

@app.post("/profile")
async def save_profile(profile: Profile):
    global current_profile
    current_profile = profile
    print("Profile saved:", current_profile)
    return {"message": "Profile saved successfully"}


    
@app.post("/chat")
async def chat_api(body: ChatRequest):
    if current_profile:
        profile_context = {
            "age": current_profile.age or "unknown",
            "weight": current_profile.weight or "unknown",
            "height": current_profile.height or "unknown",
            "goal": current_profile.goal or "unknown",
            "activity": current_profile.activity or "unknown",
            "notes": ", ".join([n.get("text", "") for n in current_profile.notes])
        }
    else:
        profile_context = {
            "age": "unknown",
            "weight": "unknown",
            "height": "unknown",
            "goal": "unknown",
            "activity": "unknown",
            "notes": "none"
        }

    response = conversation.invoke(
        {
            "input": body.message,
            **profile_context
        },
        {
            "configurable": {
                "session_id": body.session_id
            }
        }
    )

    return {"reply": response.content}
