import os
import json
import google.generativeai as genai


working_directory = os.path.dirname(os.path.abspath(__file__))

config_file_path = f"{working_directory}/config.json"

print(config_file_path)

config_data = json.load(open(config_file_path))

# Loading the api key

GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# Configuring google.generative ai with API key

genai.configure(api_key=GOOGLE_API_KEY)

# FUNCTION TO LOAD GEMINI PRO FOR CHATBOT

def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model

# FUNCTION TO LOAD FOR IMAGE CAPTIONING

def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_pro_vision_model.generate_content([prompt,image])
    result = response.text
    return result

# FUNCTION TO EMBED TEXT

def embedding_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,content=input_text, task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list

# FUNCTION TO GET A RESPONSE FROM GEMINI PRO

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result
