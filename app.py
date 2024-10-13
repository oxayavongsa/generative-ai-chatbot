import pandas as pd

# Assuming the dataset is in the same directory as your app.py
def load_data():
    lines_file = 'movie_lines.txt'
    conversations_file = 'movie_conversations.txt'

    # Load the data (adjust the parsing according to your logic)
    with open(lines_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    with open(conversations_file, 'r', encoding='utf-8', errors='replace') as f:
        conversations = f.readlines()

    return lines, conversations

# Load the data when the app starts
lines, conversations = load_data()

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import streamlit as st

# Load the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Use GPU if available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to generate chatbot responses
def generate_response(user_input):
    input_text = f"dialogue: {user_input.strip()} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Generative AI Chatbot")
st.write("This chatbot generates responses based on English dialogue.")

# Input from user
user_input = st.text_input("Ask something in English:")

# Generate response when user submits
if user_input:
    response = generate_response(user_input)
    st.write(f"Bot: {response}")
