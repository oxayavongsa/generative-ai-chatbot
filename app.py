import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import streamlit as st

# Function to load and preprocess data
def load_data():
    lines_file = 'movie_lines.txt'
    conversations_file = 'movie_conversations.txt'
    
    # Load the dataset (adjust parsing according to your needs)
    lines, conversations = [], []
    with open(lines_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    with open(conversations_file, 'r', encoding='utf-8', errors='replace') as f:
        conversations = f.readlines()
    
    # You can now return the parsed data as needed
    return lines, conversations

# Load data
lines, conversations = load_data()

# Set up T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Set up device
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

# Input from user
user_input = st.text_input("You:")

# Generate response when user submits
if user_input:
    response = generate_response(user_input)
    st.write(f"Bot: {response}")
