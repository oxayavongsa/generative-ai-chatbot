# Import necessary libraries
import os
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
import base64
from io import BytesIO
import torch
import zipfile
import requests

# Function to download file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Download the dataset from Google Drive
file_id = '1SRKwsK00pEBBezUA5zGM8So6kay4i9Qs'  # Your file ID from Google Drive
destination = 'data/cornell-movie-dialogs-corpus.zip'
download_file_from_google_drive(file_id, destination)

# Unzip the dataset
with zipfile.ZipFile(destination, 'r') as zip_ref:
    zip_ref.extractall('data/')

# Paths to the extracted files
lines_file = 'data/cornell movie-dialogs-corpus/movie_lines.txt'
conversations_file = 'data/cornell movie-dialogs-corpus/movie_conversations.txt'

# Define the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to generate chatbot responses in English
def generate_response(user_input):
    input_text = f"dialogue: {user_input.strip()} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to generate TTS audio for chatbot response
def generate_audio(text):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# Function to display avatar and play audio
def display_avatar_and_audio(avatar_url, audio_fp):
    avatar_html = f'<img src="{avatar_url}" alt="Avatar" width="150" height="150">'
    audio_html = f'<audio autoplay><source src="data:audio/mpeg;base64,{base64.b64encode(audio_fp.read()).decode("utf-8")}" type="audio/mpeg"></audio>'
    st.markdown(avatar_html, unsafe_allow_html=True)
    st.markdown(audio_html, unsafe_allow_html=True)

# Streamlit app code
def app():
    st.title("Generative AI Chatbot with Cornell Dataset")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    user_input = st.text_input("You:")
    avatar_url = "https://raw.githubusercontent.com/oxayavongsa/generative-ai-chatbot/main/images/Man%20Avatar.png"

    if user_input:
        response = generate_response(user_input)
        st.session_state.conversation.append({"user": user_input, "bot": response})
        audio_fp = generate_audio(response)
        display_avatar_and_audio(avatar_url, audio_fp)

    for chat in st.session_state.conversation:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

    if st.button("Clear Conversation"):
        st.session_state.conversation = []

if __name__ == '__main__':
    app()
