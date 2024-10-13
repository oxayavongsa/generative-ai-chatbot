from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Import necessary libraries
import os
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
import base64
from io import BytesIO
import torch
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Download dataset using Kaggle API
def download_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('rajathmc/cornell-moviedialog-corpus', 'cornell-movie-dialogs-corpus.zip', path='data/')
    with zipfile.ZipFile('data/cornell-movie-dialogs-corpus.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')
    print("Downloaded dataset from Kaggle successfully!")

# Download the dataset (remove if already downloaded)
download_dataset()

# Path to the extracted dataset files
lines_file = 'data/cornell movie-dialogs-corpus/movie_lines.txt'
conversations_file = 'data/cornell movie-dialogs-corpus/movie_conversations.txt'

# Define the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to generate chatbot responses
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
    avatar_html = f"""
    <img src="{avatar_url}" alt="Avatar" width="150" height="150">
    """
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mpeg;base64,{base64.b64encode(audio_fp.read()).decode('utf-8')}" type="audio/mpeg">
    </audio>
    """
    st.markdown(avatar_html, unsafe_allow_html=True)
    st.markdown(audio_html, unsafe_allow_html=True)

# Streamlit app code
def app():
    st.title("Generative AI Chatbot with Cornell Dataset")

    # Session state to store conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # User input for the chatbot
    user_input = st.text_input("You:")

    # Avatar link (replace with an actual avatar image link)
    avatar_url = "https://drive.google.com/uc?id=1X3dYj0dgdtu-updhhfJA4KIRX_QN96mi"

    # Generate chatbot response when the user submits input
    if user_input:
        response = generate_response(user_input)
        
        # Append conversation history
        st.session_state.conversation.append({"user": user_input, "bot": response})
        
        # Generate TTS audio for the bot response
        audio_fp = generate_audio(response)

        # Display avatar and play audio
        display_avatar_and_audio(avatar_url, audio_fp)

    # Display conversation history
    for chat in st.session_state.conversation:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

    # Button to clear conversation history
    if st.button("Clear Conversation"):
        st.session_state.conversation = []

# Launch the Streamlit app
if __name__ == '__main__':
    app()
