import streamlit as st
import pandas as pd
import base64
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
from io import BytesIO

# Load the T5 model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Function to read data from the Cornell Movie Dialog corpus
def load_data():
    lines_file = 'data/movie_lines.txt'
    conversations_file = 'data/movie_conversations.txt'

    # Load movie lines
    with open(lines_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Load movie conversations
    with open(conversations_file, 'r', encoding='utf-8', errors='replace') as f:
        conversations = f.readlines()

    return lines, conversations

# Function to generate chatbot responses
def generate_response(user_input):
    input_text = f"dialogue: {user_input} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to generate TTS audio
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

# Streamlit UI
st.title("Generative AI Chatbot with Cornell Dataset")

# Session state to store conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Input from user
user_input = st.text_input("You:")

# Avatar link (replace with an actual avatar image link)
avatar_url = "https://gravatar.com/avatar/2535e9304ade60c67219f0dc07ecd6ba?s=400&d=robohash&r=x"

# Generate response and sentiment analysis when user submits input
if user_input:
    # Generate chatbot response
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

# Adding option to clear conversation
if st.button("Clear Conversation"):
    st.session_state.conversation = []
