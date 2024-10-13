# Import necessary libraries
import os
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
import base64
from io import BytesIO
import torch
import zipfile
import gdown
from kaggle.api.kaggle_api_extended import KaggleApi

# Function to download and extract the dataset
def download_dataset():
    # Set your Kaggle credentials in environment variables (ensure they are set securely)
    os.environ['KAGGLE_USERNAME'] = 'outhaixayavongsa'
    os.environ['KAGGLE_KEY'] = '013bebdbf0776ed704f846ef0b3b3381'

    # Function to download dataset from Kaggle
    def download_from_kaggle():
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_download_files('rajathmc/cornell-moviedialog-corpus', path='data', unzip=True)
        st.write("Downloaded dataset from Kaggle successfully!")
        return True
    except Exception as e:
        st.write("Kaggle download failed: ", e)
        return False
        
# Function to download dataset from Google Drive
def download_from_gdrive():
    url = 'https://drive.google.com/uc?id=1SRKwsK00pEBBezUA5zGM8So6kay4i9Qs'
    output = 'data/cornell-movie-dialogs-corpus.zip'
    gdown.download(url, output, quiet=False)

    # Unzipping the dataset
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('data/')
    st.write("Downloaded dataset from Google Drive successfully!")

# Attempt to download the dataset from Kaggle first
if not os.path.exists('data/movie_lines.txt'):
    if not download_from_kaggle():
        # If Kaggle fails, download from Google Drive
        download_from_gdrive()

# Now you can use the extracted files
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
    avatar_url = "https://your-avatar-url.com"  # Replace with the actual avatar image URL

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
