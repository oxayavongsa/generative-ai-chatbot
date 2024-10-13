import os
import zipfile
import pandas as pd
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up Kaggle API credentials directly from environment variables
os.environ['KAGGLE_USERNAME'] = 'outhaixayavongsa'  # replace with your username
os.environ['KAGGLE_KEY'] = '013bebdbf0776ed704f846ef0b3b3381'  # replace with your API key

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Define paths
dataset_path = 'data/cornell-movie-dialog-corpus.zip'
lines_file = 'data/cornell movie-dialog-corpus/movie_lines.txt'
conversations_file = 'data/cornell movie-dialog-corpus/movie_conversations.txt'

# Download the Cornell Movie Dialog Corpus dataset
if not os.path.exists(dataset_path):
    st.write("Downloading dataset from Kaggle...")
    api.dataset_download_files('rajathmc/cornell-moviedialog-corpus', path='data/', unzip=False)

# Unzip the dataset if not already done
if not os.path.exists(lines_file) or not os.path.exists(conversations_file):
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall('data/')
    st.write("Dataset extracted.")

# Function to load and parse movie_lines.txt
def parse_movie_lines(lines_file):
    lines = {}
    with open(lines_file, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                line_id = parts[0]
                text = parts[4]
                lines[line_id] = text
    return lines

# Function to load and parse movie_conversations.txt
def parse_movie_conversations(conversations_file):
    conversations = []
    with open(conversations_file, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 4:
                line_ids = eval(parts[3])  # This is a list of line IDs in a conversation
                conversations.append(line_ids)
    return conversations

# Load the movie lines and conversations data
movie_lines = parse_movie_lines(lines_file)
movie_conversations = parse_movie_conversations(conversations_file)

# Function to create dialog pairs from movie lines and conversations
def create_dialog_pairs(conversations, lines):
    dialog_pairs = []
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            input_line = lines.get(conversation[i], "")
            response_line = lines.get(conversation[i + 1], "")
            if input_line and response_line:
                dialog_pairs.append((input_line, response_line))
    return dialog_pairs

# Create dialog pairs
dialog_pairs = create_dialog_pairs(movie_conversations, movie_lines)

# Convert dialog pairs to a DataFrame
dialog_df = pd.DataFrame(dialog_pairs, columns=['input', 'response'])

# Streamlit UI
st.title("Generative AI Chatbot with Cornell Dataset")

# Display the first few rows of dialog pairs
st.write(dialog_df.head())

# Optional: Functionality to display more detailed analysis or to work with the dialog_df can be added below
