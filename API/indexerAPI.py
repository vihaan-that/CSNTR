
import os
import torch
import random
import numpy as np
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(os.path.abspath('../API/inputProcesser'))


# Now import the function
from TenglishFormatter import process_user_input

# Load environment variables from .env file
load_dotenv()

# Get the common directories from the environment variables
NOTES_DIRECTORY = os.getenv("NOTES_DIRECTORY")
EMBEDDINGS_DIRECTORY = os.getenv("EMBEDDINGS_DIRECTORY")

# Ensure both directories are set and exist
if not NOTES_DIRECTORY:
    raise ValueError("NOTES_DIRECTORY is not set in the .env file")
if not os.path.exists(NOTES_DIRECTORY):
    os.makedirs(NOTES_DIRECTORY)

if not EMBEDDINGS_DIRECTORY:
    raise ValueError("EMBEDDINGS_DIRECTORY is not set in the .env file")
if not os.path.exists(EMBEDDINGS_DIRECTORY):
    os.makedirs(EMBEDDINGS_DIRECTORY)

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Initialize the multilingual tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Function to create a note
def createNote(fileName):
    # Combine the path and file name to create the full file path
    file_path = os.path.join(NOTES_DIRECTORY, f"{fileName}.txt")

    # Check if the file already exists
    if os.path.exists(file_path):
        raise FileExistsError(f"The file '{fileName}.txt' already exists.")

    # Create an empty file
    with open(file_path, 'w') as file:
        file.write("")  # Writes an empty string to the file

    print(f"File '{fileName}.txt' created at: {file_path}")

# Function to edit a note by adding text
def editNote(fileName, inputText):
    # Combine the path and file name to get the file path
    file_path = os.path.join(NOTES_DIRECTORY, f"{fileName}.txt")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{fileName}.txt' does not exist.")
    processed_text = process_user_input(inputText)
    # Overwrite the existing content with new text
    with open(file_path, 'w') as file:
        file.write(processed_text)  # Writes the input text to the file

    print(f"File '{fileName}.txt' edited with new content.")

    # Delete the old embedding (if it exists) and re-index the note
    delete_embedding(fileName)
    index(fileName)

# Function to delete a note and its embedding
def deleteNote(fileName):
    # Combine the path and file name to get the file path
    file_path = os.path.join(NOTES_DIRECTORY, f"{fileName}.txt")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{fileName}.txt' does not exist.")

    # Delete the file
    os.remove(file_path)
    print(f"File '{fileName}.txt' deleted.")

    # Delete the embedding associated with this note
    delete_embedding(fileName)

# Function to delete the embedding of a note
def delete_embedding(fileName):
    # Path to the embedding file
    embedding_file_path = os.path.join(EMBEDDINGS_DIRECTORY, f"{fileName}_embedding.npy")

    # Check if the embedding file exists
    if os.path.exists(embedding_file_path):
        # Delete the embedding file
        os.remove(embedding_file_path)
        print(f"Embedding for '{fileName}.txt' deleted.")
    else:
        print(f"No embedding found for '{fileName}.txt' to delete.")

# Function to compute the embedding of text and return it
def compute_embedding(text):
    # Tokenize and encode the text
    encoding = tokenizer.batch_encode_plus(
        [text],
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Generate embeddings
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling along sequence length

    return embeddings

# Function to index the file data and store its embeddings
def index(fileName):
    # Combine the path and file name to create the full file path
    file_path = os.path.join(NOTES_DIRECTORY, f"{fileName}.txt")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{fileName}.txt' does not exist.")

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Compute the embedding of the file content
    embedding = compute_embedding(file_content).numpy()

    # Path to save the embedding
    embedding_file_path = os.path.join(EMBEDDINGS_DIRECTORY, f"{fileName}_embedding.npy")

    # Save the embedding as a .npy file
    np.save(embedding_file_path, embedding)
    
    print(f"Embedding for '{fileName}.txt' saved at: {embedding_file_path}")

# Example usage:
# createNote("myNote")
# editNote("myNote", "Some updated text content")
# index("myNote")
# deleteNote("myNote")
