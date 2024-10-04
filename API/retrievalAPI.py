from dotenv import load_dotenv
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(os.path.abspath('../API/inputProcesser'))


# Now import the function
from TenglishFormatter import process_user_input

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Initialize the multilingual tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

load_dotenv()
NOTES_DIRECTORY = os.getenv("NOTES_DIRECTORY")
EMBEDDINGS_DIRECTORY = os.getenv("EMBEDDINGS_DIRECTORY")
# Path where embeddings are stored


# Function to compute the embedding of a document or query
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

    return embeddings.numpy()

# Function to load embeddings from the .npy files
def load_embeddings():
    embeddings = {}
    
    # Iterate through all the files in the embeddings directory
    for filename in os.listdir(EMBEDDINGS_DIRECTORY):
        if filename.endswith('_embedding.npy'):
            file_path = os.path.join(EMBEDDINGS_DIRECTORY, filename)
            document_name = filename.replace('_embedding.npy', '')
            
            # Load the embedding and store it
            embeddings[document_name] = np.load(file_path)

    return embeddings

# Function to find the top 3 most similar documents based on a query
def find(query):
    
    processed_query = process_user_input(query)
    
    # Step 2: Compute the embedding for the processed query
    query_embedding = compute_embedding(processed_query)
    
    # Step 2: Load all pre-computed embeddings
    document_embeddings = load_embeddings()

    # Step 3: Calculate cosine similarity between the query and all documents
    similarity_scores = {}
    for doc_name, doc_embedding in document_embeddings.items():
        similarity = cosine_similarity(doc_embedding, query_embedding)[0][0]
        similarity_scores[doc_name] = similarity

    # Step 4: Sort the documents by similarity scores in descending order
    sorted_docs = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)

    # Step 5: Retrieve the top 3 matching documents
    top_3_docs = sorted_docs[:3]

    # Step 6: Retrieve the content of the top 3 documents
    results = []
    for doc_name, similarity in top_3_docs:
        # Read the document content from the corresponding note file
        note_path = os.path.join(NOTES_DIRECTORY, f"{doc_name}.txt")
        if os.path.exists(note_path):
            with open(note_path, 'r', encoding='utf-8') as file:
                content = file.read()
                results.append((doc_name, similarity, content))
        else:
            print(f"Note file for '{doc_name}' not found.")
    
    return results

# Example usage
if __name__ == "__main__":
    query_text = "Ravi doctor అవ్వాలని అనుకున్నాడు, కానీ Arun?"
    results = find(query_text)
    
    # Print the top 3 documents
    print(f"\nTop 3 results for the query: {query_text}")
    for idx, (doc_name, similarity, content) in enumerate(results):
        print(f"\nRank {idx + 1}:\nDocument: {doc_name}\nSimilarity: {similarity}\nContent:\n{content}\n")
