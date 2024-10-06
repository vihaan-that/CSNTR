import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
import string
import pandas as pd

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('words')

# Define vocabulary and punctuation sets
english_vocab = set(w.lower() for w in words.words())
punctuation = set(string.punctuation)

# Function to label words
def label_words(sentence):
    tokens = word_tokenize(sentence)
    labels = []
    for token in tokens:
        
        if token in punctuation:
            labels.append('punct')
            continue
        # (remove punctuation, convert to lowercase)
        word = ''.join(char for char in token if char.isalpha()).lower()
        if word in english_vocab:
            labels.append('en')  
        elif word:  
            labels.append('tel')
        else:
            labels.append('other')  
    return list(zip(tokens, labels))

# Step 1: Read sentences from the input CSV file
input_file = 'input.csv'  
df_input = pd.read_csv(input_file)

# the CSV file has a column named 'sentence' that contains the text
sentences = df_input['sentence'].tolist()

#  Label all sentences
all_labeled = [label_words(sentence) for sentence in sentences]

# Flattening the data for DataFrame
words = []
labels = []
for sentence_labels in all_labeled:
    for word, label in sentence_labels:
        words.append(word)
        labels.append(label)

# Creating output DataFrame with labeled words
df_output = pd.DataFrame({'word': words, 'label': labels})

# Saving the full labeled output to a CSV file
output_file = 'labeled_output.csv'  
df_output.to_csv(output_file, index=False)

print(f"Full labeled data saved to {output_file}")

# Filter for only 'tel' labeled words (Telugu words)
df_telugu = df_output[df_output['label'] == 'tel']

# Saving only the Telugu words to a separate CSV file
telugu_output_file = 'telugu_words.csv'
df_telugu.to_csv(telugu_output_file, index=False)

print(f"Telugu words saved to {telugu_output_file}")

# creating a DataFrame with 'Latin' column
df_for_conversion = pd.DataFrame({
    'Latin': df_telugu['word'],  # Take the 'word' column from the telugu_words.csv file
    'Telugu': [''] * len(df_telugu)  # Empty 'Telugu' column to be filled later
})

# Save this DataFrame as the input CSV file for the conversion project
conversion_input_file = 'telugu_conversion_input.csv'
df_for_conversion.to_csv(conversion_input_file, index=False)

print(f"Conversion input file saved to {conversion_input_file}")
