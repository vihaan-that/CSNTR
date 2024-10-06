import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

# Read the original input sentences 
input_file = r'Z:\python\NLP\Language_detection\input.csv'  
df_input = pd.read_csv(input_file)

# Assuming the CSV file has a column named 'sentence' that contains the text
sentences = df_input['sentence'].tolist()

# Read the Latin to Telugu conversion mapping from the conversion output CSV file
conversion_output_file = r'Z:\python\NLP\latintel_to_originaltel\telugu_terms_transliterated.csv'  
df_conversion = pd.read_csv(conversion_output_file)

# Create a dictionary mapping from Latin-scripted words to their corresponding Telugu script
# Convert keys to lowercase to ensure case-insensitive matching
latin_to_telugu = {latin.lower(): telugu for latin, telugu in zip(df_conversion['Latin'], df_conversion['Telugu'])}

# Replace Latin-scripted Telugu words in each sentence with their corresponding Telugu script
def replace_latin_with_telugu(sentence, latin_to_telugu):
    # Tokenize the sentence and look for Latin-scripted Telugu words
    tokens = word_tokenize(sentence)
    
    # For each token, if it exists in the Latin-to-Telugu dictionary, replace it with its Telugu script equivalent
    modified_tokens = [
        latin_to_telugu.get(token.lower(), token)  # Replace token if it's in the dictionary, case-insensitive
        for token in tokens
    ]
    
    # Join the modified tokens back to sentence
    return ' '.join(modified_tokens)

modified_sentences = [replace_latin_with_telugu(sentence, latin_to_telugu) for sentence in sentences]

# Final output file
df_final_output = pd.DataFrame({'sentence': modified_sentences})
final_output_file = 'final_output.csv'  
df_final_output.to_csv(final_output_file, index=False)

print(f"Final sentences saved to {final_output_file}")
