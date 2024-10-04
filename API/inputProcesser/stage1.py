import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import os
import csv
import logging

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('words')

def label_words_in_sentences(input_csv, output_labeled_csv, output_telugu_csv, conversion_input_csv):
    """
    Reads sentences from input_csv, labels each word, and saves the labeled data.

    Args:
        input_csv (str): Path to the input CSV file containing sentences.
        output_labeled_csv (str): Path to save the full labeled output.
        output_telugu_csv (str): Path to save only Telugu labeled words.
        conversion_input_csv (str): Path to save the conversion input CSV.
    """
    # Define English vocabulary and punctuation
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
            # Normalize the word (remove non-alphabetic characters and convert to lowercase)
            word = ''.join(char for char in token if char.isalpha()).lower()
            if word in english_vocab:
                labels.append('en')  # English
            elif word:  # Non-empty and not in English vocab
                labels.append('tel')  # Telugu or other non-English
            else:
                labels.append('other')  # Unknown tokens
        return list(zip(tokens, labels))

    # Step 1: Read sentences from the input CSV file
    df_input = pd.read_csv(input_csv)
    if 'sentence' not in df_input.columns:
        raise ValueError("Input CSV must contain a 'sentence' column.")
    sentences = df_input['sentence'].tolist()
    logging.info("Loaded {len(sentences)} sentences from '{input_csv}'.")

    # Step 2: Label all sentences
    all_labeled = [label_words(sentence) for sentence in sentences]
    logging.info("Completed labeling of all sentences.")

    # Step 3: Flatten the data for DataFrame
    words_list = []
    labels_list = []
    for sentence_labels in all_labeled:
        for word, label in sentence_labels:
            words_list.append(word)
            labels_list.append(label)

    # Step 4: Creating output DataFrame with labeled words
    df_output = pd.DataFrame({'word': words_list, 'label': labels_list})

    # Step 5: Saving the full labeled output to a CSV file
    df_output.to_csv(output_labeled_csv, index=False)
    logging.info("Full labeled data saved to '{output_labeled_csv}'.")

    # Step 6: Filter for only 'tel' labeled words (Telugu words)
    df_telugu = df_output[df_output['label'] == 'tel']
    logging.info("Filtered {len(df_telugu)} Telugu words.")

    # Step 7: Saving only the Telugu words to a separate CSV file
    df_telugu.to_csv(output_telugu_csv, index=False)
    logging.info("Telugu words saved to '{output_telugu_csv}'.")

    # Step 8: Prepare the data for the next project by creating a DataFrame with 'Latin' and empty 'Telugu' columns
    df_for_conversion = pd.DataFrame({
        'Latin': df_telugu['word'],
        'Telugu': [''] * len(df_telugu)
    })

    # Step 9: Save this DataFrame as the input CSV file for the conversion project
    df_for_conversion.to_csv(conversion_input_csv, index=False)
    logging.info("Conversion input file saved to '{conversion_input_csv}'.")

def transliterate_telugu_words(conversion_input_csv, transliterated_output_csv):
    """
    Reads Latin-scripted Telugu words from conversion_input_csv, transliterates them,
    and saves the results to transliterated_output_csv.

    Args:
        conversion_input_csv (str): Path to the input CSV file with Latin words.
        transliterated_output_csv (str): Path to save the transliterated Telugu words.
    """
    # Import the transliteration function
    try:
        from translit_enhance import transliterate_word_enhanced
    except ImportError:
        raise ImportError("Module 'translit_enhance' not found. Ensure it is installed and accessible.")

    input_path = conversion_input_csv
    output_path = transliterated_output_csv

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['Latin', 'Telugu']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            latin_word = row['Latin']
            telugu_word = transliterate_word_enhanced(latin_word)
            writer.writerow({'Latin': latin_word, 'Telugu': telugu_word})
    
    logging.info("Transliteration completed. Check '{transliterated_output_csv}' for results.")

def replace_transliterated_words(original_input_csv, transliteration_csv, final_output_csv):
    """
    Replaces Latin-scripted Telugu words in the original sentences with their Telugu script equivalents.

    Args:
        original_input_csv (str): Path to the original input CSV file containing sentences.
        transliteration_csv (str): Path to the CSV file with transliterated Telugu words.
        final_output_csv (str): Path to save the final modified sentences.
    """
    # Ensure NLTK data is downloaded
    nltk.download('punkt')

    # Step 1: Read the original input sentences from the input CSV file
    df_input = pd.read_csv(original_input_csv)
    if 'sentence' not in df_input.columns:
        raise ValueError("Original input CSV must contain a 'sentence' column.")
    sentences = df_input['sentence'].tolist()
    logging.info("Loaded {len(sentences)} sentences from '{original_input_csv}'.")

    # Step 2: Read the Latin to Telugu conversion mapping from the transliteration output CSV file
    df_conversion = pd.read_csv(transliteration_csv)
    if 'Latin' not in df_conversion.columns or 'Telugu' not in df_conversion.columns:
        raise ValueError("Transliteration CSV must contain 'Latin' and 'Telugu' columns.")
    logging.info("Loaded {len(df_conversion)} transliteration mappings from '{transliteration_csv}'.")

    # Step 3: Create a dictionary mapping from Latin-scripted words to their corresponding Telugu script
    # Convert keys to lowercase to ensure case-insensitive matching
    latin_to_telugu = {latin.lower(): telugu for latin, telugu in zip(df_conversion['Latin'], df_conversion['Telugu'])}
    logging.info("Created Latin to Telugu mapping dictionary.")

    # Step 4: Function to replace Latin-scripted Telugu words with Telugu script
    def replace_latin_with_telugu(sentence, mapping):
        tokens = word_tokenize(sentence)
        modified_tokens = [
            mapping.get(token.lower(), token)  # Replace if in mapping; else keep original
            for token in tokens
        ]
        return ' '.join(modified_tokens)

    # Step 5: Process each sentence
    modified_sentences = [replace_latin_with_telugu(sentence, latin_to_telugu) for sentence in sentences]
    logging.info("Completed replacing transliterated words in all sentences.")

    # Step 6: Save the final modified sentences into a new CSV file
    df_final_output = pd.DataFrame({'sentence': modified_sentences})
    df_final_output.to_csv(final_output_csv, index=False)
    logging.info("Final sentences saved to '{final_output_csv}'.")

def main():
    """
    Main function to execute the language detection, transliteration, and replacement process.
    """
    # Define file paths
    base_dir = os.getcwd()  # You can set this to any directory you prefer
    input_csv = os.path.join(base_dir, 'input.csv')  # Original input CSV with sentences
    labeled_output_csv = os.path.join(base_dir, 'labeled_output.csv')  # Full labeled words
    telugu_words_csv = os.path.join(base_dir, 'telugu_words.csv')  # Only Telugu words
    conversion_input_csv = os.path.join(base_dir, 'telugu_conversion_input.csv')  # Input for transliteration
    transliterated_output_csv = os.path.join(base_dir, 'telugu_terms_transliterated.csv')  # Transliteration output
    final_output_csv = os.path.join(base_dir, 'final_output.csv')  # Final sentences with Telugu script

    # Check if input files exist
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found. Please ensure the file exists.")

    # Stage 1: Language Detection and Labeling
    label_words_in_sentences(
        input_csv=input_csv,
        output_labeled_csv=labeled_output_csv,
        output_telugu_csv=telugu_words_csv,
        conversion_input_csv=conversion_input_csv
    )

    # Stage 2: Transliteration of Telugu Words
    # Check if the conversion input file exists
    if not os.path.isfile(conversion_input_csv):
        raise FileNotFoundError(f"Conversion input file '{conversion_input_csv}' not found.")
    
    transliterate_telugu_words(
        conversion_input_csv=conversion_input_csv,
        transliterated_output_csv=transliterated_output_csv
    )

    # Stage 3: Replacing Transliteration in Original Sentences
    # Check if the transliteration output file exists
    if not os.path.isfile(transliterated_output_csv):
        raise FileNotFoundError(f"Transliteration output file '{transliterated_output_csv}' not found.")
    
    replace_transliterated_words(
        original_input_csv=input_csv,
        transliteration_csv=transliterated_output_csv,
        final_output_csv=final_output_csv
    )

    logging.info("All stages completed successfully.")

if __name__ == "__main__":
    main()
