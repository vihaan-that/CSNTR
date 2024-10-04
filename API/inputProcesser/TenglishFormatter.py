import logging
import os
import pandas as pd

# Ensure the functions from stage1.py are available
from stage1 import (
    label_words_in_sentences,
    transliterate_telugu_words,
    replace_transliterated_words
)

def process_user_input(user_sentence):
    """
    Takes a user input sentence, processes it through the transliteration pipeline, and returns the final output.

    Args:
        user_sentence (str): The input sentence from the user.

    Returns:
        str: The processed sentence with Telugu words transliterated into Telugu script.
    """
    # Define file paths
    base_dir = os.getcwd()  # Current working directory
    input_csv = os.path.join(base_dir, 'input.csv')  # Original input CSV with sentences
    labeled_output_csv = os.path.join(base_dir, 'labeled_output.csv')  # Full labeled words
    telugu_words_csv = os.path.join(base_dir, 'telugu_words.csv')  # Only Telugu words
    conversion_input_csv = os.path.join(base_dir, 'telugu_conversion_input.csv')  # Input for transliteration
    transliterated_output_csv = os.path.join(base_dir, 'telugu_terms_transliterated.csv')  # Transliteration output
    final_output_csv = os.path.join(base_dir, 'final_output.csv')  # Final sentences with Telugu script

    # Step 1: Write the user input to the input.csv file
    input_data = pd.DataFrame({'sentence': [user_sentence]})
    input_data.to_csv(input_csv, index=False)
    logging.info("User input written to '{input_csv}'.")

    # Step 2: Call the language detection and labeling function
    label_words_in_sentences(
        input_csv=input_csv,
        output_labeled_csv=labeled_output_csv,
        output_telugu_csv=telugu_words_csv,
        conversion_input_csv=conversion_input_csv
    )

    # Step 3: Call the transliteration function
    transliterate_telugu_words(
        conversion_input_csv=conversion_input_csv,
        transliterated_output_csv=transliterated_output_csv
    )

    # Step 4: Call the replacement function to update the original sentence
    replace_transliterated_words(
        original_input_csv=input_csv,
        transliteration_csv=transliterated_output_csv,
        final_output_csv=final_output_csv
    )

    # Step 5: Read the final output and return the modified sentence
    df_final_output = pd.read_csv(final_output_csv)
    final_sentence = df_final_output['sentence'].iloc[0]
    
    logging.info(f"Final sentence: {final_sentence}")
    return final_sentence

# Example usage
user_sentence = "nenu oka katha chadivanu"  # Example Latin-scripted Telugu sentence
output_sentence = process_user_input(user_sentence)
logging.info("Processed Sentence: {output_sentence}")
