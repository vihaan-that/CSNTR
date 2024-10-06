import csv
from translit_enhance import transliterate_word_enhanced

input_file = '../Language_detection/telugu_conversion_input.csv'
output_file = 'telugu_terms_transliterated.csv'  # Output file

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = ['Latin', 'Telugu']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        latin_word = row['Latin']
        telugu_word = transliterate_word_enhanced(latin_word)
        writer.writerow({'Latin': latin_word, 'Telugu': telugu_word})

print(f"Transliteration completed. Check '{output_file}' for results.")
