# transliterate_enhanced.py

from tel_transliterate import latin_to_telugu
from tel_vowel_signs import vowel_signs

# Define vowels and consonants
vowels = ['a', 'aa', 'i', 'ii', 'u', 'uu', 'e', 'ee', 'ai', 'o', 'oo', 'au', 'ri']
consonants = [
    'kshn', 'ksh', 'shn', 'kh', 'gh', 'ch', 'jh',
    'th', 'dh', 'ph', 'bh', 'sh',
    'gn', 'tr', 'tth', 'ddh', 'nj',
    'nn', 'tt', 'dd',
    'k', 'g', 'c', 'j',
    't', 'd', 'n', 'p', 'b',
    'm', 'y', 'r', 'l', 'v',
    's', 'h',
]

# Sort consonants and vowels by length in descending order to match longer patterns first
consonants_sorted = sorted(consonants, key=lambda x: -len(x))
vowels_sorted = sorted(vowels, key=lambda x: -len(x))

def transliterate_word_enhanced(latin_word):
    telugu_word = ""
    i = 0
    length = len(latin_word)
    
    while i < length:
        matched = False
        
        # Attempt to match consonant
        for cons in consonants_sorted:
            cons_len = len(cons)
            segment = latin_word[i:i+cons_len].lower()
            if segment == cons:
                telugu_consonant = latin_to_telugu.get(cons)
                if telugu_consonant:
                    i += cons_len
                    # Check if the next segment starts with a consonant
                    is_next_consonant = False
                    for next_cons in consonants_sorted:
                        next_cons_len = len(next_cons)
                        if latin_word[i:i+next_cons_len].lower() == next_cons:
                            is_next_consonant = True
                            break
                    if is_next_consonant:
                        # Append consonant with virama to suppress inherent 'a'
                        telugu_word += telugu_consonant + '్'
                    else:
                        # Attempt to match vowel after consonant
                        vowel_matched = False
                        for vowel in vowels_sorted:
                            vowel_len = len(vowel)
                            vowel_segment = latin_word[i:i+vowel_len].lower()
                            if vowel_segment == vowel:
                                vowel_sign = vowel_signs.get(vowel, '')
                                telugu_word += telugu_consonant + vowel_sign
                                i += vowel_len
                                vowel_matched = True
                                break
                        if not vowel_matched:
                            # No vowel follows; append consonant with inherent 'a'
                            telugu_word += telugu_consonant
                    matched = True
                    break  # Consonant matched; proceed to next character
        if matched:
            continue  # Proceed to next iteration of the while loop
        
        # Attempt to match standalone vowel
        for vowel in vowels_sorted:
            vowel_len = len(vowel)
            vowel_segment = latin_word[i:i+vowel_len].lower()
            if vowel_segment == vowel:
                telugu_vowel = latin_to_telugu.get(vowel)
                if telugu_vowel:
                    telugu_word += telugu_vowel
                    i += vowel_len
                matched = True
                break
        if matched:
            continue  # Proceed to next iteration of the while loop
        
        # If no match found, append the character as is or handle accordingly
        telugu_word += latin_word[i]
        i += 1
    
    return telugu_word
