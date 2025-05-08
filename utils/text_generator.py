# ============================================================================ #
#                         ✍️ TEXT GENERATOR MODULE                            #
# ============================================================================ #
# Filename     : text_generator.py
# Description  : Converts Tenglish to Telugu using a custom transliteration engine
#                and provides word suggestions with user interaction support.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# import necessary libraries
import streamlit as st
import json
import os

# Load mappings.json dynamically
MAPPINGS_PATH = os.path.join('utils', 'mappings.json')
RECENT_WORDS_PATH = os.path.join('utils', 'recent_words.txt')

with open(MAPPINGS_PATH, 'r', encoding='utf-8') as f:
    mappings = json.load(f)

# Prepare transliteration map
translit_map = {}
for section in ['guninthalu', 'combined_othulu', 'same_letter_othulu', 'basic_othulu', 'consonants', 'vowels']:
    for item in mappings.get(section, []):
        translit_map.update(item)

# Sort keys by length to prioritize longer patterns
sorted_keys = sorted(translit_map.keys(), key=lambda x: -len(x))

# Convert special_words list of dicts into a single dictionary
special_words_list = mappings.get('special_words', [])
special_words = {}
for item in special_words_list:
    special_words.update(item)

# Load extended_word_library
extended_word_library = mappings.get('extended_word_library', {})

# Function to load recent words
def load_recent_words():
    if not os.path.exists(RECENT_WORDS_PATH):
        return []
    with open(RECENT_WORDS_PATH, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

# Function to save recent words
def save_recent_word(word):
    words = load_recent_words()
    if word not in words:
        words.insert(0, word)
        words = words[:20]  # Keep only last 20
        with open(RECENT_WORDS_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(words))

# Function to convert Tenglish to Telugu using rule-based transliteration
def rule_based_tenglish_to_telugu(text):
    words = text.split()
    converted = []
    for word in words:
        if word.startswith('{') and word.endswith('}'):
            converted.append(word[1:-1])  # Keep as is
            continue
        lower_word = word.lower()
        if lower_word in special_words:
            telugu_word = special_words[lower_word]
        elif lower_word in extended_word_library:
            telugu_word = extended_word_library[lower_word]
        else:
            temp = lower_word
            for key in sorted_keys:
                temp = temp.replace(key, translit_map[key])
            telugu_word = temp
        converted.append(telugu_word)
        save_recent_word(lower_word)
    return ' '.join(converted)

# Function to render the Streamlit UI for Tenglish to Telugu conversion
def render_text_conversion_ui():
    st.title("📝 Rule-Based Tenglish ➜ Telugu Converter")

    st.markdown("""
        Type your Tenglish text below. Use `{word}` to keep words in English.
        Suggestions will appear based on recent usage and known words.
    """)

    user_input = st.text_input("Enter Tenglish Text", placeholder="Eg: Let's create better tomorrow with{CU} Next")

    suggestions = []
    last_word = user_input.split()[-1] if user_input else ''
    if last_word and not last_word.startswith('{'):
        for word in list(special_words.keys()) + list(extended_word_library.keys()) + load_recent_words():
            if word.startswith(last_word.lower()) and word not in suggestions:
                suggestions.append(word)
            if len(suggestions) >= 5:
                break

    if suggestions:
        st.markdown("**Suggestions:** " + ", ".join(suggestions))

    if st.button("Convert to Telugu"):
        telugu_output = rule_based_tenglish_to_telugu(user_input)
        st.success("Converted Text:")
        st.write(f"**Tenglish:** {user_input}")
        st.write(f"**Telugu:** {telugu_output}")

        if st.button("✅ Save to CSV"):
            save_path = os.path.join('data', 'converted_texts.csv')
            if not os.path.exists(save_path):
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('Tenglish,Telugu\n')
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(f'"{user_input}","{telugu_output}"\n')
            st.success("Saved to converted_texts.csv ✅")
