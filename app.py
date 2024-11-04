import numpy as np
import pickle
import streamlit as st
import os
import string
from sklearn.feature_extraction import DictVectorizer
import requests
import nltk
from nltk.data import find
import os
from dotenv import load_dotenv
import random

load_dotenv()


def download_nltk_data():
    # Check if 'punkt_tab' is already downloaded
    if not os.path.exists(os.path.join(nltk.data.find('tokenizers'), 'punkt_tab')):
        nltk.download('punkt_tab')


download_nltk_data()


from nltk.tokenize import word_tokenize
from nltk.tree import Tree

# Define suffixes
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er",
               "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness",
               "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish",
              "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]
punct = set(string.punctuation)
# Set random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

def word_features_test(sentence, i):
    word = sentence[i]

    prevword = sentence[i-1] if i > 0 else '<START>'

    nextword = sentence[i+1] if i < len(sentence)-1 else '<END>'
    
    features = {
        'word': word,
        'is_numeric': int(word.isdigit()),
        'contains_number': int(any(char.isdigit() for char in word)),
        'is_punctuation': int(any(char in punct for char in word)),
        'has_noun_suffix': int(any(word.endswith(suffix) for suffix in noun_suffix)),
        'has_verb_suffix': int(any(word.endswith(suffix) for suffix in verb_suffix)),
        'has_adj_suffix': int(any(word.endswith(suffix) for suffix in adj_suffix)),
        'has_adv_suffix': int(any(word.endswith(suffix) for suffix in adv_suffix)),

        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
        'prevword': prevword,
        'nextword': nextword,

        'is_capitalized': int(word[0].isupper()),
        'is_prec_capitalized': int(sentence[i-1][0].isupper() if i > 0 else 0),
        'is_next_capitalized': int(sentence[i+1][0].isupper() if i < len(sentence)-1 else 0),

        'is_all_caps': int(word.isupper()),
        'is_all_lower': int(word.islower()),

        'word_length': len(word),
        'is_first': int(i == 0),
    }
    return features


def tokenize_sentence(sentence):
    # return NISTTokenizer.tokenize(sentence)
    # return word_tokenize(sentence)
    return sentence.split() 


def predict_ner_tags(sentence):
    # Load the saved components
    vectorizer = DictVectorizer(sparse=True)
    with open('./params/dict_vectorizer_unk_final2.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('./params/best_ner_model_unk_final2.pkl', 'rb') as f:
        model = pickle.load(f)
        
    sentence = vectorizer.transform(sentence)
    pred = model.predict(sentence)
    return pred

    
def gpt4_ner(raw_text):
    api_key = os.getenv("OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",  # Use the loaded API key
        "Content-Type": "application/json"
    }
    # Create the few-shot prompt with examples
    few_shot_prompt = f"""
    your task is to do NEI(named entity identification), your output will be compared to SVM trained on the dataset CoNLL-2003 dataset.you have been provided with few shot examples, your output should be in the same format as given in the examples. There are only two labels to be assigned to the entities in the  input text, '1' if it's a named entity or '0' if it's not a named entity. Also consider the '.' at the end of the sentence to be tagged as '0' and give the right whitespace.
    Input- Raw Text: Delhi is the capital of India.
    Output- NEI markings: Delhi_1 is_0 the_0 capital_0 of_0 India_1 ._0
    
    Input- Raw Text: Washington DC is the capital of the United States of America.
    Output- NEI markings: Washington_1 DC_1 is_0 the_0 capital_0 of_0 United_1 States_1 of_1 America_1 ._0
    
    Input- Raw Text: {raw_text}
    Output- NEI markings:
    """
    
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": few_shot_prompt
            }
        ],
        "max_tokens": 200
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        response_json = response.json()
        gpt_output = response_json['choices'][0]['message']['content']
        return gpt_output.strip()  # Remove any leading or trailing whitespace
    else:
        return "Error: " + response.text


# Streamlit UI
st.title("Named Entity Identification Demo")


raw_text = st.text_area("Input Raw Text", value="", height=50)

if st.button("Run"):
    if raw_text:
        tokens = tokenize_sentence(raw_text)
        sentence_features = [word_features_test(tokens, i) for i in range(len(tokens))]
        predicted_NEI = predict_ner_tags(sentence_features)
        output = " ".join([f"{word}_{tag}" for word, tag in zip(tokens, predicted_NEI)])
       
        # Get results from GPT-4
        gpt4_output = gpt4_ner(raw_text)

        # Display results
        st.subheader("SVM NEI Task Output:")
        st.write(output)

        st.subheader("GPT-4 NEI Task Output:")
        st.write(gpt4_output)
    else:
        st.warning("Please enter some text to analyze.")

# Run the app using the command: streamlit run your_script.py
