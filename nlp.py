import nltk
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
# Set NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', quiet=True)

lemmatizer = WordNetLemmatizer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

def normalize_text(sentence):
    normalization_dict = {
        "hiii": "hi", "hii": "hi", "hello": "hi", "hey": "hi", "hiya": "hi",
    }
    words = sentence.split()
    normalized_words = [normalization_dict.get(word.lower(), word) for word in words]
    return " ".join(normalized_words)

def bag_of_words(tokenized_sentence, words):
    sentence_words = [lemmatize(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

