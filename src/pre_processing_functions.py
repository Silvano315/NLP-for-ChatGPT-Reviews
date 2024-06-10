import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import emoji

slang_dict = {
    "lol": "laugh out loud",
    "gg": "good game",
    "brb": "be right back",
    "idk": "i don't know",
    "omg": "oh my god",
    "btw": "by the way",
    "imo": "in my opinion",
    "fyi": "for your information",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "tbt": "throwback thursday",
    "nvm": "never mind",
    "np": "no problem",
    "jk": "just kidding",
    "irl": "in real life",
    "dm": "direct message",
    "ftw": "for the win",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "bff": "best friends forever",
    "afk": "away from keyboard",
    "afaik": "as far as I know",
    "gr8": "great",
    "b4": "before",
    "wth": "what the hell"
}

# function to replace abbreviations
def replace_slang(text):

    words = text.split()

    new_words = []
    for word in words:
        if word.lower() in slang_dict:
            new_words.append(slang_dict[word.lower()])
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Function to handle negative sentences
def handle_negations(text):
    words = text.split()
    new_words = []
    negation = False
    for word in words:
        if word.lower() in ["not", "no", "never"]:
            negation = True
            new_words.append(word)
        elif negation:
            new_words.append("NOT_" + word)
            negation = False
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Cleaning text function
def clean_text(text):

    text = emoji.demojize(text)

    text = replace_slang(text)

    text = handle_negations(text)

    text = text.lower()

    text = re.sub(f'[{re.escape(string.punctuation.replace("_", " "))}]', '', text)

    words = word_tokenize(text)

    words = [word for word in words if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)