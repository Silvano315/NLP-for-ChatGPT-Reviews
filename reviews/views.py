from django.shortcuts import render
from .forms import ReviewForm
from .models import Review
import lime.lime_text
import pandas as pd
import joblib
import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

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

def replace_slang(text):
    words = text.split()
    new_words = []
    for word in words:
        if word.lower() in slang_dict:
            new_words.append(slang_dict[word.lower()])
        else:
            new_words.append(word)
    return ' '.join(new_words)

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

def clean_text(text):
    text = emoji.demojize(text)
    text = replace_slang(text)
    text = handle_negations(text)
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation.replace("_", ""))}]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load pre-trained model and vectorizer using a pipeline
pipe = joblib.load('Saved_Pipeline/trained_vectorizer_XGBoost.pkl')

def predict_and_explain(review_text):    
    
    explainer_text = lime.lime_text.LimeTextExplainer(class_names=[0, 1, 2, 3, 4])
    exp_text = explainer_text.explain_instance(
        text_instance=review_text,
        classifier_fn=pipe.predict_proba,
        labels=[0, 1, 2, 3, 4]
    )
    predicted_class = pipe.predict([review_text])
    return predicted_class[0], exp_text

def review_view(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.save(commit=False)
            cleaned_text = clean_text(review.review_text)
            predicted_score, explanation = predict_and_explain(cleaned_text)
            review.predicted_score = predicted_score
            review.save()
            return render(request, 'reviews/result.html', {
                'review': review,
                'explanation': explanation.as_html(labels=(predicted_score,))
            })
    else:
        form = ReviewForm()
    return render(request, 'reviews/review_form.html', {'form': form})
