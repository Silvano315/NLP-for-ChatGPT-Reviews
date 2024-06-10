import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Plots a bar chart for a specified numerical feature 
def plot_bar(df, feature, num_bars_max=None):

    value_counts = df[feature].value_counts()
    
    if num_bars_max is not None:
        value_counts = value_counts.head(num_bars_max)
    
    total = value_counts.sum()
    percentages = (value_counts / total) * 100

    cmap = plt.get_cmap('viridis', len(value_counts))
    colors = cmap(np.arange(len(value_counts)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(value_counts.index, value_counts, color=colors)

    legend_labels = [f'{index}: {count} ({percentage:.2f}%)' for index, count, percentage in zip(value_counts.index, value_counts, percentages)]
    plt.legend(bars, legend_labels, title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# Word Cloud to visualize the most frequent and important words 
def plot_word_cloud(text):

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# Bar Plot of Top N Words
def plot_top_n_words(text, n=20):

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(n)
    
    words, counts = zip(*common_words)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette='viridis', hue=list(words), legend=False)
    plt.title(f'Top {n} Most Common Words')
    plt.xlabel('Counts')
    plt.ylabel('Words')
    plt.show()

# Distribution for word length
def plot_word_length_distribution(text):
    
    words = word_tokenize(text)
    word_lengths = [len(word) for word in words if word.isalnum()]

    plt.figure(figsize=(10, 5))
    sns.histplot(word_lengths, bins=range(1, 21), kde=False, color='salmon')
    plt.title('Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.show()


# Sentiment Analysis Plot
def sentiment_analysis(df):

    sia = SentimentIntensityAnalyzer()
    sentiments = df['content'].dropna().apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment'] = sentiments
    return df

def plot_sentiment_distribution(df):

    plt.figure(figsize=(10, 5))
    sns.histplot(df['sentiment'], bins=20, kde=True, color='lightblue')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()


# N-Grams distribution with bar plots
def plot_top_n_ngrams(text, n=2, top_k=20):
    
    words = word_tokenize(text.lower())
    n_grams = ngrams(words, n)
    n_grams_counts = Counter(n_grams)
    common_ngrams = n_grams_counts.most_common(top_k)
    
    ngrams_list, counts = zip(*common_ngrams)
    ngrams_labels = [' '.join(ngram) for ngram in ngrams_list]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=ngrams_labels, palette='magma', hue=ngrams_labels, legend=False)
    plt.title(f'Top {top_k} Most Common {n}-grams')
    plt.xlabel('Counts')
    plt.ylabel(f'{n}-grams')
    plt.show()