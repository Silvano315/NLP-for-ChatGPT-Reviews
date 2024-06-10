import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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