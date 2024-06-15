# NLP & LIME for ChatGPT Reviews

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
  - [Kaggle](#kaggle)
  - [Description](#description)
3. [Methods and Results](#methods-and-results)
  - [Data Cleaning and Exploration](#data-cleaning-and-exploration)
  - [Feature Engineering](#feature-engineering)
  - [Machine Learning Predictions](#machine-learning-predictions)
  - [LIME explainability](#lime-explainability)
4. [Django Web App](#django-web-app)
5. [Set up and Requirements](#set-up-and-requirements)
6. [References](#references)

## Introduction
The main idea for this project is to explore a Kaggle dataset about ChatGPT reviews using an NLP approach to apply machine learning models for predicting review scores. The LIME algorithm is used to evaluate explainability, providing text and feature explanations. Additionally, a Docker container is utilized to set up a Django web application for easier interaction with the data and models.

For a project like ChatGPT, there is an enormous amount of reviews. Understanding which ones are valid, explaining why they are valid, or allowing a user to generate a rating from their comment with an added layer of explainability is certainly a step forward in the world of marketing and communication. The reasons for these advantages are threefold:
1. **Enhanced Credibility**
2. **User Engagement**
3. **Informed Decision-Making**

## Dataset

### Kaggle
The dataset can be found on Kaggle and is constantly updated by the owner, ensuring that the data remains relevant and up-to-date. [Link to the dataset](https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated/data)

### Description
The dataset is composed of more than 130,000 rows with the following columns:
- `reviewId`: Unique identifier for each review
- `userName`: Username of the reviewer
- `content`: Text content of the review
- `score`: Rating score given by the reviewer [1,2,3,4,5]
- `thumbsUpCount`: Number of thumbs up received by the review
- `reviewCreatedVersion`: Version of the app when the review was created
- `at`: Timestamp of when the review was posted
- `appVersion`: Version of the app being reviewed

## Methods and Results

### Data Cleaning and Exploration
After observing some statistical information about the dataset, I proceeded to clean it by removing possible duplicates and eliminating the few rows with NaN values in the `content` column. For visualization, I initially used a bar plot which revealed that the dataset is highly imbalanced, with 76.07% of the reviews having a score of 5. This imbalance poses a challenge for classification tasks.

To gain further insights, I used Word Cloud python library to visualize the most common words in the text data, with larger words representing higher frequencies. You can view the word cloud [here](Images_ReadMe/Word_Cloud.png).

I also examined the distribution of review lengths, finding that most reviews are relatively short, not exceeding five words. Additionally, I plotted a histogram to perform sentiment analysis using the SentimentIntensityAnalyzer() from the `nltk.sentiment.vader` (nltk Natural Language Toolkit) library in Python, which is a popular tool for this task. VADER is designed to analyze sentiment in short texts, such as tweets or reviews, and performs well with informal language, emoticons, and other elements commonly found in social media. VADER also adjusts scores based on context, such as intensity and negation. This analysis highlighted that the majority of reviews have a positive sentiment score, aligning with the predominance of reviews with a score of 5.

### Feature Engineering
After initial data cleaning where unnecessary features like ID and name were removed, several functions were applied to preprocess and engineer features from the text data:
1. **Text Cleaning and Preprocessing:**
   - Text cleaning involved converting emojis to text equivalents, replacing slang using a predefined dictionary, handling negations by prefixing "NOT_" to words following negation terms, converting text to lowercase, removing punctuation, tokenizing words, removing stopwords, and lemmatizing words.
2. **TF-IDF Vectorization:**
   - The `create_tfidf_features` function was utilized to create TF-IDF (Term Frequency-Inverse Document Frequency) features from the cleaned text data. TF-IDF vectors represent the importance of words in a document relative to a collection of documents. TF-IDF provides a better representation of the relative importance of words in documents, it is not an easy method like Bag of Words.
3. **Sentiment Analysis:**
   - Sentiment analysis was performed using the VADER tool from NLTK. The `add_sentiment_features` function calculated sentiment scores (compound score) for each review, indicating the overall sentiment expressed in the text.
4. **Feature Scaling:**
   - Features such as `thumbsUpCount`and `text_length` were scaled using MinMaxScaler to ensure uniformity in their ranges and improve model performance.

After applying these feature engineering techniques, the dataset was enriched with the following columns:
- `score`
- `thumbsUpCount`
- `sentiment`
- `text_length`
- 100 columns representing TF-IDF features capturing important terms from the text data.

These engineered features are intended to provide meaningful insights and inputs for machine learning models aimed at predicting review scores based on textual content and associated metadata.

### Machine Learning Predictions
I evaluated three machine learning models (Random Forest, Logistic Regression, XGBoost) using a Stratified 10-fold Cross-Validation. During each cycle of cross-validation, I saved metrics such as accuracy, precision, recall, f1 score, balanced_accuracy, confusion_matrix on the test dataset. These metrics were then visualized and evaluated using the average results, which can be seen in this [image link](Images_ReadMe/Evaluation_metrics.png). It's important to note that this was a classification task with 5 labels and an imbalanced dataset. The results highlighted these factors, showing relatively good metrics overall but notably poor balanced accuracy. This indicates that the models tended to favor the majority class (label 5), as evidenced by the confusion matrix in [notebook link](ML_prediction.ipynb).

To address this imbalance, I experimented with and tested various undersampling techniques such as Cluster Centroids, Tomek Links, and random undersampling. Concerns about losing a significant amount of data led me to consider these techniques as exploratory examples, while avoiding oversampling due to the dataset's already substantial size and to prevent additional computational overhead.

Visualizing the new datasets using PCA and t-SNE analyses, where the feature set was reduced to two principal components and Silhouette scores were calculated, revealed that these datasets did not form well-defined clusters. Evaluating balanced accuracy through Stratified 10-fold Cross-Validation showed that while the results were not perfect, techniques like Cluster Centroids demonstrated promising performance with a balanced accuracy reaching around 54% ± 5%. You can find everything in the same [notebook link](ML_prediction.ipynb).

Finally, I saved the TF-IDF pipeline with XGBoost trained solely on the entire cleaned review dataset. This pipeline will be used for the Django web application project.

### LIME explainability
I used the **LIME** algorithm to obtain local explanations by applying it in two ways. 

First, I used it to evaluate how XGBoost (which proved to be the most stable and fastest algorithm) performed with my dataset of extracted features, both textual and non-textual. I split the dataset and trained it on the training portion, then created the `LimeTabularExplainer` and the explainer instance to evaluate how the model interpreted the features. LIME provides explanations for individual predictions based on the features and their contributions to the model's decision. The results highlighted the enormous potential of the project, but also the limitations of training on an imbalanced dataset. Useful insights were obtained regarding how the model considered certain features for the chosen prediction, but the result was not fully satisfactory.

For this reason, I decided to use LIME as a text explainer (`LimeTextExplainer`) and trained and tested XGBoost solely on the cleaned and TF-IDF vectorized textual reviews. The results significantly improved, with the explanations proving to be very useful in showing which words were most important in correctly predicting the chosen review. Some examples can be found in this [notebook](LIME.ipynb). 

These results gave me the idea to create a web app with **Django** and containerize it with **Docker** to utilize the pre-trained TF-IDF + XGBoost pipeline on the entire dataset. This application allows a new user to provide their username and a review, receive a predicted score based on their review, and see the explanation for why this decision was made, and finally ask the user if they are satisfied.

## Django Web App
A Django web application is implemented to provide an interactive interface for exploring the dataset and making predictions using the trained machine learning models. This web app is containerized using Docker, ensuring easy deployment and scalability.

## Set up and Requirements
To set up the project locally, ensure you have the following prerequisites:

- Docker
- Python 3.11.x
- Django
- Required Python libraries (listed in [requirements.txt](requirements.txt))

### Steps to Set Up
1. Clone the repository: 
`git clone <repository-url>`
2. Navigate to the project directory: 
`cd <project-directory>`
3. Build the Docker Image and starts containers: 
`docker-compose up -d --build`
4. Create initial migrations for your Django app: 
`docker-compose exec web python manage.py makemigrations`
5. Apply migrations to create necessary database tables: 
`docker-compose exec web python manage.py migrate`
6. Access the web application at:
`http://localhost:8000`

## References
1. OpenAI: [https://www.openai.com/](https://www.openai.com/)
2. Kaggle Dataset: [Link to the dataset](https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated/data)
3. LIME Algorithm: Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should I trust you?": Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 2016.
4. Django Documentation: [https://docs.djangoproject.com/](https://docs.djangoproject.com/)
5. Docker Documentation: [https://docs.docker.com/](https://docs.docker.com/)