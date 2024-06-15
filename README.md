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

### Machine Learning Predictions

### LIME explainability

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