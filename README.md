# NLP & LIME for ChatGPT Reviews

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
  * [Kaggle](#kaggle)
  * [Description](#description)
3. [Methods and Results](#methods-and-results)
  * [Data Cleaning and Exploration](#data-cleaning-and-exploration)
  * [Feature Engineering](#feature-engineering)
  * [Machine Learning Predictions](#machine-learning-predictions)
  * [LIME explainability](#lime-explainability)
4. [Django Web App](#django-web-app)
5. [Set up and Requirements](#set-up-and-requirements)
6. [References](#references)

## Introduction
The main idea for this project is to explore a Kaggle dataset about ChatGPT reviews using an NLP approach to apply machine learning models for predicting review scores. The LIME algorithm is used to evaluate explainability, providing text and feature explanations. Additionally, a Docker container is utilized to set up a Django web application for easier interaction with the data and models.

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
- Required Python libraries (listed in `requirements.txt`)

### Steps to Set Up
1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd <project-directory>`
3. Build the Docker Image and starts containers: `docker-compose up -d --build`
4. Create initial migrations for your Django app: `docker-compose exec web python manage.py makemigrations`
5. Apply migrations to create necessary database tables: `docker-compose exec web python manage.py migrate`
6. Access the web application at `http://localhost:8000`

## References
- Kaggle Dataset: [Link to the dataset](https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated/data)
- LIME Algorithm: Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should I trust you?": Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 2016.
- Django Documentation: [https://docs.djangoproject.com/](https://docs.djangoproject.com/)
- Docker Documentation: [https://docs.docker.com/](https://docs.docker.com/)


The dataset in [this folder](Data/) comes from this Kaggle link: https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated/data
