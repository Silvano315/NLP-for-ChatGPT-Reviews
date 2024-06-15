# DOCKER

FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y sqlite3

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Ensure NLTK data is downloaded
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('vader_lexicon')"

# Add these lines after COPY . .
RUN mkdir /app/staticfiles

# Adjust permissions if necessary
RUN chmod 755 /app/staticfiles

# Run migrations and collect static files
RUN python manage.py migrate
RUN python manage.py collectstatic --noinput

# Run the Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
