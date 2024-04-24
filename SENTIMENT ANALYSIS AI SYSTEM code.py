import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load the dataset
data = pd.read_csv('imdb_movie_reviews.csv')  # Replace 'imdb_movie_reviews.csv' with your dataset file path

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment score for each review
def get_sentiment_score(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

# Add a new column for sentiment score
data['sentiment_score'] = data['review'].apply(get_sentiment_score)

# Function to categorize sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
# Add a new column for sentiment category
data['sentiment_category'] = data['sentiment_score'].apply(categorize_sentiment)

# Print the first few rows of the dataset with sentiment scores and categories
print(data[['review', 'sentiment_score', 'sentiment_category']].head())
