import tweepy
import pandas as pd
from transformers import pipeline
import os
import logging
from twitter_client import TwitterClient

class SentimentClient:
    """
    A client for performing sentiment analysis on text data.
    """
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="StephanAkkerman/FinTwitBERT-sentiment"
        )
        logging.info("SentimentClient initialized with FinTwitBERT-sentiment model.")

    def analyze_tweets(self, tweets: list) -> pd.DataFrame:
        """
        Analyzes the sentiment of a list of tweets and returns a pandas DataFrame.
        """
        if not tweets:
            logging.warning("No tweets provided for analysis.")
            return pd.DataFrame(columns=["text", "label", "score"])

        texts = [tweet.text for tweet in tweets]
        results = self.sentiment_analyzer(texts)
        
        data = []
        for tweet, res in zip(tweets, results):
            data.append({
                "text": tweet.text,
                "label": res["label"],
                "score": res["score"],
                "created_at": tweet.created_at
            })
        logging.info(f"Analyzed sentiment for {len(data)} tweets.")
        return pd.DataFrame(data)

# Main block to demonstrate usage
if __name__ == "__main__":
    try:
        # Instantiate the two new clients
        twitter_client = TwitterClient()
        sentiment_client = SentimentClient()

        # Step 1: Fetch tweets using the TwitterClient
        logging.info("Starting tweet fetching...")
        tweets_to_analyze = twitter_client.fetch_tweets("Ethereum", "ETH", max_tweets=50)

        # Step 2: Analyze the sentiment of the fetched tweets using the SentimentClient
        logging.info("Starting sentiment analysis...")
        sentiment_df = sentiment_client.analyze_tweets(tweets_to_analyze)

        # Print the first few rows of the resulting DataFrame
        print("\nSentiment Analysis Results:")
        print(sentiment_df.head())
    except ValueError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
