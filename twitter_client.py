import tweepy
import pandas as pd
from transformers import pipeline
import os
from datetime import datetime, timedelta
import logging

# Configure logging for better error handling and information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TwitterClient:
    """
    A client for retrieving tweets from the Twitter API.
    Handles authentication and different tweet fetching methods.
    """
    def __init__(self):
        # Ensure the BEARER_TOKEN is set in the environment variables
        bearer_token = os.getenv('BEARER_TOKEN')
        if not bearer_token:
            raise ValueError("BEARER_TOKEN environment variable is not set.")
        self.client = tweepy.Client(bearer_token=bearer_token)
        logging.info("TwitterClient initialized successfully.")

    def generate_coin_queries(self, coin_name: str, ticker: str) -> list[str]:
        """
        Generates a list of search terms for a given cryptocurrency.
        """
        variants = {
            ticker,
            coin_name, coin_name.lower(), coin_name.upper(),
            f"#{coin_name}", f"#{ticker}",
            f"#{coin_name}Price"
        }
        for i in range(len(coin_name)):
            # This is a simple form of misspelling, which might be useful
            missp = coin_name[:i] + coin_name[i+1:]
            variants.add(missp.lower())
        leet = coin_name.replace('O', '0').replace('I', '1').replace('E', '3')
        variants.add(leet)
        # Add common crypto-related slang
        variants.update({"HODL", "FOMO", "DYOR", "WAGMI", "moon", "rugpull", "rekt"})
        return list(variants)

    def fetch_tweets_in_range(self, coin_name: str, ticker: str, max_tweets: int, start_date: datetime, end_date: datetime) -> list:
        """
        Fetches tweets within a specific date range.
        This method handles pagination to retrieve more than 100 tweets.
        """
        query = " OR ".join(self.generate_coin_queries(coin_name, ticker))
        query += " -is:retweet lang:en"
        all_tweets = []
        current_end_time = end_date
        while current_end_time > start_date and len(all_tweets) < max_tweets:
            try:
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=min(max_tweets - len(all_tweets), 100),
                    end_time=current_end_time,
                    tweet_fields=["created_at"]
                )
                tweets = response.data or []
                all_tweets.extend(tweets)
                if not tweets:
                    break
                oldest_tweet_time = min(tweet.created_at for tweet in tweets)
                current_end_time = oldest_tweet_time - timedelta(seconds=1)
            except Exception as e:
                logging.error(f"Error fetching tweets: {e}")
                break
        logging.info(f"Fetched {len(all_tweets)} tweets in the specified range.")
        return all_tweets

    def fetch_tweets(self, coin_name: str, ticker: str, max_tweets: int = 100, start_date: str = None, end_date: str = None, use_all_tweets: bool = False) -> list:
        """
        Main method to fetch tweets based on different parameters.
        """
        query = " OR ".join(self.generate_coin_queries(coin_name, ticker))
        query += " -is:retweet lang:en"

        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        if use_all_tweets:
            if not start_dt or not end_dt:
                raise ValueError("search_all_tweets requires both start_date and end_date.")
            try:
                response = self.client.search_all_tweets(
                    query=query,
                    start_time=start_dt,
                    end_time=end_dt,
                    max_results=min(max_tweets, 500),
                    tweet_fields=["created_at"]
                )
                tweets = response.data or []
                logging.info(f"Fetched {len(tweets)} historical tweets.")
                return tweets
            except Exception as e:
                logging.error(f"Error fetching historical tweets: {e}")
                return []
        else:
            if start_dt and end_dt:
                return self.fetch_tweets_in_range(coin_name, ticker, max_tweets, start_dt, end_dt)
            elif end_dt:
                try:
                    response = self.client.search_recent_tweets(
                        query=query,
                        max_results=max_tweets,
                        end_time=end_dt,
                        tweet_fields=["created_at"]
                    )
                    tweets = response.data or []
                    logging.info(f"Fetched {len(tweets)} recent tweets up to end_date.")
                    return tweets
                except Exception as e:
                    logging.error(f"Error fetching recent tweets with end_date: {e}")
                    return []
            else:
                try:
                    response = self.client.search_recent_tweets(query=query, max_results=max_tweets)
                    tweets = response.data or []
                    logging.info(f"Fetched {len(tweets)} most recent tweets.")
                    return tweets
                except Exception as e:
                    logging.error(f"Error fetching most recent tweets: {e}")
                    return []