import tweepy
import pandas as pd
from transformers import pipeline
import os
from datetime import datetime, timedelta

class TwitterSentimentClient:
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="StephanAkkerman/FinTwitBERT-sentiment"
        )

    def generate_coin_queries(self, coin_name: str, ticker: str):
        variants = {
            ticker,
            coin_name, coin_name.lower(), coin_name.upper(),
            f"#{coin_name}", f"#{ticker}",
            f"#{coin_name}Price"
        }
        for i in range(len(coin_name)):
            missp = coin_name[:i] + coin_name[i+1:]
            variants.add(missp.lower())
        leet = coin_name.replace('O', '0').replace('I', '1').replace('E', '3')
        variants.add(leet)
        variants.update({"HODL", "FOMO", "DYOR", "WAGMI", "moon", "rugpull", "rekt"})
        return list(variants)

    def fetch_tweets_in_range(self, coin_name, ticker, max_tweets, start_date, end_date):
        query = " OR ".join(self.generate_coin_queries(coin_name, ticker))
        query += " -is:retweet lang:en"
        all_tweets = []
        current_end_time = end_date
        while current_end_time > start_date and len(all_tweets) < max_tweets:
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
        return all_tweets

    def fetch_tweets(self, coin_name, ticker, max_tweets, start_date=None, end_date=None, use_all_tweets=False):
        query = " OR ".join(self.generate_coin_queries(coin_name, ticker))
        query += " -is:retweet lang:en"
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        if use_all_tweets:
            if not start_dt or not end_dt:
                raise ValueError("search_all_tweets requires both start_date and end_date.")
            response = self.client.search_all_tweets(
                query=query,
                start_time=start_dt,
                end_time=end_dt,
                max_results=min(max_tweets, 500),
                tweet_fields=["created_at"]
            )
            return response.data or []
        else:
            if start_dt and end_dt:
                return self.fetch_tweets_in_range(coin_name, ticker, max_tweets, start_dt, end_dt)
            elif end_dt:
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=max_tweets,
                    end_time=end_dt,
                    tweet_fields=["created_at"]
                )
                return response.data or []
            else:
                response = self.client.search_recent_tweets(query=query, max_results=max_tweets)
                return response.data or []

    def fetch_and_analyze(self, coin_name, ticker, max_tweets=100, start_date=None, end_date=None, use_all_tweets=False):
        tweets = self.fetch_tweets(coin_name, ticker, max_tweets, start_date, end_date, use_all_tweets)
        if not tweets:
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
        return pd.DataFrame(data)

# Instantiate and use the client
bearer_token = os.environ.get('BEARER_TOKEN') or "YOUR_BEARER_TOKEN_HERE"
twitter_client = TwitterSentimentClient(bearer_token)

if __name__ == "__main__":
    df = twitter_client.fetch_and_analyze("Ethereum", "ETH", max_tweets=50)
    print(df.head())