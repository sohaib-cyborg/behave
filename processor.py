import praw
from datetime import datetime, timezone, timedelta
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import string
import os
from bybit_client import BybitClient
import nltk
from nltk.corpus import stopwords  # <-- Add this line
import requests
import asyncio
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from twitter_sentiments import fetch_and_analyze  # <-- Add this import

# Reddit API credentials (from step above)
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Remove stock tickers like $BTC or BTC-USD
    text = re.sub(r"\$[A-Za-z0-9]+", "", text)
    text = re.sub(r"\b[A-Za-z]{2,5}-USD\b", "", text)
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Lowercase
    text = text.lower()
    
    # Remove stopwords (but keep sentiment-rich words like 'love', 'fear')
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Remove extra spaces and join back
    return " ".join(tokens).strip()


def fetch_post_for_one_month():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    subreddits = [
        "CryptoCurrency",
        "CryptoMarkets",
        "BitcoinBeginners",
        "CryptoMoonShots",
    ]

    posts = []
    query = 'crpto bitcoin moon hodl pump dump rekt whale ATH FOMO BULL BEAR ATL'
   

    for sub in subreddits:
        for submission in reddit.subreddit(sub).search(query, sort="hot", limit=None, time_filter="month"):
            post_dt = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            text = f"{submission.title} {submission.selftext}".strip()
            cleaned = clean_text(text)
            if cleaned:
                    posts.append({
                        "post_id": submission.id,
                        "date": post_dt.strftime("%Y-%m-%d"),
                        "post": cleaned
                    })

    return pd.DataFrame(posts)

def fetch_reddit_posts(start_date, end_date):
    # Convert to epoch timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    # Subreddits to scrape
    subreddits = [
        "CryptoCurrency",
        "CryptoMarkets",
        "BitcoinBeginners",
        "CryptoMoonShots",

    ]

    posts = []
    for sub in subreddits:
        for submission in reddit.subreddit(sub).search(
            'crpto bitcoin moon hodl pump dump rekt whale ATH FOMO BULL BEAR ATL', sort="hot", limit=None, time_filter="all"
        ):
            post_dt = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            if start_dt <= post_dt <= end_dt:
                text = f"{submission.title} {submission.selftext}".strip()
                cleaned = clean_text(text)
                if cleaned:  # avoid empty posts
                    posts.append({
                        "post_id": submission.id,
                        "date": post_dt.strftime("%Y-%m-%d"),  # formatted date
                        "post": cleaned
                    })

    return pd.DataFrame(posts)



# Mapping dictionary
emotion_to_bias = {
    # FOMO-related biases
    "excitement": "FOMO",
    "optimism": "FOMO",
    "desire": "FOMO",
    "amusement": "FOMO",        # thrill-based attraction can trigger FOMO
    "curiosity": "FOMO",        # curiosity can drive FOMO
    "surprise": "FOMO",         # unexpected events can trigger fear of missing out

    # Herding (following the crowd)
    "approval": "Herding",
    "admiration": "Herding",
    "gratitude": "Herding",     # positive reinforcement from group behavior
    "love": "Herding",          # attachment-driven following of others
    "caring": "Herding",        # empathetic group alignment
    "pride": "Herding",         # aligning with group achievement

    # Panic Selling / Defensive reactions
    "fear": "Panic Selling",
    "nervousness": "Panic Selling",
    "disappointment": "Panic Selling",
    "confusion": "Panic Selling",  # uncertainty can lead to panic exits
    "realization": "Panic Selling",# sudden awareness leading to quick action
    "embarrassment": "Panic Selling", # fear of loss of face may trigger quick exit

    # Loss Aversion
    "sadness": "Loss Aversion",
    "remorse": "Loss Aversion",
    "grief": "Loss Aversion",
    "relief": "Loss Aversion",   # relief from avoiding a perceived loss
    "disapproval": "Loss Aversion", # avoidance of disfavored outcomes

    # Revenge Trading
    "anger": "Revenge Trading",
    "annoyance": "Revenge Trading",
    "disgust": "Revenge Trading",

    # Profit-taking / Overconfidence (positive high bias)
    "joy": "Overconfidence Bias",
    "neutral": "Neutral"  # baseline, not strongly bias-driven
}

# tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
# model = AutoModelForSequenceClassification.from_pretrained(
#     "SamLowe/roberta-base-go_emotions"
# )
# emotion_classifier = pipeline(
#     task="text-classification",
#     model=model,
#     tokenizer=tokenizer,
#     top_k = None,  # Get all labels
#     device=-1
# )

HF_API_TOKEN = ""  

def hf_emotion_inference(text):
    if not text or not text.strip():
        return []
    api_url = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 400:
        print("Bad request for text:", repr(text))
        return []
    response.raise_for_status()
    return response.json()

split_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

def split_text_for_hf_api(text, max_tokens=510):
    tokens = split_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = split_tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def get_daily_bias(df):
    print(df)
    if df.empty:
        return pd.DataFrame()
    daily_posts = df.groupby("date")["post"].apply(lambda posts: " ".join(posts)).reset_index()
    dates = daily_posts["date"].tolist()
    texts = daily_posts["post"].tolist()

    output = []
    for date, text in zip(dates, texts):
        chunks = split_text_for_hf_api(text)
        all_scores = []
        for chunk in chunks:
            result = hf_emotion_inference(chunk)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                all_scores.append({item['label']: item['score'] for item in result[0]})
        if all_scores:
            all_labels = set()
            for scores in all_scores:
                all_labels.update(scores.keys())
            avg_scores = {}
            for label in all_labels:
                avg_scores[label] = sum(scores.get(label, 0.0) for scores in all_scores) / len(all_scores)
            top_emotion = max(avg_scores, key=avg_scores.get)
            print(f"Top emotion for {date}: {top_emotion} with score {avg_scores[top_emotion]}")
            bias = emotion_to_bias.get(top_emotion, "None")
        else:
            bias = "None"
        output.append({"date": date, "bias": bias})

    return pd.DataFrame(output)

def add_irrationality_logic(df: pd.DataFrame):
    irrational_biases = {
        "FOMO": 0.9,
        "Revenge Trading": 0.8,
        "Herding": 0.85,
        "Loss Aversion": 0.7
    }
    
    # Continuous irrationality index based on bias type
    df["irrationality_index"] = df["bias"].map(irrational_biases).fillna(0.0)

    # Emotional Pulse = similarity score × irrationality
    df["emotional_pulse_score"] = df["similarity_score"] * df["irrationality_index"]

    # Contrarian Suggestion with thresholds
    def contrarian(bias, irr, pulse):
        if irr >= 0.75 and pulse >= 0.6 and bias in {"FOMO", "Herding"}:
            return "SELL / Take Profit"
        elif irr >= 0.65 and pulse >= 0.5 and bias in {"Loss Aversion", "Revenge Trading"}:
            return "BUY / Accumulate"
        else:
            return "No action"
    
    df["contrarian_suggestion"] = df.apply(
        lambda row: contrarian(row["bias"], row["irrationality_index"], row["emotional_pulse_score"]),
        axis=1
    )

    # Risk Adjustment Level: scaled
    def risk_level(irr):
        if irr >= 0.8:
            return "High"
        elif irr >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    df["risk_adjustment_level"] = df["irrationality_index"].apply(risk_level)

    return df
def train_models(coin_symbol, start_date, end_date,df_his: pd.DataFrame):
  
    # Fetch Twitter sentiment data for the same date range
    twitter_df = fetch_and_analyze(coin_symbol.split('/')[0], coin_symbol.split('/')[0], max_tweets=100, start_date=start_date, end_date=end_date)
    train_df = merge_price_bias_twitter_posts(
        asyncio.run(main(coin_symbol, start_date, end_date)),
        get_daily_bias(fetch_reddit_posts(start_date, end_date)),
        twitter_df
    )
    train_df = compute_similarity(df_his, train_df)
    train_df = add_irrationality_logic(train_df)

    features = ["price", "volume", "similarity_score", "irrationality_index", "emotional_pulse_score", "twitter_sentiment_score"]  # <-- Add twitter_sentiment_score

    X = train_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: KMeans to generate pseudo-labels
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pseudo_labels = kmeans.fit_predict(X_scaled)

    # Step 2: Train Logistic Regression & RandomForest
    log_reg = LogisticRegression(max_iter=500, random_state=42)
    log_reg.fit(X_scaled, pseudo_labels)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, pseudo_labels)
    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "log_reg": log_reg,
        "rf": rf
    }

def predict_trade_signal(coin_symbol, models, df_his: pd.DataFrame, symbol: str):
    bias_df = get_daily_bias(fetch_post_for_one_month())
    start = bias_df["date"].min()
    end = bias_df["date"].max()
    twitter_df = fetch_and_analyze(coin_symbol.split('/')[0], coin_symbol.split('/')[0], max_tweets=100, start_date=start, end_date=end)
    current_df = merge_price_bias_twitter_posts(
        asyncio.run(main(coin_symbol, start, end)),
        bias_df,
        twitter_df
    )
    current_df = compute_similarity(df_his, current_df)
    current_df = add_irrationality_logic(current_df)

    features = ["price", "volume", "similarity_score", "irrationality_index", "emotional_pulse_score", "twitter_sentiment_score"]  # <-- Add twitter_sentiment_score
    X = current_df[features]
    X_scaled = models["scaler"].transform(X)

    # 1️⃣ Predictions
    log_probs = models["log_reg"].predict_proba(X_scaled)
    rf_probs = models["rf"].predict_proba(X_scaled)
    kmeans_pred = models["kmeans"].predict(X_scaled)

    # 2️⃣ Average ensemble probabilities (only log_reg + rf have probabilities)
    avg_probs = (log_probs + rf_probs) / 2
    best_class = np.argmax(avg_probs, axis=1)[0]
    confidence = float(np.max(avg_probs, axis=1)[0])

    if kmeans_pred[0] == best_class:
        confidence = min(1.0, confidence + 0.1)  # boost
    else:
        confidence = max(0.0, confidence - 0.1)  # reduce
    # Map cluster to signal
    cluster_to_signal = {
        0: "BUY",
        1: "SELL",
        2: "HOLD"
    }
    signal = cluster_to_signal.get(best_class, 2)
    if confidence < 0.6:
        contrarian = current_df["contrarian_suggestion"].iloc[0]
        if "SELL" in contrarian:
            signal = 1
        elif "BUY" in contrarian:
            signal = 0
        else:
            signal = 2

    return {
        "symbol": symbol,
        "signal": signal,
        "source": "behavioral_analysis",
        "confidence": float(confidence),
        "timestamp": datetime.now(),
        "metadata": {
            "predicted_cluster": int(best_class),
            "probabilities": avg_probs[0].tolist()
        }
    }
    
def compute_similarity(historical_df: pd.DataFrame, current_df: pd.DataFrame):
    
    # One-hot encode bias
    encoder = OneHotEncoder(handle_unknown='ignore')
    hist_bias_encoded = encoder.fit_transform(historical_df[['bias']]).toarray()
    curr_bias_encoded = encoder.transform(current_df[['bias']]).toarray()

    # Normalize price and volume
    scaler = MinMaxScaler()
    hist_num_scaled = scaler.fit_transform(historical_df[['price', 'volume']])
    curr_num_scaled = scaler.transform(current_df[['price', 'volume']])

    # Combine encoded bias + scaled price/volume
    hist_features = np.hstack([hist_bias_encoded, hist_num_scaled])
    curr_features = np.hstack([curr_bias_encoded, curr_num_scaled])

    # Compute cosine similarity
    sim_matrix = cosine_similarity(curr_features, hist_features)

    # Get max similarity for each current row (match with closest historical pattern)
    similarity_scores = sim_matrix.max(axis=1)

    current_df = current_df.copy()
    current_df['similarity_score'] = similarity_scores
    return current_df

def merge_price_bias_twitter_posts(bias_df, coin_df, twitter_df):
    # Merge bias and coin data on date
    merged = pd.merge(bias_df, coin_df, on="date", how="inner")
    merged = merged.set_index("date")
    merged = merged.sort_values("date")
    # Prepare twitter sentiment score per date
    if not twitter_df.empty:
        twitter_df['date'] = pd.to_datetime(twitter_df['created_at']).dt.strftime('%Y-%m-%d')
        twitter_daily = twitter_df.groupby('date')['score'].mean().reset_index().rename(columns={'score': 'twitter_sentiment_score'})
        merged = pd.merge(merged.reset_index(), twitter_daily, on="date", how="left").set_index("date")
        merged['twitter_sentiment_score'] = merged['twitter_sentiment_score'].fillna(0.5)  # Neutral if missing
    else:
        merged['twitter_sentiment_score'] = 0.5  # Default neutral
    return merged

def train_models(coin_symbol, start_date, end_date, df_his: pd.DataFrame):
    # Fetch Twitter sentiment data for the same date range
    twitter_df = fetch_and_analyze(coin_symbol.split('/')[0], coin_symbol.split('/')[0], max_tweets=100, start_date=start_date, end_date=end_date)
    train_df = merge_price_bias_twitter_posts(
        asyncio.run(main(coin_symbol, start_date, end_date)),
        get_daily_bias(fetch_reddit_posts(start_date, end_date)),
        twitter_df
    )
    train_df = compute_similarity(df_his, train_df)
    train_df = add_irrationality_logic(train_df)

    features = ["price", "volume", "similarity_score", "irrationality_index", "emotional_pulse_score", "twitter_sentiment_score"]  # <-- Add twitter_sentiment_score

    X = train_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: KMeans to generate pseudo-labels
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pseudo_labels = kmeans.fit_predict(X_scaled)

    # Step 2: Train Logistic Regression & RandomForest
    log_reg = LogisticRegression(max_iter=500, random_state=42)
    log_reg.fit(X_scaled, pseudo_labels)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, pseudo_labels)
    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "log_reg": log_reg,
        "rf": rf
    }

def predict_trade_signal(coin_symbol, models, df_his: pd.DataFrame, symbol: str):
    bias_df = get_daily_bias(fetch_post_for_one_month())
    start = bias_df["date"].min()
    end = bias_df["date"].max()
    twitter_df = fetch_and_analyze(coin_symbol.split('/')[0], coin_symbol.split('/')[0], max_tweets=100, start_date=start, end_date=end)
    current_df = merge_price_bias_twitter_posts(
        asyncio.run(main(coin_symbol, start, end)),
        bias_df,
        twitter_df
    )
    current_df = compute_similarity(df_his, current_df)
    current_df = add_irrationality_logic(current_df)

    features = ["price", "volume", "similarity_score", "irrationality_index", "emotional_pulse_score", "twitter_sentiment_score"]  # <-- Add twitter_sentiment_score
    X = current_df[features]
    X_scaled = models["scaler"].transform(X)

    # 1️⃣ Predictions
    log_probs = models["log_reg"].predict_proba(X_scaled)
    rf_probs = models["rf"].predict_proba(X_scaled)
    kmeans_pred = models["kmeans"].predict(X_scaled)

    # 2️⃣ Average ensemble probabilities (only log_reg + rf have probabilities)
    avg_probs = (log_probs + rf_probs) / 2
    best_class = np.argmax(avg_probs, axis=1)[0]
    confidence = float(np.max(avg_probs, axis=1)[0])

    if kmeans_pred[0] == best_class:
        confidence = min(1.0, confidence + 0.1)  # boost
    else:
        confidence = max(0.0, confidence - 0.1)  # reduce
    # Map cluster to signal
    cluster_to_signal = {
        0: "BUY",
        1: "SELL",
        2: "HOLD"
    }
    signal = cluster_to_signal.get(best_class, 2)
    if confidence < 0.6:
        contrarian = current_df["contrarian_suggestion"].iloc[0]
        if "SELL" in contrarian:
            signal = 1
        elif "BUY" in contrarian:
            signal = 0
        else:
            signal = 2

    return {
        "symbol": symbol,
        "signal": signal,
        "source": "behavioral_analysis",
        "confidence": float(confidence),
        "timestamp": datetime.now(),
        "metadata": {
            "predicted_cluster": int(best_class),
            "probabilities": avg_probs[0].tolist()
        }
    }
# ...existing code...
async def main(symbol,start_date, end_date):
    client = BybitClient(testnet=True)  # create an instance
    df = await client.get_coin_price_volume(
        symbol, start_date, end_date, interval="1d"
    )
    await client.close()  # Close the clie
    return df


if __name__ == "__main__":
    if not os.path.exists("reddit_posts.csv") or os.stat("reddit_posts.csv").st_size == 0:
        CRYPTO_EVENTS = [
            {"name": "2017 Bull Run", "start": "2017-10-01", "end": "2018-01-31"},
            {"name": "COVID Crash", "start": "2020-02-15", "end": "2020-03-31"},
            {"name": "2021 Bull Run", "start": "2020-10-01", "end": "2021-04-30"},
            {"name": "LUNA-UST Collapse", "start": "2022-05-01", "end": "2022-05-31"},
            {"name": "FTX Collapse", "start": "2022-11-01", "end": "2022-11-30"}]
        for event in CRYPTO_EVENTS:
            df_new = get_daily_bias(fetch_reddit_posts(event['start'], event['end']))
            if  df_new.empty:
                continue
            file_path = "reddit_posts.csv"
            if os.path.exists(file_path):
                df_existing = pd.read_csv(file_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                # Drop duplicates if needed
                df_combined.to_csv(file_path, index=False)
            else:
                df_new.to_csv(file_path, index=False)
               
    else:
        bias_df_his = pd.read_csv("reddit_posts.csv")
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        coin = input("Enter coin symbol (e.g., BTC/USDT): ")

        start = bias_df_his["date"].min()
        end = bias_df_his["date"].max()
        coin_symbol = coin  # <-- change as needed
        price_df = asyncio.run(main(coin_symbol, start, end))
        if price_df is None or price_df.empty:
            print(f"No price data found for {coin_symbol} in the specified range.")
            exit(1)  # Reset index before merging
        
    
        merged_df_his = pd.merge(bias_df_his, price_df, on="date", how="inner")
      # Set the index to 'date' after merging    
        models=train_models(coin_symbol, start_date, end_date, merged_df_his)
        

        print(predict_trade_signal(coin_symbol,models, merged_df_his, coin_symbol))