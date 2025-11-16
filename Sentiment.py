# Databricks notebook source
df = spark.read.table("workspace.default.stock_tweets")
display(df)

# COMMAND ----------

# MAGIC %pip install vaderSentiment
# MAGIC

# COMMAND ----------

df_spark = spark.read.table("workspace.default.stock_tweets")

# Completely remove the Date column BEFORE conversion
df_spark = df_spark.drop("Date")

pdf = df_spark.toPandas()


# COMMAND ----------

pdf["clean"] = pdf["Tweet"].apply(clean_tweet)
pdf["sentiment_vader"] = pdf["clean"].apply(sentiment_vader)
pdf["fear_score"] = pdf["clean"].apply(fear_score)
pdf["finance_keyword_score"] = pdf["clean"].apply(finance_keyword_score)


# COMMAND ----------

print(pdf)

# COMMAND ----------

# 1. Load Spark table
df_spark = spark.read.table("workspace.default.stock_tweets")

# 2. Remove Date column (optional)
df_spark = df_spark.drop("Date")

# 3. Convert to pandas
pdf = df_spark.toPandas()

# 4. Clean + feature functions
import re, html
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def clean_tweet(text):
    if text is None:
        return ""
    text = html.unescape(text)
    text = (text.replace("â€œ", '"')
                .replace("â€", '"')
                .replace("â€™", "'"))
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\$\w+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def sentiment_vader(text):
    if not text: return 0.0
    return analyzer.polarity_scores(text)["compound"]

fear_words = {"crash","fear","panic","worried","scared","scary",
              "dump","tank","plunge","collapse","bleeding","nervous"}

def fear_score(text):
    if not text: return 0.0
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens: return 0.0
    return sum(1 for t in tokens if t in fear_words) / len(tokens)

finance_keywords = {
    "delivery","deliveries","estimate","estimates",
    "earnings","guidance","margin","margins",
    "revenue","profit","profits","loss","losses",
    "layoff","layoffs","acquisition","equity",
    "rsu","rsus","valuation","price","target",
    "dividend","cashflow","debt"
}

def finance_keyword_score(text):
    if not text: return 0
    tokens = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for t in tokens if t in finance_keywords)

# 5. Apply features
pdf["clean"] = pdf["Tweet"].apply(clean_tweet)
pdf["sentiment_vader"] = pdf["clean"].apply(sentiment_vader)
pdf["fear_score"] = pdf["clean"].apply(fear_score)
pdf["finance_keyword_score"] = pdf["clean"].apply(finance_keyword_score)

# 6. Volume z-score per stock
counts = pdf.groupby("Stock Name")["Tweet"].transform("count")
pdf["volume_zscore"] = (counts - counts.mean()) / counts.std()

# 7. Convert back to Spark DataFrame
df_features = spark.createDataFrame(pdf)

display(df_features)


# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

pdf[["sent_norm", "fear_norm", "keyword_norm", "volume_norm"]] = scaler.fit_transform(
    pdf[[
        "sentiment_vader",
        "fear_score",
        "finance_keyword_score",
        "volume_zscore"
    ]]
)



# COMMAND ----------

pdf["tweet_alpha"] = (
      0.4 * pdf["sent_norm"]
    + 0.3 * pdf["fear_norm"]
    + 0.2 * pdf["keyword_norm"]
    + 0.1 * pdf["volume_norm"]
)


# COMMAND ----------

pdf = pdf.rename(columns={
    "Stock Name": "Stock_Name",
    "Company Name": "Company_Name"
})
df_features = spark.createDataFrame(pdf)


# COMMAND ----------

pdf = pdf.rename(columns={
    "Stock Name": "Stock_Name",
    "Company Name": "Company_Name"
})
df_clean = spark.createDataFrame(pdf)


# COMMAND ----------

df_clean.write \
    .mode("overwrite") \
    .saveAsTable("workspace.default.stock_tweet_features_final")


# COMMAND ----------

