# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT COUNT(*) AS row_count FROM workspace.default.retail_transactions_dataset;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE workspace.default.retail_transactions_dataset;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM workspace.default.retail_transactions_dataset LIMIT 10;

# COMMAND ----------

df = spark.table("workspace.default.retail_transactions_dataset")
row_count = df.count()
col_count = len(df.columns)
print((row_count, col_count))

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df_silver = spark.table("workspace.default.retail_silver")

# exploded view with a clean date column
df = (
    df_silver
    .withColumn("Product", F.explode("Product"))
    .withColumn("DateOnly", F.to_date("Date"))
)

# COMMAND ----------

df_silver.head()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df_silver = spark.table("workspace.default.retail_silver")

# exploded view with a clean date column
df = (
    df_silver
    .withColumn("Product", F.explode("Product"))
    .withColumn("DateOnly", F.to_date("Date"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Demand momentum per product and `day`

# COMMAND ----------

daily_prod = (
    df.groupBy("DateOnly", "Product")
      .agg(F.sum("Total_Items").alias("Daily_Items"))
)

w_prod = Window.partitionBy("Product").orderBy("DateOnly")

daily_prod = (
    daily_prod
    .withColumn("Items_7d", F.sum("Daily_Items").over(w_prod.rowsBetween(-7, 0)))
    .withColumn("Items_30d", F.sum("Daily_Items").over(w_prod.rowsBetween(-30, 0)))
    .withColumn(
        "Demand_Momentum",
        (F.col("Items_7d") - F.col("Items_30d")) / F.col("Items_30d")
    )
)

# COMMAND ----------

daily_prod.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Spend momentum per day

# COMMAND ----------

daily_spend = (
    df.groupBy("City", "DateOnly")
      .agg(F.sum("Total_Cost").alias("Daily_Spend"))
)

w_date = Window.partitionBy("City").orderBy("DateOnly")

daily_spend = (
    daily_spend
    .withColumn("Spend_7d", F.avg("Daily_Spend").over(w_date.rowsBetween(-7, 0)))
    .withColumn("Spend_30d", F.avg("Daily_Spend").over(w_date.rowsBetween(-30, 0)))
    .withColumn(
        "Spend_Momentum",
        (F.col("Spend_7d") - F.col("Spend_30d")) / F.col("Spend_30d")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Discount elasticity per product

# COMMAND ----------

discount = (
    df.groupBy("Product")
      .agg(
          F.avg(F.when(F.col("Discount_Applied") == True, F.col("Total_Items"))).alias("Avg_Items_Discount"),
          F.avg(F.when(F.col("Discount_Applied") == False, F.col("Total_Items"))).alias("Avg_Items_NoDiscount")
      )
      .withColumn(
          "Discount_Elasticity",
          (F.col("Avg_Items_Discount") - F.col("Avg_Items_NoDiscount")) / F.col("Avg_Items_NoDiscount")
      )
)

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Promotion lift per product

# COMMAND ----------

promo = (
    df.groupBy("Product", "Promotion")
      .agg(F.avg("Total_Items").alias("Avg_Items"))
)

baseline = (
    promo.filter(F.col("Promotion") == "None")
         .select("Product", F.col("Avg_Items").alias("Baseline_Avg_Items"))
)

promo_lift = (
    promo.join(baseline, "Product", "left")
         .withColumn(
             "Promo_Lift",
             (F.col("Avg_Items") - F.col("Baseline_Avg_Items")) / F.col("Baseline_Avg_Items")
         )
)

# one promo score per product, take the max lift as the strongest effect
promo_lift_prod = (
    promo_lift.groupBy("Product")
              .agg(F.max("Promo_Lift").alias("Max_Promo_Lift"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Seasonality factor per product and season

# COMMAND ----------

season = (
    df.groupBy("Product", "Season")
      .agg(F.sum("Total_Items").alias("Season_Items"))
)

w_season = Window.partitionBy("Product")

season = season.withColumn(
    "Seasonality_Factor",
    F.col("Season_Items") / F.sum("Season_Items").over(w_season)
)

# COMMAND ----------

# MAGIC %md
# MAGIC 6. Regional activity score per city

# COMMAND ----------

city = (
    df.groupBy("City")
      .agg(
          F.sum("Total_Items").alias("City_Items"),
          F.avg("Total_Cost").alias("Avg_Spend"),
          F.avg(F.col("Discount_Applied").cast("int")).alias("Discount_Rate"),
          F.count("*").alias("Transaction_Count")
      )
)

cols = ["City_Items", "Avg_Spend", "Discount_Rate", "Transaction_Count"]
city_norm = city

for colname in cols:
    stats = city.agg(
        F.min(colname).alias("min"),
        F.max(colname).alias("max")
    ).first()
    city_norm = city_norm.withColumn(
        f"{colname}_norm",
        (F.col(colname) - stats["min"]) / (stats["max"] - stats["min"])
    )

city_norm = city_norm.withColumn(
    "Regional_Activity_Score",
    0.25 * F.col("City_Items_norm") +
    0.25 * F.col("Avg_Spend_norm") +
    0.25 * F.col("Discount_Rate_norm") +
    0.25 * F.col("Transaction_Count_norm")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Build the Gold table with all signals

# COMMAND ----------

base = (
    df.select("DateOnly", "Product", "City", "Season")
      .dropDuplicates()
)

gold = (
    base
    .join(daily_prod, ["DateOnly", "Product"], "left")
    .join(daily_spend.select("DateOnly", "Spend_Momentum"), ["DateOnly"], "left")
    .join(discount.select("Product", "Discount_Elasticity"), ["Product"], "left")
    .join(promo_lift_prod.select("Product", "Max_Promo_Lift"), ["Product"], "left")
    .join(season.select("Product", "Season", "Seasonality_Factor"), ["Product", "Season"], "left")
    .join(city_norm.select("City", "Regional_Activity_Score"), ["City"], "left")
)

# COMMAND ----------

gold.write.mode("overwrite").saveAsTable("workspace.default.retail_altdata_signals")

# COMMAND ----------

spark.table("workspace.default.retail_altdata_signals").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Creating final retail activity score

# COMMAND ----------

signal_cols = [
    "Demand_Momentum",
    "Spend_Momentum",
    "Discount_Elasticity",
    "Max_Promo_Lift",
    "Seasonality_Factor",
    "Regional_Activity_Score",
]

gold_norm = gold

for colname in signal_cols:
    stats = (
        gold.select(
            F.min(colname).alias("min"),
            F.max(colname).alias("max")
        )
        .first()
    )
    cmin, cmax = stats["min"], stats["max"]

    # handle edge case where min == max ie. column is constant
    if cmin is None or cmax is None or cmax == cmin:
        # just set normalized column to 0.5 in that degenerate case
        gold_norm = gold_norm.withColumn(f"{colname}_norm", F.lit(0.5))
    else:
        gold_norm = gold_norm.withColumn(
            f"{colname}_norm",
            (F.col(colname) - F.lit(cmin)) / (F.lit(cmax) - F.lit(cmin))
        )

# COMMAND ----------

gold_scored = (
    gold_norm
    .withColumn(
        "Retail_Activity_Score",
        0.25 * F.col("Demand_Momentum_norm") +
        0.25 * F.col("Spend_Momentum_norm") +
        0.15 * F.col("Discount_Elasticity_norm") +
        0.15 * F.col("Max_Promo_Lift_norm") +
        0.10 * F.col("Seasonality_Factor_norm") +
        0.10 * F.col("Regional_Activity_Score_norm")
    )
)

# COMMAND ----------

gold_scored.select(
    "City", "Product", "DateOnly",
    "Demand_Momentum", "Spend_Momentum",
    "Retail_Activity_Score"
).show(10, truncate=False)

# COMMAND ----------

# gold table
gold_scored.write.mode("overwrite").saveAsTable(
    "workspace.default.retail_altdata_signals_gold"
)

# COMMAND ----------

spark.sql("SELECT COUNT(*) FROM workspace.default.retail_altdata_signals_gold").show()

# COMMAND ----------

spark.sql("""
SELECT 
    City, Product, DateOnly, 
    Retail_Activity_Score 
FROM workspace.default.retail_altdata_signals_gold 
LIMIT 10
""").show()

# COMMAND ----------

display(spark.table("workspace.default.retail_altdata_signals_gold").limit(10))

# COMMAND ----------

gold_df = spark.table("workspace.default.retail_altdata_signals_gold")
row_count = gold_df.count()
col_count = len(gold_df.columns)
print(f"Rows: {row_count}, Columns: {col_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Retail Activity Score (Gold Layer)
# MAGIC
# MAGIC **Table:** `workspace.default.retail_altdata_signals_gold`
# MAGIC
# MAGIC This is the final Gold table for the retail alt-data pipeline.  
# MAGIC Each row represents one combination of `DateOnly`, `Product`, `City`, and `Season` and includes:
# MAGIC
# MAGIC - Raw signals:
# MAGIC   - `Demand_Momentum`
# MAGIC   - `Spend_Momentum`
# MAGIC   - `Discount_Elasticity`
# MAGIC   - `Max_Promo_Lift`
# MAGIC   - `Seasonality_Factor`
# MAGIC   - `Regional_Activity_Score`
# MAGIC
# MAGIC - Normalized signals (`*_norm`)
# MAGIC - **Retail_Activity_Score** (weighted final score from 0 to 1)
# MAGIC
# MAGIC The **Retail Activity Score** is a single metric (ranging from 0 to 1) that captures the overall retail “health” for each **Product × City × Date**.  
# MAGIC It combines six different signals computed from the retail transactions dataset:
# MAGIC
# MAGIC 1. **Demand Momentum**  
# MAGIC    Measures whether demand for a product is rising or falling compared to its 30-day trend.
# MAGIC
# MAGIC 2. **Spend Momentum**  
# MAGIC    Tracks whether total consumer spending in a city is strengthening or weakening over time.
# MAGIC
# MAGIC 3. **Discount Elasticity**  
# MAGIC    Shows how much a product’s demand changes when discounts are applied.
# MAGIC
# MAGIC 4. **Promotion Lift**  
# MAGIC    Captures how effective promotions are at boosting demand for that product.
# MAGIC
# MAGIC 5. **Seasonality Factor**  
# MAGIC    Indicates how strongly the current season contributes to the product’s demand.
# MAGIC
# MAGIC 6. **Regional Activity Score**  
# MAGIC    City-level retail activity based on total items sold, spend levels, discount frequency, and transaction volume.
# MAGIC
# MAGIC All signals are normalized and combined into a weighted **Retail Activity Score**:
# MAGIC
# MAGIC - **High score (closer to 1):**  
# MAGIC   Strong or improving demand, healthy city-level spending, effective promotions, and favorable seasonal conditions.
# MAGIC
# MAGIC - **Low score (closer to 0):**  
# MAGIC   Weakening demand, soft retail spending, low promotional impact, or an unfavorable seasonal period.
# MAGIC
# MAGIC ### Sample Interpretations
# MAGIC
# MAGIC - **New York – Pickles (0.57)**  
# MAGIC   Strong retail conditions with stable demand and high city activity.
# MAGIC
# MAGIC - **Chicago – Dish Soap (0.43)**  
# MAGIC   Stable product demand supported by strong regional activity.
# MAGIC
# MAGIC - **Seattle – Lawn Mower (0.28)**  
# MAGIC   Soft demand momentum and average city spend lead to a lower score.
# MAGIC
# MAGIC - **Los Angeles – Extension Cords (0.25)**  
# MAGIC   Weak demand combined with declining spend momentum.

# COMMAND ----------

