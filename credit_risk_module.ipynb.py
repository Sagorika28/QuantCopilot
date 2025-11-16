# Databricks notebook source
import requests
import pandas as pd
from datetime import datetime

# COMMAND ----------

FRED_API_KEY = "baea789fdd6f6a459dfeebd0e6dfb82d"

# COMMAND ----------

df_macro = spark.read.table("workspace.default.fred_credit_macro")
df_macro.display()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW workspace.default.credit_macro_latest AS
# MAGIC SELECT *
# MAGIC FROM workspace.default.fred_credit_macro
# MAGIC WHERE date = (SELECT max(date) FROM workspace.default.fred_credit_macro);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM workspace.default.credit_macro_latest;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- This is the full history table
# MAGIC SELECT count(*) FROM workspace.default.fred_credit_macro;
# MAGIC
# MAGIC SELECT *
# MAGIC FROM workspace.default.fred_credit_macro
# MAGIC ORDER BY date DESC
# MAGIC LIMIT 5;
# MAGIC

# COMMAND ----------

