# Databricks notebook source
# ============================================
# QuantCopilot – LLM Skeleton
# Using Databricks Model: meta-llama-3-1-405b-instruct
# ============================================

from typing import Dict, Any, List
import ast

# -------------------------------------------------
# 1) LLM backend: call meta-llama-3-1-405b via ai_query
# -------------------------------------------------
def call_llm_api(prompt: str) -> str:
    """
    Real LLM call using Databricks AI functions (ai_query).
    Make sure you're running this on a cluster/SQL warehouse
    that supports ai_query and has access to:
        'databricks-meta-llama-3-1-405b-instruct'
    """

    safe_prompt = prompt.replace("'", "''")  # Escape quotes

    df = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-1-405b-instruct',
            '{safe_prompt}'
        ) AS response
    """)

    row = df.collect()[0]
    return row["response"]


# -------------------------------------------------
# 2) QuantCopilotAgent – main wrapper class
# -------------------------------------------------
class QuantCopilotAgent:
    def __init__(self, model_name: str = "databricks-meta-llama-3-1-405b-instruct"):
        self.model_name = model_name

    # -------------------------------------------------
    # a) Fetch context from tables
    # -------------------------------------------------
    def fetch_company_context(self, ticker: str) -> Dict[str, Any]:

        # ---------- Company data ----------
        df = spark.sql(f"""
            SELECT *
            FROM workspace.default.yfin_with_anomalies
            WHERE ticker = '{ticker.upper()}'
            LIMIT 1
        """).toPandas()

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        row = df.iloc[0]

        # ---------- Macro data ----------
        macro_df = spark.sql("""
            SELECT *
            FROM workspace.default.credit_macro_latest
            LIMIT 1
        """).toPandas()

        macro: Dict[str, Any] = {}
        if not macro_df.empty:
            m = macro_df.iloc[0]
            macro = {
                "date": str(m.get("date")),
                "spread_bps": float(m.get("spread_bps")) if "spread_bps" in m else None,
                "spread_3m_change": float(m.get("spread_3m_change")) if "spread_3m_change" in m else None,
                "credit_risk_bucket": m.get("credit_risk_bucket"),
            }

        # ---------- Parse red flags ----------
        red_flags_list: List[str] = []

        # Preferred: red_flags column
        if "red_flags" in row and row["red_flags"] is not None:
            val = row["red_flags"]
            if isinstance(val, list):
                red_flags_list = [f for f in val if f]
            else:
                try:
                    parsed = ast.literal_eval(str(val))
                    if isinstance(parsed, list):
                        red_flags_list = [f for f in parsed if f]
                except:
                    pass

        # Fallback: red_flags_array column
        if not red_flags_list and "red_flags_array" in row and row["red_flags_array"] is not None:
            arr_val = row["red_flags_array"]
            if isinstance(arr_val, list):
                red_flags_list = [f for f in arr_val if f]
            else:
                try:
                    parsed = ast.literal_eval(str(arr_val))
                    if isinstance(parsed, list):
                        red_flags_list = [f for f in parsed if f]
                except:
                    pass

        red_flags = {
            "red_flag_score": float(row.get("red_flag_score", 0.0)),
            "red_flag_severity": row.get("red_flag_severity"),
            "red_flag_count": int(row.get("red_flag_count", 0)),
            "red_flags": red_flags_list,
        }

        # ---------- Build final context dict ----------
        context: Dict[str, Any] = {
            "ticker": row["ticker"],
            "company_name": row.get("company_name"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "country": row.get("country"),
            "exchange": row.get("exchange"),
            "currency": row.get("currency"),
            "market_cap": float(row.get("market_cap")) if row.get("market_cap") else None,

            # Scores
            "value_score": row.get("value_score"),
            "growth_score": row.get("growth_score"),
            "quality_score": row.get("quality_score"),
            "risk_score": row.get("risk_score"),
            "overall_alpha_score": row.get("overall_alpha_score"),
            "conviction_rating": row.get("conviction"),

            # Volatility & risk
            "beta": row.get("beta"),
            "volatility_30d": row.get("volatility_30d"),
            "volatility_90d": row.get("volatility_90d"),
            "volatility_52w": row.get("volatility_52w"),

            # Cash flow & leverage
            "free_cash_flow": row.get("free_cashflow"),
            "free_cash_flow_margin": row.get("free_cash_flow_margin"),
            "fcf_yield": row.get("fcf_yield"),
            "debt_to_equity": row.get("debt_to_equity"),
            "current_ratio": row.get("current_ratio"),
            "cash_to_debt": row.get("cash_to_debt"),

            # Macro + red flags
            "macro": macro,
            "red_flags": red_flags,

            "peers": [],  # Placeholder
        }

        return context

    # -------------------------------------------------
    # b) Build the LLM prompt
    # -------------------------------------------------
    def build_prompt(self, context: Dict[str, Any], user_question: str) -> str:
        macro = context.get("macro", {})
        red = context.get("red_flags", {})

        red_flags_list = red.get("red_flags", [])
        red_flags_str = ", ".join(red_flags_list) if red_flags_list else "None detected"

        prompt = f"""
You are **QuantCopilot**, an AI equity analyst. Provide a realistic, numerically grounded, institution-grade analysis.

CRITICAL RULES:
- Use ONLY the data provided below.
- Always tie reasoning to specific metrics.
- Do NOT hallucinate peers or future events.
- Final verdict MUST match the conviction rating and overall alpha score.

========================
USER QUESTION
========================
{user_question}

========================
COMPANY SNAPSHOT
========================
Ticker: {context.get('ticker')}
Name: {context.get('company_name')}
Sector: {context.get('sector')}
Industry: {context.get('industry')}
Country: {context.get('country')}
Exchange / Currency: {context.get('exchange')} / {context.get('currency')}
Market cap: {context.get('market_cap')}

========================
FACTOR SCORES
========================
Value score:      {context.get('value_score')}
Growth score:     {context.get('growth_score')}
Quality score:    {context.get('quality_score')}
Risk score:       {context.get('risk_score')}
Overall alpha:    {context.get('overall_alpha_score')}
Conviction:       {context.get('conviction_rating')}

========================
RISK, LEVERAGE & CASH FLOW
========================
Beta:                 {context.get('beta')}
30D volatility:       {context.get('volatility_30d')}
90D volatility:       {context.get('volatility_90d')}
52W volatility:       {context.get('volatility_52w')}
Debt to equity:       {context.get('debt_to_equity')}
Current ratio:        {context.get('current_ratio')}
Cash to debt:         {context.get('cash_to_debt')}
FCF yield:            {context.get('fcf_yield')}
Free cash flow margin:{context.get('free_cash_flow_margin')}

========================
MACRO CREDIT ENVIRONMENT
========================
BAA-AAA spread (bps): {macro.get('spread_bps')}
3M change (bps):      {macro.get('spread_3m_change')}
Credit risk bucket:   {macro.get('credit_risk_bucket')}

Macro interpretation rules:
- LOW bucket = favorable credit backdrop.
- Negative 3M change = spreads narrowed → improving credit conditions.
- Positive 3M change = spreads widened → rising credit stress.

========================
RED FLAGS
========================
Red flag score:    {red.get('red_flag_score')}
Red flag severity: {red.get('red_flag_severity')}
Red flag count:    {red.get('red_flag_count')}
Flags:             {red_flags_str}

========================
RESPONSE FORMAT
========================
Respond in 5 sections:

1. **High-Level View**

2. **Key Positives**
- 3–5 bullet points, each grounded in real metrics.

3. **Key Risks**
- MUST discuss leverage if D/E is high or "extreme_debt" is flagged.
- Include volatility and quality issues.

4. **Macro & Scenario View**
- Explain how the credit regime affects a leveraged company.
- Add a short scenario: "what if spreads widen again?"

5. **Final Verdict**
- MUST match conviction rating: {context.get('conviction_rating')}
- Explain what type of investor (risk-tolerant vs conservative) might consider this.

Do NOT restate raw numbers—always interpret their meaning.
"""
        return prompt

    # -------------------------------------------------
    # c) LLM call wrapper
    # -------------------------------------------------
    def call_llm(self, prompt: str) -> str:
        return call_llm_api(prompt)

    # -------------------------------------------------
    # d) Public entrypoint
    # -------------------------------------------------
    def answer(self, ticker: str, user_question: str) -> str:
        context = self.fetch_company_context(ticker)
        prompt = self.build_prompt(context, user_question)
        return self.call_llm(prompt)


# COMMAND ----------

agent = QuantCopilotAgent()

response = agent.answer(
    "NEM",
    "Is this a good long-term investment, and what are the main risks I should worry about?"
)

print(response)


# COMMAND ----------

agent = QuantCopilotAgent()

for t in ["NEM", "CTVA", "AAPL", "MSFT"]:
    resp = agent.answer(t, "Is this a good long-term investment, and what are the key risks?")
    print(f"\n\n===== {t} =====\n")
    print(resp)

# COMMAND ----------

# ==============================
# UI: Single Stock Analysis
# ==============================

# Create widgets
dbutils.widgets.text("ticker", "NEM", "Ticker")
dbutils.widgets.text("question", "Is this a good long-term investment, and what are the main risks?", "Question")

# Read widget values
ticker = dbutils.widgets.get("ticker")
question = dbutils.widgets.get("question")

# Instantiate the agent (you can also reuse a global one if already created)
agent = QuantCopilotAgent()

# Call the agent
response = agent.answer(ticker, question)

print(f"=== QuantCopilot Analysis for {ticker.upper()} ===\n")
print(response)

# COMMAND ----------

