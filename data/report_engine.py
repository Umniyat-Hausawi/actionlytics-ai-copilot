import sqlite3
import pandas as pd
import anthropic
import json
import os
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from analytics import load_data, _ensure_total_price, COMPLETED_STATUSES, CANCELLED_STATUS, get_revenue_by_city
from analytics import get_conversion_rate, get_repeat_purchase_rate, get_most_profitable

# ===== Config =====
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
DB_PATH = "store.db"
client  = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL   = "claude-opus-4-5"


# ===== JSON Serializer =====
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# ===== Filter Data by Date Range =====
def filter_by_period(start_date, end_date, orders_df, campaigns_df):
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    filtered_orders = orders_df[
        (orders_df['order_date'] >= start) &
        (orders_df['order_date'] <= end)
    ].copy()

    filtered_campaigns = campaigns_df[
        (campaigns_df['start_date'] >= start) &
        (campaigns_df['end_date']   <= end)
    ].copy()

    return filtered_orders, filtered_campaigns


# ===== Get Period Label =====
def get_period_label(period, orders_df):
    today = datetime.now()
    periods = {
        "today":      (today, today),
        "week":       (today - timedelta(days=7),   today),
        "month":      (today - timedelta(days=30),  today),
        "quarter":    (today - timedelta(days=90),  today),
        "half_year":  (today - timedelta(days=180), today),
        "year":       (today - timedelta(days=365), today),
        "two_years":  (today - timedelta(days=730), today),
        "all":        (orders_df['order_date'].min(), today),
    }
    return periods.get(period, None)


# ===== Report Limitations =====
def get_report_limitations(metrics, ml_insights):
    """
    Automatically detects what data is missing or limited.
    Returns a list of limitation strings to show in the report.
    """
    limitations = []

    # Campaign data
    camp = metrics.get('campaign_summary', {})
    if camp.get('total_campaigns', 0) == 0:
        limitations.append("No campaign data available for this period")
    elif camp.get('roi_note'):
        limitations.append(f"Campaign ROI: {camp['roi_note']}")

    # Market basket
    basket = ml_insights.get('market_basket', {})
    if isinstance(basket, dict) and 'note' in basket:
        limitations.append("Market basket analysis: insufficient co-purchases in this period")

    # Conversion rate
    if metrics.get('conversion_rate', 0) == 0:
        limitations.append("Conversion rate: calculated from order ratio — no visitor tracking data")

    # Profitability
    if not metrics.get('top_profitable'):
        limitations.append("Profitability analysis: cost price data unavailable")

    # Repeat purchase
    repeat = metrics.get('repeat_purchase_rate', {})
    if not repeat or repeat.get('active_customers', 0) == 0:
        limitations.append("Repeat purchase rate: insufficient order history for this period")

    return limitations


# ===== Calculate Report Metrics =====
def calculate_period_metrics(filtered_orders, filtered_campaigns, products_df, customers_df):
    if len(filtered_orders) == 0:
        return {
            'no_data': True,
            'message': 'No orders found for the selected period. Please choose a different date range.'
        }

    filtered_orders = _ensure_total_price(filtered_orders, products_df)
    completed       = filtered_orders[filtered_orders['status'].isin(COMPLETED_STATUSES)]

    total_revenue   = round(float(completed['total_price'].sum()), 2)
    total_orders    = int(len(completed))
    avg_order_value = round(float(completed['total_price'].mean()), 2) if total_orders > 0 else 0.0
    cancellation_rate = round(
        len(filtered_orders[filtered_orders['status'] == CANCELLED_STATUS]) /
        max(len(filtered_orders), 1) * 100, 2
    )
    unique_customers = int(completed['customer_id'].nunique())

    conversion_rate = get_conversion_rate(filtered_orders)
    if isinstance(conversion_rate, str):
        conversion_rate = 0.0

    repeat_data = get_repeat_purchase_rate(filtered_orders, customers_df)

    name_col     = 'product_name' if 'product_name' in products_df.columns else 'name'
    top_products = completed.groupby('product_id')['quantity'].sum().reset_index()
    merge_cols   = [c for c in ['product_id', name_col, 'category'] if c in products_df.columns]
    top_products = top_products.merge(products_df[merge_cols], on='product_id', how='left')
    if 'product_name' not in top_products.columns and 'name' in top_products.columns:
        top_products = top_products.rename(columns={'name': 'product_name'})
    top_products = top_products.sort_values('quantity', ascending=False).head(3)

    cat_cols = [c for c in ['product_id', 'category'] if c in products_df.columns]
    merged   = completed.merge(products_df[cat_cols], on='product_id', how='left')
    if 'category' not in merged.columns:
        merged['category'] = 'Unknown'
    revenue_by_category = merged.groupby('category')['total_price'].sum().reset_index()
    revenue_by_category = revenue_by_category.sort_values('total_price', ascending=False)

    profitable = get_most_profitable(products_df, filtered_orders)
    if isinstance(profitable, pd.DataFrame) and len(profitable) > 0:
        top_profitable = profitable.head(3).to_dict('records')
    else:
        top_profitable = []

    if len(filtered_campaigns) > 0:
        fc = filtered_campaigns.copy()
        if 'campaign_revenue' in fc.columns and 'budget' in fc.columns and fc['budget'].sum() > 0:
            fc['roi'] = ((fc['campaign_revenue'] - fc['budget']) / fc['budget'] * 100).round(2)
        else:
            fc['roi']      = 0.0
            fc['roi_note'] = 'Campaign revenue data unavailable'

        campaign_name_col = 'campaign_name' if 'campaign_name' in fc.columns else 'name'
        best = fc.loc[fc['roi'].idxmax()]
        campaign_summary = {
            'best_campaign':   str(best.get(campaign_name_col, 'N/A')),
            'best_platform':   str(best.get('platform', 'N/A')),
            'best_roi':        float(best['roi']),
            'total_campaigns': int(len(fc)),
            'roi_note':        str(best.get('roi_note', ''))
        }
    else:
        campaign_summary = {
            'best_campaign':   'No campaigns in this period',
            'best_platform':   'N/A',
            'best_roi':        0.0,
            'total_campaigns': 0,
            'roi_note':        ''
        }

    return {
        'no_data':              False,
        'total_revenue':        total_revenue,
        'total_orders':         total_orders,
        'avg_order_value':      avg_order_value,
        'cancellation_rate':    cancellation_rate,
        'conversion_rate':      conversion_rate,
        'unique_customers':     unique_customers,
        'repeat_purchase_rate': repeat_data if isinstance(repeat_data, dict) else {},
        'top_products':         [
            {'product_name': str(r.get('product_name', '')), 'category': str(r.get('category', '')), 'quantity': int(r['quantity'])}
            for r in top_products.to_dict('records')
        ],
        'top_profitable':       top_profitable,
        'revenue_by_category':  [
            {'category': str(r['category']), 'total_price': float(r['total_price'])}
            for r in revenue_by_category.to_dict('records')
        ],
        'campaign_summary':     campaign_summary,
    }


# ===== Calculate ML Insights =====
def calculate_ml_insights(orders_df, customers_df, products_df):
    insights = {}
    try:
        from ml_engine import (detect_anomalies, segment_customers,
                               predict_churn, market_basket_analysis)

        try:
            anomalies, _ = detect_anomalies(orders_df)
            if len(anomalies) > 0:
                insights['anomalies'] = anomalies[['month', 'type', 'severity']].to_dict('records')
                for a in insights['anomalies']:
                    a['month'] = str(a['month'])
            else:
                insights['anomalies'] = []
        except Exception as e:
            insights['anomalies'] = {'note': f'Anomaly detection unavailable: {e}'}

        try:
            _, seg_summary = segment_customers(orders_df, customers_df)
            insights['segments'] = seg_summary.to_dict('records')
        except Exception as e:
            insights['segments'] = {'note': f'Segmentation unavailable: {e}'}

        try:
            _, churn_summary = predict_churn(orders_df, customers_df)
            insights['churn'] = churn_summary
        except Exception as e:
            insights['churn'] = {'note': f'Churn prediction unavailable: {e}'}

        try:
            basket = market_basket_analysis(orders_df, products_df, min_support=0.01, min_confidence=0.1)
            if len(basket) > 0:
                insights['market_basket'] = basket.head(3).to_dict('records')
            else:
                insights['market_basket'] = {'note': 'No basket patterns found'}
        except Exception as e:
            insights['market_basket'] = {'note': f'Market basket unavailable: {e}'}

    except ImportError as e:
        insights['note'] = f'ML insights unavailable: {e}'

    return insights


# ===== Generate Report with Claude API =====
def generate_report(metrics, ml_insights, period_label, language="ar",
                    data_source="Demo Store", limitations=None):

    lang_instruction = "Respond in Arabic only." if language == "ar" else "Respond in English only."
    generated_on     = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Data source header — fixed Python string, not dependent on LLM
    if language == "ar":
        data_header = f"📁 مصدر البيانات: {data_source} | تاريخ التوليد: {generated_on}"
        intro_msg   = f"التقرير الشامل جاهز — سأرسله على جزأين 📊\n{data_header}"
    else:
        data_header = f"📁 Data Source: {data_source} | Generated: {generated_on}"
        intro_msg   = f"Full report ready — sending in two parts 📊\n{data_header}"

    # Limitations section
    limitations_text = ""
    if limitations:
        if language == "ar":
            limitations_text = "\n\n⚠️ **ملاحظات على البيانات:**\n" + "\n".join(f"- {l}" for l in limitations)
        else:
            limitations_text = "\n\n⚠️ **Report Limitations:**\n" + "\n".join(f"- {l}" for l in limitations)

    prompt_part1 = f"""
You are a professional e-commerce business analyst.
Generate PART 1 of a comprehensive store performance report.

Data Source: {data_source}
Period: {period_label}
Generated: {generated_on}

Store Metrics:
{json.dumps(metrics, ensure_ascii=False, indent=2, default=convert_to_serializable)}

IMPORTANT ALERTS — Flag these clearly in the report:
- If cancellation_rate > 10%: add ⚠️ RED ALERT — cancellation rate is above industry benchmark (5-10%), needs immediate action
- If repeat_purchase_rate > 80%: add ⚠️ NOTE — this rate seems unusually high, flag it for review
- If conversion_rate < 1%: add ⚠️ ALERT — conversion rate is below industry benchmark

Write PART 1 including:
1. Executive Summary — overall performance in 2-3 sentences, highlight any alerts
2. Revenue Analysis — total revenue, avg order value, conversion rate
3. Top Products — best sellers and most profitable
4. Customer Insights — unique customers, repeat purchase rate, segmentation (VIP/Regular/Dormant)
5. Campaign Performance — best campaign and ROI

Always start with: "Data Source: {data_source}"
If any metric shows "unavailable" or "No data", mention it briefly and skip.
Do NOT generate or mention any discount codes.
{lang_instruction}
Keep it professional, data-driven, and actionable.
"""

    prompt_part2 = f"""
You are a professional e-commerce business analyst.
Generate PART 2 of a comprehensive store performance report.

Data Source: {data_source}
Period: {period_label}
Generated: {generated_on}

Historical ML Insights (analysis of past data only — no forecasting):
{json.dumps(ml_insights, ensure_ascii=False, indent=2, default=convert_to_serializable)}

Write PART 2 including:
1. Anomalies — any unusual months detected in the historical data
2. Churn Risk — customers at risk based on historical purchase patterns
3. Market Basket — product combinations customers bought together historically
4. Key Recommendations — 3 actionable insights based on historical analysis

Always start with: "Data Source: {data_source}"
This is a HISTORICAL report only. Do NOT include any future forecasts or predictions.
If any insight shows "unavailable" or "No data", mention it briefly and skip.
Do NOT generate or mention any discount codes.
{lang_instruction}
Keep it professional, data-driven, and actionable.
"""

    response1 = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt_part1}]
    )

    response2 = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt_part2}]
    )

    return {
        "intro":        intro_msg,
        "part1":        response1.content[0].text + limitations_text,
        "part2":        response2.content[0].text,
        "generated_on": generated_on,
        "data_header":  data_header,
    }


# ===== Main Report Runner =====
def run_report(period="month", start_date=None, end_date=None, language="ar",
               products_df=None, customers_df=None, orders_df=None, campaigns_df=None,
               data_source="Demo Store"):

    if orders_df is None or products_df is None:
        products_df, customers_df, orders_df, campaigns_df = load_data()

    orders_df    = orders_df.copy()
    campaigns_df = campaigns_df.copy()
    orders_df['order_date']    = pd.to_datetime(orders_df['order_date'])
    campaigns_df['start_date'] = pd.to_datetime(campaigns_df['start_date'])
    campaigns_df['end_date']   = pd.to_datetime(campaigns_df['end_date'])

    if start_date and end_date:
        period_label = f"{start_date} → {end_date}"
    else:
        dates = get_period_label(period, orders_df)
        if dates is None:
            return {"error": "Invalid period selected"}
        start_date, end_date = dates
        period_names = {
            "today":     "اليوم"        if language == "ar" else "Today",
            "week":      "آخر أسبوع"    if language == "ar" else "Last Week",
            "month":     "آخر شهر"      if language == "ar" else "Last Month",
            "quarter":   "آخر 3 أشهر"  if language == "ar" else "Last Quarter",
            "half_year": "آخر 6 أشهر"  if language == "ar" else "Last 6 Months",
            "year":      "آخر سنة"      if language == "ar" else "Last Year",
            "two_years": "آخر سنتين"   if language == "ar" else "Last 2 Years",
            "all":       "منذ البداية"  if language == "ar" else "All Time",
        }
        period_label = period_names.get(period, period)

    filtered_orders, filtered_campaigns = filter_by_period(
        start_date, end_date, orders_df, campaigns_df
    )

    metrics = calculate_period_metrics(
        filtered_orders, filtered_campaigns, products_df, customers_df
    )

    if metrics.get('no_data'):
        return {
            "error":        metrics['message'],
            "period_label": period_label,
            "data_source":  data_source
        }

    ml_insights  = calculate_ml_insights(orders_df, customers_df, products_df)
    limitations  = get_report_limitations(metrics, ml_insights)
    report       = generate_report(metrics, ml_insights, period_label, language,
                                   data_source, limitations)

    return {
        "period_label":    period_label,
        "start_date":      str(start_date),
        "end_date":        str(end_date),
        "metrics":         metrics,
        "ml_insights":     ml_insights,
        "limitations":     limitations,
        "generated_on":    report["generated_on"],
        "data_header":     report["data_header"],
        "report_intro":    report["intro"],
        "report_part1":    report["part1"],
        "report_part2":    report["part2"],
        "filtered_orders": filtered_orders,
        "data_source":     data_source
    }


# ===== Main =====
if __name__ == "__main__":
    try:
        result = run_report(start_date="2024-01-01", end_date="2024-12-31",
                            language="ar", data_source="Demo Store")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Period       : {result['period_label']}")
            print(f"Data Source  : {result['data_source']}")
            print(f"Generated On : {result['generated_on']}")
            print(f"Data Header  : {result['data_header']}")
            print(f"Revenue      : {result['metrics']['total_revenue']} SAR")
            if result['limitations']:
                print(f"\nLimitations:")
                for l in result['limitations']:
                    print(f"  - {l}")
            print(f"\n{result['report_intro']}")
            print(f"\n===== Part 1 =====\n{result['report_part1'][:300]}...")
            print(f"\n===== Part 2 =====\n{result['report_part2'][:300]}...")
    except Exception as e:
        print(f"Report generation failed: {e}")