import os
import sys
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.dirname(__file__))
from analytics import *
from analytics import _ensure_total_price, COMPLETED_STATUSES
from ml_engine import (forecast_sales, detect_anomalies, forecast_product_demand,
                       segment_customers, predict_churn, market_basket_analysis)
from rag_engine import build_rag, query_rag
from action_engine import run_churn_reminders, run_restock_alerts, run_best_send_time
from report_engine import run_report

# Build RAG once at startup
try:
    rag_vectorstore = build_rag()
except Exception as e:
    rag_vectorstore = None
    print(f'Warning: RAG could not be initialized: {e}')

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# ===== Conversation History Limit =====
MAX_HISTORY = 10  # آخر 10 رسائل = 5 exchanges

# ===== Keywords =====
RAG_KEYWORDS = [
    'benchmark', 'معيار', 'مقارنة', 'compare', 'industry', 'قطاع',
    'average', 'متوسط', 'best practice', 'أفضل ممارسة',
    'cancellation', 'إلغاء', 'conversion', 'تحويل', 'roi', 'عائد',
    'churn', 'تذبذب', 'retention', 'احتفاظ', 'repeat', 'تكرار'
]

ACTION_KEYWORDS = [
    'رسائل', 'رسالة', 'استرجاع', 'win-back', 'winback',
    'مخزون', 'restock', 'تخزين', 'نفاد',
    'سلة', 'cart', 'abandoned',
    'vip عرض', 'عرض vip', 'vip offer',
    'حملة توصية', 'campaign tip',
    'أرسل', 'send message', 'generate message'
]

# Keywords that trigger VIP names inclusion
VIP_NAME_KEYWORDS = [
    'اسم', 'أسماء', 'من هم', 'قائمة', 'list', 'names',
    'who are', 'vip customers', 'عملاء vip', 'عملاء الـ vip',
    'اعرض', 'show me', 'أظهر'
]


def _needs_rag(question):
    q = question.lower()
    return any(kw in q for kw in RAG_KEYWORDS)

def _needs_action_engine(question):
    q = question.lower()
    return any(kw in q for kw in ACTION_KEYWORDS)

def _needs_vip_names(question):
    """Returns True if the question is asking for VIP customer names."""
    q = question.lower()
    has_vip  = 'vip' in q
    has_name = any(kw in q for kw in VIP_NAME_KEYWORDS)
    return has_vip and has_name

def _friendly_error(language="en"):
    if language == "ar":
        return "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى."
    return "Sorry, something went wrong while processing your request. Please try again."

def _detect_language(text):
    return "ar" if any('\u0600' <= c <= '\u06ff' for c in text) else "en"

def _action_engine_redirect(language="ar"):
    if language == "ar":
        return (
            "لإنشاء رسائل أو تنبيهات مخصصة مبنية على بيانات متجرك الفعلية، "
            "استخدم قسم ⚡ **توصيات وإجراءات** في لوحة التحكم — "
            "ستجد هناك رسائل استرجاع، عروض VIP، تنبيهات المخزون، والسلة المتروكة."
        )
    return (
        "To generate personalized messages or alerts based on your actual store data, "
        "use the ⚡ **Smart Actions** section in the dashboard — "
        "you'll find win-back messages, VIP offers, restock alerts, and abandoned cart reminders there."
    )


# ===== System Prompt =====
SYSTEM_PROMPT = """
You are an AI Copilot for e-commerce store owners.
You help them understand their store performance, identify problems, and suggest actionable decisions.

You have access to the following store data:
- Products: name, category, cost price, selling price
- Customers: name, city, registration date
- Orders: date, time, quantity, total price, status, visitors
- Campaigns: name, platform, budget, clicks, conversions, campaign_revenue, ROI

When answering questions:
1. Always provide specific numbers from the data — never make up numbers or percentages.
2. For direct questions (single metric), give a concise answer with the number + one-line context. Do NOT add extra metrics that were not asked about.
3. For diagnostic questions (why is X low/high?), provide 2-3 POSSIBLE reasons based on the data patterns. Frame them as possibilities, not conclusions: "This could be due to..." or "Possible causes include..."
4. For ML insights (forecast, churn, segmentation), ALWAYS use the actual ML results from the summary. NEVER invent segment names or percentages. Use exactly: VIP / Regular / Dormant.
5. If a benchmark is available in the context, compare the store's number to it and say if it's above or below average.
6. If data is missing, say clearly: "This metric is unavailable" — do NOT say "data not found for this period" if the data exists overall.
7. For time-period questions (e.g. "sales in April 2024", "revenue last month"), extract the date range from the question and calculate directly from the available orders data. Do NOT redirect to reports for simple single-metric questions with a date filter.
7b. For city/geography revenue questions, use the provided 'Revenue by City (ALL)' values. Do NOT say city revenue is unavailable if that line contains values.
7c. For customer distribution by city questions, use the provided 'Customers by City (ALL)' values. Do NOT use Revenue by City for customer-count questions.
7d. Never confuse customer counts with revenue values: customer distribution = number of customers, revenue distribution = SAR revenue.
8. For requests that need messages, alerts, or operational actions (win-back, restock, VIP offers, abandoned cart), redirect to the Smart Actions section — do NOT try to generate these yourself.

8b. When responding in Arabic, NEVER use the term "Smart Actions" in the user-facing answer.

8b. When responding in Arabic, always use the Arabic names of Actionlytics sections if available.

Section translations:
- Smart Actions → قسم التوصيات والإجراءات
- Reports → قسم التقارير
- Forecasting → قسم التنبؤ
- Customer Segmentation → قسم تقسيم العملاء

Never use the English section names inside Arabic answers.

Example:
Correct:
"لإرسال عرض استعادة العملاء، انتقل إلى قسم التوصيات والإجراءات."

Incorrect:
"اذهب إلى Smart Actions."

9. Respond in the same language the user uses (Arabic or English).
10. The data source is already shown above the response — do NOT repeat it again inside your answer.
11. Include the source of recommendations: if from ML model say "Based on ML analysis...", if from benchmarks say "Industry benchmark shows..."
12. At the end of every response, add exactly one line:
    - Arabic: "💡 هل تريد ملخصاً كاملاً عن أداء متجرك؟"
    - English: "💡 Would you like a full summary of your store performance?"
13. If user confirms (yes/نعم/أيوه), provide full analytics summary.
"""


# ===== Load Store Data =====
def _load_chatbot_data(db_path=None):
    if db_path and db_path != "store.db" and os.path.exists(db_path):
        import sqlite3
        try:
            conn         = sqlite3.connect(db_path)
            products_df  = pd.read_sql('SELECT * FROM products',  conn)
            customers_df = pd.read_sql('SELECT * FROM customers', conn)
            orders_df    = pd.read_sql('SELECT * FROM orders',    conn)
            campaigns_df = pd.read_sql('SELECT * FROM campaigns', conn)
            conn.close()
            orders_df = _ensure_total_price(orders_df, products_df)
            return products_df, customers_df, orders_df, campaigns_df, "Your Store"
        except Exception as e:
            print(f'Warning: could not load store db ({e}), falling back to demo store.')

    return *load_data(), "Demo Store"


# ===== Analytics Summary Cache =====
_analytics_cache = {}

def _get_cache_key(db_path):
    key = db_path or "demo"
    if db_path and os.path.exists(db_path):
        key = f"{db_path}_{os.path.getmtime(db_path)}"
    return key

def invalidate_analytics_cache(db_path=None):
    key = _get_cache_key(db_path)
    _analytics_cache.pop(key, None)


# ===== Get Analytics Summary =====
def get_analytics_summary(db_path=None):
    cache_key = _get_cache_key(db_path)
    if cache_key in _analytics_cache:
        return _analytics_cache[cache_key]

    import pandas as pd
    try:
        products_df, customers_df, orders_df, campaigns_df, data_source = _load_chatbot_data(db_path)
    except Exception as e:
        return _friendly_error(), "Demo Store"

    try:
        total_revenue        = get_total_revenue(orders_df, products_df)
        best_products        = get_best_products(orders_df, products_df)
        cancellation_rate    = get_cancellation_rate(orders_df)
        revenue_by_category  = get_revenue_by_category(orders_df, products_df)
        revenue_by_city      = get_revenue_by_city(orders_df, customers_df, products_df)
        if customers_df is not None and 'city' in customers_df.columns:
            customers_by_city = (
                customers_df['city']
                .fillna('Unknown')
                .value_counts()
                .reset_index()
            )
            customers_by_city.columns = ['city', 'customer_count']
        else:
            customers_by_city = pd.DataFrame()
        most_profitable      = get_most_profitable(products_df, orders_df)
        conversion_rate      = get_conversion_rate(orders_df)
        monthly_revenue      = get_period_comparison(orders_df, products_df)
        campaign_performance = get_campaign_performance(campaigns_df, orders_df, products_df)
        repeat_data          = get_repeat_purchase_rate(orders_df, customers_df)
        churn_df, churn_summary = predict_churn(orders_df, customers_df)
        basket_rules         = market_basket_analysis(orders_df, products_df, min_support=0.01, min_confidence=0.1)
    except Exception as e:
        return _friendly_error(), data_source

    # Safe column access
    top_product_name = best_products.iloc[0]['product_name'] if isinstance(best_products, pd.DataFrame) and len(best_products) > 0 else 'Unavailable'
    top_product_qty  = best_products.iloc[0]['quantity']     if isinstance(best_products, pd.DataFrame) and len(best_products) > 0 else 0
    # Top 5 products for richer context
    if isinstance(best_products, pd.DataFrame) and len(best_products) > 0:
        top5_products = best_products.head(5)[['product_name','quantity']].to_dict('records')
        top5_str = ' | '.join([f"{p['product_name']} ({p['quantity']} units)" for p in top5_products])
    else:
        top5_str = 'Unavailable'
    top_category     = revenue_by_category.iloc[0]['category']    if isinstance(revenue_by_category, pd.DataFrame) and len(revenue_by_category) > 0 else 'Unavailable'
    top_category_rev = revenue_by_category.iloc[0]['total_price'] if isinstance(revenue_by_category, pd.DataFrame) and len(revenue_by_category) > 0 else 0
    # All categories for full context
    if isinstance(revenue_by_category, pd.DataFrame) and len(revenue_by_category) > 0:
        all_categories_str = ' | '.join([f"{row['category']}: {row['total_price']:,.0f} SAR" for _, row in revenue_by_category.iterrows()])
    else:
        all_categories_str = 'Unavailable'

    # Revenue by city for geography-based questions
    if isinstance(revenue_by_city, pd.DataFrame) and len(revenue_by_city) > 0:
        revenue_by_city_str = ' | '.join([
            f"{row['city']}: {row['total_price']:,.0f} SAR"
            for _, row in revenue_by_city.iterrows()
        ])
    else:
        revenue_by_city_str = 'Unavailable'

    # Customer distribution by city for customer geography questions
    if isinstance(customers_by_city, pd.DataFrame) and len(customers_by_city) > 0:
        customers_by_city_str = ' | '.join([
            f"{row['city']}: {int(row['customer_count'])} customers"
            for _, row in customers_by_city.iterrows()
        ])
    else:
        customers_by_city_str = 'Unavailable'

    top_profit_name  = most_profitable.iloc[0]['product_name'] if isinstance(most_profitable, pd.DataFrame) and len(most_profitable) > 0 and 'product_name' in most_profitable.columns else 'Unavailable'
    top_profit_val   = most_profitable.iloc[0]['total_profit'] if isinstance(most_profitable, pd.DataFrame) and len(most_profitable) > 0 else 0
    best_camp_name   = campaign_performance.iloc[0]['campaign_name'] if isinstance(campaign_performance, pd.DataFrame) and len(campaign_performance) > 0 else 'Unavailable'
    best_camp_roi    = campaign_performance.iloc[0]['roi']           if isinstance(campaign_performance, pd.DataFrame) and len(campaign_performance) > 0 else 0
    # All campaigns with platforms for comparison questions
    if isinstance(campaign_performance, pd.DataFrame) and len(campaign_performance) > 0:
        camp_cols = [c for c in ['campaign_name','platform','roi','campaign_revenue','budget'] if c in campaign_performance.columns]
        all_camps_str = ' | '.join([
            f"{row.get('campaign_name','?')} ({row.get('platform','?')}) ROI:{row.get('roi',0):.0f}%"
            for _, row in campaign_performance.iterrows()
        ])
    else:
        all_camps_str = 'Unavailable'

    completed_orders = orders_df[orders_df['status'].isin(COMPLETED_STATUSES)]
    avg_order_value  = round(float(completed_orders['total_price'].mean()), 2) if len(completed_orders) > 0 else 0

    repeat_rate      = repeat_data.get('repeat_rate',        'Unavailable') if isinstance(repeat_data, dict) else 'Unavailable'
    repeat_customers = repeat_data.get('repeat_customers',   'Unavailable') if isinstance(repeat_data, dict) else 'Unavailable'
    active_customers = repeat_data.get('active_customers',   'Unavailable') if isinstance(repeat_data, dict) else 'Unavailable'
    one_time         = repeat_data.get('one_time_customers', 'Unavailable') if isinstance(repeat_data, dict) else 'Unavailable'
    never            = repeat_data.get('never_purchased',    'Unavailable') if isinstance(repeat_data, dict) else 'Unavailable'
    total_customers  = repeat_data.get('total_customers',    'Unavailable') if isinstance(repeat_data, dict) else 'Unavailable'

    store_niche = "Unknown"
    if 'category' in products_df.columns:
        categories  = products_df['category'].dropna().unique().tolist()
        store_niche = f"Store categories: {', '.join(str(c) for c in categories)}"

    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'], errors='coerce')
    orders_date_range = f"{orders_df['order_date'].min().strftime('%Y-%m-%d')} to {orders_df['order_date'].max().strftime('%Y-%m-%d')}"

    summary = f"""
Data Source: {data_source}
{store_niche}
Orders Date Range: {orders_date_range}

Current Store Analytics:
- Total Revenue: {total_revenue} SAR
- Average Order Value: {avg_order_value} SAR
- Cancellation Rate: {cancellation_rate}%
- Conversion Rate: {conversion_rate}% (calculated from actual visitors data — this is a real CVR, not orders/total ratio)
- Top Product: {top_product_name} ({top_product_qty} units)
- Top 5 Products by Sales: {top5_str}
- Revenue by Category (ALL): {all_categories_str}
- Revenue by City (ALL): {revenue_by_city_str}
- Customers by City (ALL): {customers_by_city_str}
- Most Profitable: {top_profit_name} ({top_profit_val} SAR profit)
- Best Campaign: {best_camp_name} (ROI: {best_camp_roi:.1f}%)
- All Campaigns by Platform: {all_camps_str}
- Repeat Purchase Rate: {repeat_rate}% (among {active_customers} active buyers)
  NOTE: If repeat rate exceeds 80%, flag it as statistically unusual — real-world repeat rates are typically 25-40% for most e-commerce stores.
- Repeat Customers: {repeat_customers} | One-time: {one_time} | Never Purchased: {never}
    """

    # ML Insights
    try:
        forecast_df, reliability, backtesting = forecast_sales(orders_df, products_df)
        anomalies, _                           = detect_anomalies(orders_df)
        demand_df                              = forecast_product_demand(orders_df, products_df)
        customer_stats, seg_summary            = segment_customers(orders_df, customers_df)

        seg_dict      = {row['segment_label']: row for _, row in seg_summary.iterrows()}
        vip_count     = int(seg_dict['VIP']['count'])       if 'VIP'     in seg_dict else 0
        vip_avg       = float(seg_dict['VIP']['avg_spent']) if 'VIP'     in seg_dict else 0
        regular_count = int(seg_dict['Regular']['count'])   if 'Regular' in seg_dict else 0
        dormant_count = int(seg_dict['Dormant']['count'])   if 'Dormant' in seg_dict else 0

        # VIP names — stored separately, added to context only when asked
        vip_names = customer_stats[
            customer_stats['segment_label'] == 'VIP'
        ]['customer_name'].tolist() if 'customer_name' in customer_stats.columns else []

        top_demand   = demand_df.iloc[0]['product_name']     if len(demand_df) > 0 else 'Unavailable'
        top_demand_q = demand_df.iloc[0]['predicted_demand'] if len(demand_df) > 0 else 0
        declining    = demand_df[demand_df['trend'] == 'Declining']['product_name'].tolist() if 'trend' in demand_df.columns else []

        anomaly_list = []
        for _, row in anomalies.iterrows():
            # إضافة قيمة الإيراد الفعلي للشذوذ
            month_str = row['month'].strftime('%Y-%m')
            month_revenue = ''
            try:
                orders_df['order_date'] = pd.to_datetime(orders_df['order_date'], errors='coerce')
                mask = orders_df['order_date'].dt.to_period('M').astype(str) == month_str
                month_rev_val = orders_df[mask & orders_df['status'].isin(COMPLETED_STATUSES)]['total_price'].sum()
                month_revenue = f', revenue: {month_rev_val:,.0f} SAR'
            except Exception:
                pass
            anomaly_list.append(f"{month_str}: {row['type']} ({row['severity']} severity{month_revenue})")

        basket_info = 'No patterns found'
        if len(basket_rules) > 0:
            top_rule    = basket_rules.iloc[0]
            basket_info = f"{top_rule['antecedents']} → {top_rule['consequents']} (confidence: {top_rule['confidence']:.2f})"

        # Forecast with month names
        if len(forecast_df) > 0:
            forecast_pairs   = [f"{row['month']}: {row['predicted_revenue']:,.0f} SAR" for _, row in forecast_df.iterrows()]
            forecast_display = " | ".join(forecast_pairs)
        else:
            forecast_display = "Unavailable"

        ml_summary = f"""
ML Insights (from actual models — use these numbers, do not invent):
- Forecast Reliability: {reliability}% | MAE: {backtesting.get('MAE', 'N/A')} SAR | MAPE: {backtesting.get('MAPE', 'N/A')}%
- Forecast next 4 months: {forecast_display}
- Anomalies ({len(anomalies)} detected): {'; '.join(anomaly_list) if anomaly_list else 'None'}
- Customer Segments (from KMeans clustering with RFM + recency):
  * VIP: {vip_count} customers, avg spend {vip_avg:.0f} SAR
  * Regular: {regular_count} customers
  * Dormant: {dormant_count} customers (low recency + low spend)
- Top Demand Product: {top_demand} ({top_demand_q} units predicted)
- Declining Products: {declining if declining else 'None detected'}
- Churn Risk: {churn_summary['churn_risk_count']} customers at risk out of {churn_summary['total_repeat_customers']}
  * High Risk: {churn_summary['high_risk']} | Medium Risk: {churn_summary.get('medium_risk', 0)}
  * Churn Rate: {churn_summary['churn_rate']}%
- Market Basket: {basket_info}
        """
        summary += ml_summary

    except Exception as e:
        customer_stats = pd.DataFrame()
        vip_names      = []
        summary += f"\nML Insights: Unavailable — {str(e)}\n"

    # Action Engine quick stats
    try:
        restock_actions = run_restock_alerts(db_path=db_path)
        send_time       = run_best_send_time(db_path=db_path)

        summary += f"""
Action Engine Quick Stats:
- Products needing restock: {restock_actions[0]['products_affected'] if restock_actions and restock_actions[0].get('products_affected') else 0}
- Best send time overall: {send_time[0]['top_hours'] if send_time else 'Unavailable'}
- Best time for VIP: {send_time[0]['best_per_segment'].get('VIP', 'Unavailable') if send_time else 'Unavailable'}
Note: For win-back messages, VIP offers, and abandoned cart — use Smart Actions section in dashboard.
        """
    except Exception as e:
        vip_names = vip_names if 'vip_names' in dir() else []
        summary += f"\nAction Engine: Unavailable — {str(e)}\n"

    summary += """
Report Engine:
- Reports available for: today, week, month, quarter, half_year, year, two_years, all time, or custom date range
- Reports are historical analysis only (no forecast in reports)
- For time-period questions, calculate directly from orders data above — do NOT always redirect to reports
"""

    result = (summary, data_source, vip_names if 'vip_names' in dir() else [])
    _analytics_cache[cache_key] = result
    return result


# ===== Get Uploaded Data Summary =====
def get_uploaded_summary(uploaded_df, table_name, products_df=None):
    try:
        if table_name == "orders" and products_df is not None:
            uploaded_df = _ensure_total_price(uploaded_df, products_df)

        summary = f"""
Data Source: Uploaded Data ({table_name})

Uploaded Store Data:
- Total Rows: {len(uploaded_df)}
- Columns Available: {list(uploaded_df.columns)}
"""
        if table_name == "orders":
            if "total_price" in uploaded_df.columns:
                if "status" in uploaded_df.columns:
                    completed = uploaded_df[uploaded_df["status"].isin(COMPLETED_STATUSES)]
                else:
                    completed = uploaded_df
                total_revenue = completed["total_price"].sum()
                avg_order     = completed["total_price"].mean()
                summary += f"- Total Revenue: {total_revenue:,.0f} SAR\n"
                summary += f"- Average Order Value: {avg_order:,.0f} SAR\n"
            else:
                summary += "- Total Revenue: Unavailable — total_price column not found\n"

            if "status" in uploaded_df.columns:
                cancelled         = (uploaded_df["status"] == "cancelled").sum()
                cancellation_rate = round(cancelled / len(uploaded_df) * 100, 2)
                summary += f"- Cancellation Rate: {cancellation_rate}%\n"

            if "order_date" in uploaded_df.columns:
                summary += f"- Date Range: {uploaded_df['order_date'].min()} to {uploaded_df['order_date'].max()}\n"

        elif table_name == "customers":
            if "city" in uploaded_df.columns:
                top_city = uploaded_df["city"].value_counts().index[0]
                summary += f"- Top City: {top_city}\n"
                summary += f"- Cities: {uploaded_df['city'].nunique()} unique cities\n"

        elif table_name == "products":
            if "category" in uploaded_df.columns:
                summary += f"- Categories: {uploaded_df['category'].unique().tolist()}\n"
            price_col = "selling_price" if "selling_price" in uploaded_df.columns else "price"
            if price_col in uploaded_df.columns:
                summary += f"- Price Range: {uploaded_df[price_col].min()} - {uploaded_df[price_col].max()} SAR\n"

        elif table_name == "campaigns":
            if "campaign_revenue" in uploaded_df.columns and "budget" in uploaded_df.columns:
                uploaded_df['roi'] = ((uploaded_df['campaign_revenue'] - uploaded_df['budget']) / uploaded_df['budget'] * 100).round(2)
                summary += f"- Average ROI: {uploaded_df['roi'].mean():.1f}%\n"
            elif "roi" in uploaded_df.columns:
                summary += f"- Average ROI: {uploaded_df['roi'].mean():.1f}%\n"
            else:
                summary += "- ROI: Unavailable — campaign_revenue data not found\n"
            if "platform" in uploaded_df.columns:
                summary += f"- Platforms: {uploaded_df['platform'].unique().tolist()}\n"

        return summary
    except Exception as e:
        return f"Data Source: Uploaded Data\n\nUnavailable — error processing uploaded data: {e}"


# ===== ask_actionlytics =====
def ask_actionlytics(question, conversation_history,
                     uploaded_df=None, uploaded_table=None,
                     db_path=None, products_df=None):
    # Guard against None or empty question
    if not question or not str(question).strip():
        return "", conversation_history
    try:
        lang = _detect_language(str(question))

        if _needs_action_engine(question):
            redirect = _action_engine_redirect(lang)
            conversation_history.append({"role": "user",      "content": question})
            conversation_history.append({"role": "assistant", "content": redirect})
            return redirect, conversation_history

        if uploaded_df is not None and uploaded_table is not None:
            analytics_summary = get_uploaded_summary(uploaded_df, uploaded_table, products_df=products_df)
            data_source       = "Uploaded Data"
            data_note         = "You are analyzing data uploaded by the store owner. Focus on this data only."
            vip_names         = []
        else:
            result = get_analytics_summary(db_path=db_path)
            if isinstance(result, tuple) and len(result) == 3:
                analytics_summary, data_source, vip_names = result
            elif isinstance(result, tuple) and len(result) == 2:
                analytics_summary, data_source = result
                vip_names = []
            else:
                analytics_summary = result
                data_source       = "Demo Store"
                vip_names         = []
            data_note = f"Data source: {data_source}"

        # Add VIP names to context only when asked
        vip_section = ""
        if _needs_vip_names(question) and vip_names:
            vip_section = f"\nVIP Customer Names (use only when asked): {', '.join(vip_names[:20])}"

        # Conditional RAG
        if _needs_rag(question) and rag_vectorstore is not None:
            rag_context = query_rag(rag_vectorstore, question)
            rag_section = f"\nMarket Benchmarks & Best Practices:\n{rag_context}"
        else:
            rag_section = ""

        # History limit — sliding window آخر 10 رسائل
        conversation_history.append({"role": "user", "content": question})
        trimmed_history = conversation_history[-MAX_HISTORY:] if len(conversation_history) > MAX_HISTORY else conversation_history

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            system=f"{SYSTEM_PROMPT}\n\n{data_note}\n\n{analytics_summary}{vip_section}{rag_section}",
            messages=trimmed_history
        )

        # Data source prefix — ثابت من Python مو من الـ LLM
        clean_answer = response.content[0].text
        data_prefix  = f"📁 **{data_source}**\n\n"
        answer       = data_prefix + clean_answer

        # نحفظ بدون prefix في الـ history — منع التكرار
        conversation_history.append({"role": "assistant", "content": clean_answer})
        return answer, conversation_history
    except Exception as e:
        lang  = _detect_language(question)
        error = _friendly_error(lang)
        conversation_history.append({"role": "user",      "content": question})
        conversation_history.append({"role": "assistant", "content": error})
        return error, conversation_history


# ===== Main =====
if __name__ == '__main__':
    print('Actionlytics is ready! Type your question or "exit" to quit.')
    print('=' * 50)

    conversation_history = []
    while True:
        try:
            question = input('\nYou: ')
        except KeyboardInterrupt:
            print('\nGoodbye!')
            break

        if question.lower() == 'exit':
            break
        if not question.strip():
            continue

        print('\nActionlytics: Thinking...')
        answer, conversation_history = ask_actionlytics(question, conversation_history)
        print(f'\nActionlytics: {answer}')
        print('-' * 50)