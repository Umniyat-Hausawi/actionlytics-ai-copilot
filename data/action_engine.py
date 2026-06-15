import sqlite3
import pandas as pd
import anthropic
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from ml_engine import predict_churn, segment_customers
from analytics import load_data, _ensure_total_price, COMPLETED_STATUSES

# ===== Config =====
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
DEFAULT_DB_PATH = "store.db"
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL  = "claude-haiku-4-5-20251001"  # Haiku: أرخص بـ 80% من Opus — كافي للرسائل


# ===== User-friendly error =====
def _friendly_error(context="action"):
    return f"Sorry, something went wrong while running {context}. Please try again."


# ===== Priority Calculator =====
def calculate_priority(impact, urgency):
    """
    impact x urgency matrix — returns 1-9 score.
    High=3, Medium=2, Low=1
    """
    scores = {'High': 3, 'Medium': 2, 'Low': 1}
    return scores.get(impact, 1) * scores.get(urgency, 1)


def get_priority_label(score):
    if score >= 7:
        return '🔴 Critical'
    elif score >= 4:
        return '🟡 High'
    else:
        return '🟢 Normal'


# ===== Load Store Data =====
def load_store_data_for_actions(db_path=None):
    target_db   = db_path if db_path else DEFAULT_DB_PATH
    data_source = "Your Store"
    try:
        conn         = sqlite3.connect(target_db)
        products_df  = pd.read_sql('SELECT * FROM products',  conn)
        customers_df = pd.read_sql('SELECT * FROM customers', conn)
        orders_df    = pd.read_sql('SELECT * FROM orders',    conn)
        campaigns_df = pd.read_sql('SELECT * FROM campaigns', conn)
        conn.close()
    except Exception as e:
        print(f'Warning: could not load store db ({e}), falling back to demo store.')
        products_df, customers_df, orders_df, campaigns_df = load_data()
        data_source = "Demo Store"

    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    orders_df = _ensure_total_price(orders_df, products_df)
    return products_df, customers_df, orders_df, campaigns_df, data_source


# ===== Load Carts =====
def load_carts(db_path=None):
    target_db = db_path if db_path else DEFAULT_DB_PATH
    try:
        conn     = sqlite3.connect(target_db)
        carts_df = pd.read_sql('SELECT * FROM carts', conn)
        conn.close()
        return carts_df
    except Exception as e:
        print(f'Warning: could not load carts table: {e}')
        return pd.DataFrame()


# ===== All Inventory Status =====
def get_all_inventory_status(products_df, orders_df, threshold=20):
    stock_col = 'stock_quantity' if 'stock_quantity' in products_df.columns else 'stock'
    price_col = 'selling_price'  if 'selling_price'  in products_df.columns else 'price'
    name_col  = 'product_name'   if 'product_name'   in products_df.columns else 'name'

    if stock_col not in products_df.columns:
        return pd.DataFrame()

    cutoff   = pd.Timestamp.now() - pd.Timedelta(days=30)
    recent   = orders_df[
        (orders_df['order_date'] >= cutoff) &
        (orders_df['status'].isin(COMPLETED_STATUSES))
    ]
    velocity = recent.groupby('product_id')['quantity'].sum().reset_index()
    velocity.columns = ['product_id', 'sold_last_30days']

    merged = products_df.copy().merge(velocity, on='product_id', how='left')
    merged['sold_last_30days'] = merged['sold_last_30days'].fillna(0)
    merged['daily_velocity']   = (merged['sold_last_30days'] / 30).round(2)
    merged['days_to_stockout']  = merged.apply(
        lambda r: int(r[stock_col] / r['daily_velocity']) if r['daily_velocity'] > 0 else 999,
        axis=1
    )

    def get_status(row):
        if row[stock_col] == 0:          return 'OUT OF STOCK'
        elif row[stock_col] <= threshold / 2: return 'CRITICAL'
        elif row[stock_col] <= threshold:     return 'LOW'
        else:                                 return 'OK'

    merged['status'] = merged.apply(get_status, axis=1)

    result = merged[['product_id', name_col, 'category', stock_col,
                      'status', 'days_to_stockout', 'sold_last_30days']].copy()
    result = result.rename(columns={name_col: 'product_name', stock_col: 'stock_quantity'})

    status_order = {'OUT OF STOCK': 0, 'CRITICAL': 1, 'LOW': 2, 'OK': 3}
    result['_sort'] = result['status'].map(status_order)
    result = result.sort_values('_sort').drop(columns=['_sort'])
    return result


# ===== Low Stock Products =====
def get_low_stock_products(products_df, orders_df, threshold=20):
    all_inventory = get_all_inventory_status(products_df, orders_df, threshold)
    if all_inventory.empty:
        return pd.DataFrame()
    return all_inventory[all_inventory['status'].isin(['OUT OF STOCK', 'CRITICAL', 'LOW'])]


# ===== Top Products =====
def get_top_products_from_df(products_df, orders_df, limit=5):
    name_col  = 'product_name' if 'product_name' in products_df.columns else 'name'
    price_col = 'selling_price' if 'selling_price' in products_df.columns else 'price'

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    recent = orders_df[
        (orders_df['order_date'] >= cutoff) &
        (orders_df['status'].isin(COMPLETED_STATUSES))
    ]
    if recent.empty:
        recent = orders_df[orders_df['status'].isin(COMPLETED_STATUSES)]

    sales  = recent.groupby('product_id').agg(
        units_sold=('quantity', 'sum'),
        revenue=('total_price', 'sum')
    ).reset_index()

    cols   = [c for c in ['product_id', name_col, 'category', price_col] if c in products_df.columns]
    merged = sales.merge(products_df[cols], on='product_id', how='left')
    merged = merged.rename(columns={name_col: 'product_name', price_col: 'price'})
    return merged.sort_values('revenue', ascending=False).head(limit)


# ===== Campaign Performance =====
def get_campaign_performance_from_df(campaigns_df):
    df       = campaigns_df.copy()
    name_col = 'campaign_name' if 'campaign_name' in df.columns else 'name'
    if name_col == 'name':
        df = df.rename(columns={'name': 'campaign_name'})

    has_revenue = (
        'campaign_revenue' in df.columns and
        'budget' in df.columns and
        df['budget'].sum() > 0 and
        df['campaign_revenue'].sum() > 0
    )

    if not has_revenue:
        df['roi']                 = None
        df['roi_note']            = 'Campaign revenue data unavailable — cannot calculate ROI'
        df['cost_per_click']      = None
        df['cost_per_conversion'] = None
        df['data_status']         = 'incomplete'
        return df

    df['roi']     = ((df['campaign_revenue'] - df['budget']) / df['budget'] * 100).round(2)
    df['roi_note'] = ''

    if 'clicks' in df.columns:
        df['cost_per_click'] = (df['budget'] / df['clicks'].replace(0, 1)).round(2)
    else:
        df['cost_per_click'] = None

    if 'conversions' in df.columns:
        df['cost_per_conversion'] = (df['budget'] / df['conversions'].replace(0, 1)).round(2)
    else:
        df['cost_per_conversion'] = None

    df['data_status'] = 'complete'
    return df.sort_values('roi', ascending=False)


# ===== Smart Discount Recommender =====
def recommend_discount(days_since_last, total_orders, total_spent):
    r_score   = 3 if days_since_last <= 30  else (2 if days_since_last <= 90 else 1)
    f_score   = 3 if total_orders    >= 8   else (2 if total_orders    >= 4  else 1)
    m_score   = 3 if total_spent     >= 3000 else (2 if total_spent    >= 1000 else 1)
    rfm_score = r_score + f_score + m_score

    if rfm_score >= 8:
        return 5,  "VIP customer — small incentive to maintain loyalty"
    elif rfm_score >= 6:
        return 10, "Good customer — moderate discount to re-engage"
    elif rfm_score >= 4:
        return 15, "At-risk customer — strong discount to win back"
    else:
        return 20, "Dormant customer — maximum discount to recover"


# ===== LLM Action Generator =====
def generate_action(action_type, context, language="ar", tone="friendly"):
    lang_instruction  = "Respond in Arabic only." if language == "ar" else "Respond in English only."
    tone_instructions = {
        "friendly": "Use a warm and friendly tone.",
        "formal":   "Use a professional and formal tone.",
        "urgent":   "Use an urgent tone — limited time offer.",
        "casual":   "Use a casual and conversational tone."
    }
    tone_instruction = tone_instructions.get(tone, tone_instructions["friendly"])

    prompts = {
        "churn_reminder": f"""
You are an e-commerce Business Intelligence AI.
A customer has not purchased in a while.

Customer data:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return a JSON object with exactly two keys:
1. "business_recommendation": object with keys:
   - "problem": one sentence describing the issue
   - "evidence": specific numbers from the data
   - "recommendation": one clear action to take
   - "expected_outcome": expected result based on industry benchmarks (re-engagement campaigns recover 15-25% of at-risk customers)
   - "impact": "High" or "Medium" or "Low"
   - "urgency": "High" or "Medium" or "Low"
2. "customer_message": short personalized message to send to this customer (under 100 words), include greeting, how long away, discount {context.get('recommended_discount')}%, one call-to-action. Do NOT mention discount codes.

Return ONLY valid JSON, no markdown, no explanation.
{lang_instruction}
{tone_instruction}
""",
        "restock_alert": f"""
You are a supply chain Business Intelligence AI.

Products data:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return a JSON object with exactly two keys:
1. "business_recommendation": object with keys:
   - "problem": one sentence describing the stock situation
   - "evidence": specific stock numbers and days to stockout
   - "recommendation": specific restock action with quantities
   - "expected_outcome": impact if not restocked (lost revenue estimate)
   - "impact": "High" or "Medium" or "Low"
   - "urgency": "High" or "Medium" or "Low"
2. "customer_message": internal restock alert for the store manager, concise and actionable, include priority per product and recommended restock quantity.

Return ONLY valid JSON, no markdown, no explanation.
{lang_instruction}
{tone_instruction}
""",
        "vip_offer": f"""
You are an e-commerce Business Intelligence AI.

Customer data:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return a JSON object with exactly two keys:
1. "business_recommendation": object with keys:
   - "problem": opportunity statement (VIP retention)
   - "evidence": customer loyalty stats (orders, spent)
   - "recommendation": specific retention action
   - "expected_outcome": expected result (VIP customers spend 3-5x more than regular)
   - "impact": "High" or "Medium" or "Low"
   - "urgency": "High" or "Medium" or "Low"
2. "customer_message": exclusive VIP offer message under 150 words, include VIP greeting, loyalty stats, discount {context.get('recommended_discount')}%, top product recommendations. Do NOT mention discount codes.

Return ONLY valid JSON, no markdown, no explanation.
{lang_instruction}
{tone_instruction}
""",
        "abandoned_cart": f"""
You are an e-commerce Business Intelligence AI.

Cart data:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return a JSON object with exactly two keys:
1. "business_recommendation": object with keys:
   - "problem": cart abandonment issue
   - "evidence": cart value and customer history
   - "recommendation": recovery action
   - "expected_outcome": recovery rate estimate (20-30% cart recovery with timely follow-up)
   - "impact": "High" or "Medium" or "Low"
   - "urgency": "High" or "Medium" or "Low"
2. "customer_message": short reminder message under 100 words. If recommended_discount is 0: warm reminder no discount. If 5%: mention as loyalty thank you. Do NOT mention discount codes.

Return ONLY valid JSON, no markdown, no explanation.
{lang_instruction}
{tone_instruction}
""",
        "campaign": f"""
You are an e-commerce Business Intelligence AI.

Campaign data:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return a JSON object with exactly two keys:
1. "business_recommendation": object with keys:
   - "problem": campaign performance issue or opportunity
   - "evidence": specific ROI numbers or data gaps
   - "recommendation": one specific actionable recommendation
   - "expected_outcome": expected improvement
   - "impact": "High" or "Medium" or "Low"
   - "urgency": "High" or "Medium" or "Low"
2. "customer_message": concise campaign analysis for the store owner. If data incomplete, state tracking unavailable and give general recommendations. If complete, give verdict per campaign.

Return ONLY valid JSON, no markdown, no explanation.
{lang_instruction}
{tone_instruction}
"""
    }

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompts[action_type]}]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError:
        # Fallback: return raw text as message if JSON parsing fails
        return {
            "business_recommendation": {
                "problem": "Analysis unavailable",
                "evidence": "",
                "recommendation": "",
                "expected_outcome": "",
                "impact": "Low",
                "urgency": "Low"
            },
            "customer_message": response.content[0].text if 'response' in dir() else _friendly_error(action_type)
        }
    except Exception as e:
        return {
            "business_recommendation": {
                "problem": f"Error: {e}",
                "evidence": "", "recommendation": "",
                "expected_outcome": "", "impact": "Low", "urgency": "Low"
            },
            "customer_message": _friendly_error(action_type)
        }


# ===== Action Runners =====


# ===== Generate Message On Demand (Lazy) =====
def generate_message_on_demand(action_type, context, language="ar", tone="friendly"):
    """يُولَّد الرسالة فقط عند الطلب — بدل توليد كل الرسائل مرة وحدة"""
    try:
        action = generate_action(action_type, context, language, tone)
        rec    = action.get("business_recommendation", {})
        return {
            "message":          action.get("customer_message", ""),
            "problem":          rec.get("problem", ""),
            "evidence":         rec.get("evidence", ""),
            "recommendation":   rec.get("recommendation", ""),
            "expected_outcome": rec.get("expected_outcome", "")
        }
    except Exception as e:
        return {
            "message":          _friendly_error(action_type),
            "problem":          "",
            "evidence":         "",
            "recommendation":   "",
            "expected_outcome": ""
        }


def run_churn_reminders(language="ar", tone="friendly", limit=None, db_path=None):  # limit=None = عرض الكل
    try:
        products_df, customers_df, orders_df, campaigns_df, data_source = load_store_data_for_actions(db_path)
        churn_df, summary = predict_churn(orders_df, customers_df)

        if len(churn_df) == 0:
            return []

        results = []
        for _, customer in (churn_df.head(limit) if limit else churn_df).iterrows():
            discount, reason = recommend_discount(
                days_since_last=int(customer["days_since_last"]),
                total_orders=int(customer.get("order_count", 1)),
                total_spent=float(customer.get("total_spent", 0))
            )
            risk = customer["risk_level"]
            impact  = "High" if risk == "High" else "Medium"
            urgency = "High" if int(customer["days_since_last"]) > 90 else "Medium"
            priority_score = calculate_priority(impact, urgency)

            results.append({
                "customer_id":           customer["customer_id"],
                "customer_name":         customer["customer_name"],
                "risk_level":            risk,
                "days_inactive":         int(customer["days_since_last"]),
                "recommended_discount":  discount,
                "category":              "Customer Retention",
                "priority_score":        priority_score,
                "priority_label":        get_priority_label(priority_score),
                "problem":               f"لم يشتر منذ {int(customer['days_since_last'])} يوم" if language=="ar" else f"Inactive for {int(customer['days_since_last'])} days",
                "evidence":              "",
                "recommendation":        "",
                "expected_outcome":      "",
                "message":               None,
                "language":              language,
                "tone":                  tone,
                "context": {
                    "customer_name":        customer["customer_name"],
                    "city":                 customer["city"],
                    "days_since_last":      int(customer["days_since_last"]),
                    "avg_days_between":     round(float(customer["avg_days_between"]), 1),
                    "risk_level":           risk,
                    "recommended_discount": discount,
                    "discount_reason":      reason
                }
            })

        return sorted(results, key=lambda x: x["priority_score"], reverse=True)
    except Exception as e:
        print(f'Warning: run_churn_reminders failed: {e}')
        return []


def run_restock_alerts(language="ar", tone="formal", threshold=20, db_path=None):
    try:
        products_df, customers_df, orders_df, campaigns_df, data_source = load_store_data_for_actions(db_path)

        full_inventory = get_all_inventory_status(products_df, orders_df, threshold)
        low_stock      = full_inventory[full_inventory['status'].isin(['OUT OF STOCK', 'CRITICAL', 'LOW'])]

        if full_inventory.empty:
            return [{"message": "No inventory data available.", "full_inventory": pd.DataFrame(), "products_detail": []}]

        if low_stock.empty:
            return [{
                "message":         "✅ All products are well-stocked. No restock needed at this time.",
                "full_inventory":  full_inventory,
                "products_detail": [],
                "products_affected": 0,
                "category":        "Inventory Management",
                "priority_score":  0,
                "priority_label":  "🟢 Normal"
            }]

        products_context = []
        for _, product in low_stock.iterrows():
            products_context.append({
                "product_name":        product["product_name"],
                "category":            product["category"],
                "current_stock":       int(product["stock_quantity"]),
                "status":              product["status"],
                "days_to_stockout":    int(product["days_to_stockout"]) if product["days_to_stockout"] < 999 else "N/A",
                "sold_last_30days":    int(product["sold_last_30days"]),
                "recommended_restock": 30
            })

        # Determine urgency based on worst status
        has_out_of_stock = any(p["status"] == "OUT OF STOCK" for p in products_context)
        has_critical     = any(p["status"] == "CRITICAL"     for p in products_context)
        impact  = "High"   if has_out_of_stock or has_critical else "Medium"
        urgency = "High"   if has_out_of_stock else ("Medium" if has_critical else "Low")

        action         = generate_action("restock_alert", {"products": products_context}, language, tone)
        rec            = action.get("business_recommendation", {})
        priority_score = calculate_priority(impact, urgency)

        return [{
            "products_affected": len(products_context),
            "products_detail":   products_context,
            "full_inventory":    full_inventory,
            "category":          "Inventory Management",
            "priority_score":    priority_score,
            "priority_label":    get_priority_label(priority_score),
            "problem":           rec.get("problem", ""),
            "evidence":          rec.get("evidence", ""),
            "recommendation":    rec.get("recommendation", ""),
            "expected_outcome":  rec.get("expected_outcome", ""),
            "message":           action.get("customer_message", "")
        }]
    except Exception as e:
        print(f'Warning: run_restock_alerts failed: {e}')
        return []


def run_vip_offers(language="ar", tone="formal", limit=None, db_path=None):  # limit=None = عرض الكل
    try:
        products_df, customers_df, orders_df, campaigns_df, data_source = load_store_data_for_actions(db_path)
        customer_stats, summary = segment_customers(orders_df, customers_df)
        vip_customers = customer_stats[customer_stats["segment_label"] == "VIP"] if limit is None else customer_stats[customer_stats["segment_label"] == "VIP"].head(limit)
        top_products  = get_top_products_from_df(products_df, orders_df)

        if vip_customers.empty:
            return []

        results = []
        for _, customer in vip_customers.iterrows():
            discount, reason = recommend_discount(
                days_since_last=0,
                total_orders=int(customer["order_count"]),
                total_spent=float(customer["total_spent"])
            )
            priority_score = calculate_priority("High", "Medium")

            results.append({
                "customer_id":          customer["customer_id"],
                "customer_name":        customer["customer_name"],
                "total_spent":          float(customer["total_spent"]),
                "recommended_discount": discount,
                "category":             "Revenue Growth",
                "priority_score":       priority_score,
                "priority_label":       get_priority_label(priority_score),
                "problem":              "عميل VIP يحتاج عرض حصري" if language=="ar" else "VIP customer needs exclusive offer",
                "evidence":             "",
                "recommendation":       "",
                "expected_outcome":     "",
                "message":              None,
                "language":             language,
                "tone":                 tone,
                "context": {
                    "customer_name":        customer["customer_name"],
                    "city":                 customer["city"],
                    "total_orders":         int(customer["order_count"]),
                    "total_spent":          float(customer["total_spent"]),
                    "recommended_discount": discount,
                    "discount_reason":      reason,
                    "top_products":         top_products["product_name"].tolist()
                }
            })

        return sorted(results, key=lambda x: x["priority_score"], reverse=True)
    except Exception as e:
        print(f'Warning: run_vip_offers failed: {e}')
        return []


def run_abandoned_cart_action(language="ar", tone="friendly", limit=None, db_path=None):  # limit=None = عرض الكل
    try:
        products_df, customers_df, orders_df, campaigns_df, data_source = load_store_data_for_actions(db_path)
        carts = load_carts(db_path)

        if carts.empty:
            return []

        name_col  = 'customer_name' if 'customer_name' in customers_df.columns else 'name'
        price_col = 'selling_price' if 'selling_price' in products_df.columns else 'price'
        pname_col = 'product_name'  if 'product_name'  in products_df.columns else 'name'

        abandoned = carts[carts['status'] == 'abandoned'] if limit is None else carts[carts['status'] == 'abandoned'].head(limit)
        if abandoned.empty:
            return []

        merged = abandoned.merge(
            customers_df[['customer_id', name_col, 'city']], on='customer_id', how='left'
        ).merge(
            products_df[['product_id', pname_col, 'category', price_col]], on='product_id', how='left'
        )
        merged = merged.rename(columns={
            name_col: 'customer_name', pname_col: 'product_name', price_col: 'selling_price'
        })

        results = []
        for _, cart in merged.iterrows():
            total_orders = int(cart.get("total_orders", 0))
            discount     = 5 if total_orders >= 2 else 0
            cart_value   = round(float(cart.get("selling_price", 0)) * int(cart.get("quantity", 1)), 2)
            priority_score = calculate_priority("Medium", "High")

            results.append({
                "cart_id":              cart.get("cart_id", ""),
                "customer_name":        cart.get("customer_name", ""),
                "product_name":         cart.get("product_name", ""),
                "cart_value":           cart_value,
                "total_orders":         total_orders,
                "recommended_discount": discount,
                "category":             "Revenue Growth",
                "priority_score":       priority_score,
                "priority_label":       get_priority_label(priority_score),
                "problem":              "سلة متروكة" if language=="ar" else "Abandoned cart",
                "evidence":             "",
                "recommendation":       "",
                "expected_outcome":     "",
                "message":              None,
                "language":             language,
                "tone":                 tone,
                "context": {
                    "customer_name":         cart.get("customer_name", ""),
                    "city":                  cart.get("city", ""),
                    "product_name":          cart.get("product_name", ""),
                    "category":              cart.get("category", ""),
                    "quantity":              int(cart.get("quantity", 1)),
                    "cart_value":            cart_value,
                    "total_previous_orders": total_orders,
                    "recommended_discount":  discount
                }
            })

        return sorted(results, key=lambda x: x["priority_score"], reverse=True)
    except Exception as e:
        print(f'Warning: run_abandoned_cart_action failed: {e}')
        return []


def run_campaign_action(language="ar", tone="formal", db_path=None):
    try:
        products_df, customers_df, orders_df, campaigns_df, data_source = load_store_data_for_actions(db_path)
        campaigns = get_campaign_performance_from_df(campaigns_df)

        if campaigns.empty:
            return []

        data_complete = campaigns.get('data_status', pd.Series(['incomplete'])).iloc[0] == 'complete'

        campaigns_context = []
        for _, campaign in campaigns.iterrows():
            campaigns_context.append({
                "campaign_name":       str(campaign.get("campaign_name", "")),
                "platform":            str(campaign.get("platform", "N/A")),
                "budget":              float(campaign.get("budget", 0)),
                "clicks":              int(campaign.get("clicks", 0)),
                "conversions":         int(campaign.get("conversions", 0)),
                "campaign_revenue":    float(campaign.get("campaign_revenue", 0)) if "campaign_revenue" in campaign else 0,
                "roi":                 float(campaign.get("roi", 0)) if campaign.get("roi") is not None else None,
                "data_status":         str(campaign.get("data_status", "incomplete")),
                "cost_per_click":      float(campaign.get("cost_per_click", 0)) if campaign.get("cost_per_click") is not None else None,
                "cost_per_conversion": float(campaign.get("cost_per_conversion", 0)) if campaign.get("cost_per_conversion") is not None else None
            })

        action  = generate_action("campaign", {"campaigns": campaigns_context, "data_complete": data_complete}, language, tone)
        rec     = action.get("business_recommendation", {})
        best    = campaigns.iloc[0]

        priority_score = calculate_priority(
            rec.get("impact", "Medium"),
            rec.get("urgency", "Medium")
        )

        return [{
            "campaigns_analyzed": len(campaigns_context),
            "best_campaign":      str(best.get("campaign_name", "")),
            "best_platform":      str(best.get("platform", "N/A")),
            "data_complete":      data_complete,
            "category":           "Marketing Optimization",
            "priority_score":     priority_score,
            "priority_label":     get_priority_label(priority_score),
            "problem":            rec.get("problem", ""),
            "evidence":           rec.get("evidence", ""),
            "recommendation":     rec.get("recommendation", ""),
            "expected_outcome":   rec.get("expected_outcome", ""),
            "message":            action.get("customer_message", "")
        }]
    except Exception as e:
        print(f'Warning: run_campaign_action failed: {e}')
        return []


def run_best_send_time(language="ar", db_path=None):
    try:
        products_df, customers_df, orders_df, campaigns_df, data_source = load_store_data_for_actions(db_path)

        if 'order_time' not in orders_df.columns:
            return []

        completed = orders_df[orders_df['status'].isin(COMPLETED_STATUSES)].copy()
        completed['hour'] = pd.to_datetime(completed['order_time'], format='%H:%M', errors='coerce').dt.hour
        completed = completed.dropna(subset=['hour'])

        hourly    = completed.groupby('hour')['order_id'].count().reset_index()
        hourly.columns = ['hour', 'order_count']
        top_hours = hourly.sort_values('order_count', ascending=False).head(3)['hour'].tolist()

        customer_stats, _ = segment_customers(orders_df, customers_df)
        merged            = completed.merge(customer_stats[['customer_id', 'segment_label']], on='customer_id', how='left')
        segment_hours     = merged.groupby(['segment_label', 'hour'])['order_id'].count().reset_index()

        best_per_segment = {}
        for segment in ['VIP', 'Regular', 'Dormant']:
            seg_data = segment_hours[segment_hours['segment_label'] == segment]
            if not seg_data.empty:
                best_hour = seg_data.loc[seg_data['order_id'].idxmax(), 'hour']
                best_per_segment[segment] = f"{int(best_hour):02d}:00"

        context          = {
            "top_hours_overall":     [f"{h:02d}:00" for h in top_hours],
            "best_time_per_segment": best_per_segment
        }
        lang_instruction = "Respond in Arabic only." if language == "ar" else "Respond in English only."
        prompt           = f"""
You are an e-commerce AI assistant.
Based on customer purchase time analysis, provide recommendations on the best time to send marketing messages.

Data:
{json.dumps(context, ensure_ascii=False, indent=2)}

Include:
1. Best overall time to send messages
2. Best time per customer segment (VIP, Regular, Dormant)
3. One actionable recommendation for the store owner

{lang_instruction}
Keep it concise and practical. Under 150 words.
"""
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return [{
            "top_hours":        context["top_hours_overall"],
            "best_per_segment": best_per_segment,
            "message":          response.content[0].text
        }]
    except Exception as e:
        print(f'Warning: run_best_send_time failed: {e}')
        return []


# ===== Main =====
if __name__ == "__main__":
    try:
        print("===== Churn Reminders =====")
        for r in run_churn_reminders(language="ar", limit=2):
            print(f"\n{r['customer_name']} | {r['risk_level']} | {r['days_inactive']} days")
            print(f"Category: {r['category']} | Priority: {r['priority_label']} ({r['priority_score']})")
            print(f"Problem: {r['problem']}")
            print(f"Recommendation: {r['recommendation']}")
            print(f"Expected: {r['expected_outcome']}")
            print(f"Message: {r['message']}")

        print("\n===== Restock Alerts =====")
        for r in run_restock_alerts(language="ar"):
            print(f"\nProducts affected: {r.get('products_affected', 0)}")
            print(f"Category: {r['category']} | Priority: {r['priority_label']} ({r['priority_score']})")
            print(f"Problem: {r['problem']}")
            print(f"Recommendation: {r['recommendation']}")
            print(f"Message: {r['message']}")

        print("\n===== VIP Offers =====")
        for r in run_vip_offers(language="ar", limit=2):
            print(f"\n{r['customer_name']} | {r['total_spent']} SAR")
            print(f"Category: {r['category']} | Priority: {r['priority_label']} ({r['priority_score']})")
            print(f"Problem: {r['problem']}")
            print(f"Recommendation: {r['recommendation']}")
            print(f"Message: {r['message']}")

        print("\n===== Campaign Action =====")
        for r in run_campaign_action(language="ar"):
            print(f"\nBest: {r['best_campaign']} | Data complete: {r['data_complete']}")
            print(f"Category: {r['category']} | Priority: {r['priority_label']} ({r['priority_score']})")
            print(f"Problem: {r['problem']}")
            print(f"Recommendation: {r['recommendation']}")
            print(f"Message: {r['message']}")

    except Exception as e:
        print(f"Action engine failed: {e}")