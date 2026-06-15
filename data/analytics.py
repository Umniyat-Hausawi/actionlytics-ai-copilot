import pandas as pd
import sqlite3
import os

# ===== Constants =====
COMPLETED_STATUSES = ['completed', 'delivered']
CANCELLED_STATUS   = 'cancelled'


# ===== Price Column Helper =====
def _get_price_col(df):
    if 'selling_price' in df.columns:
        return 'selling_price'
    if 'price' in df.columns:
        return 'price'
    return None


# ===== Load Data =====
def load_data():
    try:
        db_path = os.path.join(os.path.dirname(__file__), 'store.db')
        conn = sqlite3.connect(db_path)
        products_df  = pd.read_sql('SELECT * FROM products',  conn)
        customers_df = pd.read_sql('SELECT * FROM customers', conn)
        orders_df    = pd.read_sql('SELECT * FROM orders',    conn)
        campaigns_df = pd.read_sql('SELECT * FROM campaigns', conn)
        conn.close()
        return products_df, customers_df, orders_df, campaigns_df
    except Exception as e:
        raise RuntimeError(f'Failed to load data: {e}')


# ===== Helper: ensure total_price column exists =====
def _ensure_total_price(orders_df, products_df=None):
    df = orders_df.copy()
    if 'total_price' in df.columns:
        return df

    price_col = _get_price_col(products_df) if products_df is not None else None

    if products_df is not None and price_col:
        price_map = products_df.set_index('product_id')[price_col]
        df['unit_price']  = df['product_id'].map(price_map).fillna(0)
        df['total_price'] = df['quantity'] * df['unit_price']
    else:
        df['total_price'] = df.get('quantity', 0)

    return df


# ===== Total Revenue =====
def get_total_revenue(orders_df, products_df=None, start_date=None, end_date=None):
    try:
        df = _ensure_total_price(orders_df, products_df)
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]
        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        return round(completed['total_price'].sum(), 2)
    except Exception as e:
        return f'Error calculating total revenue: {e}'


# ===== Best Selling Products =====
def get_best_products(orders_df, products_df, start_date=None, end_date=None):
    try:
        df = orders_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        sales = completed.groupby('product_id')['quantity'].sum().reset_index()

        name_col = 'product_name' if 'product_name' in products_df.columns else 'name'
        cols = [c for c in ['product_id', name_col, 'category'] if c in products_df.columns]
        sales = sales.merge(products_df[cols], on='product_id', how='left')

        if 'product_name' not in sales.columns and 'name' in sales.columns:
            sales = sales.rename(columns={'name': 'product_name'})

        return sales.sort_values('quantity', ascending=False)
    except Exception as e:
        return f'Error calculating best products: {e}'


# ===== Cancellation Rate =====
def get_cancellation_rate(orders_df, start_date=None, end_date=None):
    try:
        df = orders_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        total = len(df)
        if total == 0:
            return 0
        cancelled = len(df[df['status'] == CANCELLED_STATUS])
        return round((cancelled / total) * 100, 2)
    except Exception as e:
        return f'Error calculating cancellation rate: {e}'


# ===== Revenue by Category =====
def get_revenue_by_category(orders_df, products_df, start_date=None, end_date=None):
    try:
        df = _ensure_total_price(orders_df, products_df)
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        cols      = [c for c in ['product_id', 'category'] if c in products_df.columns]
        merged    = completed.merge(products_df[cols], on='product_id', how='left')

        if 'category' not in merged.columns:
            merged['category'] = 'Unknown'

        revenue = merged.groupby('category')['total_price'].sum().reset_index()
        return revenue.sort_values('total_price', ascending=False)
    except Exception as e:
        return f'Error calculating revenue by category: {e}'


# ===== Most Profitable Products =====
def get_most_profitable(products_df, orders_df, start_date=None, end_date=None):
    try:
        df = orders_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        sales     = completed.groupby('product_id')['quantity'].sum().reset_index()
        merged    = sales.merge(products_df, on='product_id', how='left')

        if 'selling_price' in merged.columns and 'cost_price' in merged.columns:
            merged['profit_per_unit'] = merged['selling_price'] - merged['cost_price']
        else:
            merged['profit_per_unit'] = None
            merged['total_profit']    = None
            name_col    = 'product_name' if 'product_name' in merged.columns else 'name'
            return_cols = [c for c in [name_col, 'category', 'profit_per_unit', 'total_profit'] if c in merged.columns]
            result = merged[return_cols].copy()
            result['note'] = 'Cost data missing — profit unavailable'
            return result

        merged['total_profit'] = merged['profit_per_unit'] * merged['quantity']
        merged = merged.sort_values('total_profit', ascending=False)

        name_col    = 'product_name' if 'product_name' in merged.columns else 'name'
        return_cols = [c for c in [name_col, 'category', 'profit_per_unit', 'total_profit'] if c in merged.columns]
        return merged[return_cols]
    except Exception as e:
        return f'Error calculating profitability: {e}'


# ===== Conversion Rate =====
def get_conversion_rate(orders_df, start_date=None, end_date=None):
    try:
        df = orders_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        if 'visitors' in df.columns:
            daily          = df.drop_duplicates('order_date')
            total_visitors = daily['visitors'].sum()
            if total_visitors == 0:
                return 0
            total_completed = len(df[df['status'].isin(COMPLETED_STATUSES)])
            return round((total_completed / total_visitors) * 100, 4)

        total = len(df)
        if total == 0:
            return 0
        completed = len(df[df['status'].isin(COMPLETED_STATUSES)])
        return round((completed / total) * 100, 2)
    except Exception as e:
        return f'Error calculating conversion rate: {e}'


# ===== Period Comparison =====
def get_period_comparison(orders_df, products_df=None, start_date=None, end_date=None):
    try:
        df = _ensure_total_price(orders_df, products_df)
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        df['month']  = df['order_date'].dt.to_period('M')
        completed    = df[df['status'].isin(COMPLETED_STATUSES)]
        monthly      = completed.groupby('month')['total_price'].sum().reset_index()
        monthly['month'] = monthly['month'].astype(str)
        return monthly.sort_values('month')
    except Exception as e:
        return f'Error calculating period comparison: {e}'


# ===== Campaign Performance =====
# ROI = (campaign_revenue - budget) / budget * 100
# campaign_revenue is computed from orders within campaign start → end dates only
# No attribution window — revenue is attributed only to the campaign's active period
def get_campaign_performance(campaigns_df, orders_df=None, products_df=None):
    try:
        df = campaigns_df.copy()

        if 'campaign_name' not in df.columns and 'name' in df.columns:
            df = df.rename(columns={'name': 'campaign_name'})

        # Recompute campaign_revenue from start → end only (no attribution extension)
        if orders_df is not None and products_df is not None:
            enriched = _ensure_total_price(orders_df, products_df)
            enriched['order_date_dt'] = pd.to_datetime(enriched['order_date'])
            completed = enriched[enriched['status'].isin(COMPLETED_STATUSES)]

            revenues = []
            for _, camp in df.iterrows():
                start = pd.to_datetime(camp['start_date'])
                end   = pd.to_datetime(camp['end_date'])
                mask  = (completed['order_date_dt'] >= start) & (completed['order_date_dt'] <= end)
                revenues.append(round(completed[mask]['total_price'].sum(), 2))
            df['campaign_revenue'] = revenues

        # ROI from real campaign_revenue
        if 'campaign_revenue' in df.columns and 'budget' in df.columns:
            df['roi'] = ((df['campaign_revenue'] - df['budget']) / df['budget'] * 100).round(2)
        else:
            df['roi'] = None

        if 'clicks' in df.columns and 'budget' in df.columns:
            df['cost_per_click'] = (df['budget'] / df['clicks'].replace(0, 1)).round(2)
        else:
            df['cost_per_click'] = None

        keep = [c for c in ['campaign_id', 'campaign_name', 'platform', 'budget',
                             'clicks', 'conversions', 'campaign_revenue',
                             'roi', 'cost_per_click', 'start_date', 'end_date']
                if c in df.columns]
        return df[keep].sort_values('roi', ascending=False)
    except Exception as e:
        return f'Error calculating campaign performance: {e}'


# ===== Repeat Purchase Rate =====
# Repeat rate = customers who bought more than once / customers who bought at least once
# Not divided by total registered customers — measures loyalty among actual buyers
def get_repeat_purchase_rate(orders_df, customers_df, start_date=None, end_date=None):
    try:
        df = orders_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        purchase_counts = completed.groupby('customer_id')['order_id'].count().reset_index()
        purchase_counts.columns = ['customer_id', 'purchase_count']

        total_customers    = len(customers_df)
        active_customers   = len(purchase_counts)
        repeat_customers   = len(purchase_counts[purchase_counts['purchase_count'] > 1])
        one_time_customers = len(purchase_counts[purchase_counts['purchase_count'] == 1])
        never_purchased    = total_customers - active_customers

        # Repeat rate among customers who actually purchased (not all registered)
        repeat_rate = (repeat_customers / active_customers * 100) if active_customers > 0 else 0

        return {
            'repeat_rate':        round(repeat_rate, 2),
            'repeat_customers':   repeat_customers,
            'one_time_customers': one_time_customers,
            'never_purchased':    never_purchased,
            'active_customers':   active_customers,
            'total_customers':    total_customers
        }
    except Exception as e:
        return f'Error calculating repeat purchase rate: {e}'


# ===== Revenue by City =====
def get_revenue_by_city(orders_df, customers_df, products_df=None, start_date=None, end_date=None):
    """إجمالي الإيرادات حسب مدينة العميل"""
    try:
        df = _ensure_total_price(orders_df, products_df)
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        if 'city' not in customers_df.columns:
            return pd.DataFrame(columns=['city', 'total_price'])

        merged = completed.merge(customers_df[['customer_id', 'city']], on='customer_id', how='left')
        merged['city'] = merged['city'].fillna('Unknown')
        revenue = merged.groupby('city')['total_price'].sum().reset_index()
        return revenue.sort_values('total_price', ascending=False)
    except Exception as e:
        return f'Error calculating revenue by city: {e}'


# ===== Top Products by City =====
def get_top_products_by_city(orders_df, customers_df, products_df, start_date=None, end_date=None, top_n=3):
    """أكثر المنتجات مبيعاً لكل مدينة"""
    try:
        df = orders_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        if start_date:
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['order_date'] <= pd.to_datetime(end_date)]

        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        if 'city' not in customers_df.columns:
            return pd.DataFrame()

        merged = completed.merge(customers_df[['customer_id', 'city']], on='customer_id', how='left')
        name_col = 'product_name' if 'product_name' in products_df.columns else 'name'
        cols = [c for c in ['product_id', name_col, 'category'] if c in products_df.columns]
        merged = merged.merge(products_df[cols], on='product_id', how='left')
        if 'product_name' not in merged.columns and 'name' in merged.columns:
            merged = merged.rename(columns={'name': 'product_name'})

        merged['city'] = merged['city'].fillna('Unknown')
        grouped = merged.groupby(['city', 'product_name'])['quantity'].sum().reset_index()
        # Top N per city
        top_per_city = (
            grouped.sort_values(['city', 'quantity'], ascending=[True, False])
                   .groupby('city')
                   .head(top_n)
                   .reset_index(drop=True)
        )
        return top_per_city
    except Exception as e:
        return f'Error calculating top products by city: {e}'


# ===== Main =====
if __name__ == '__main__':
    try:
        products_df, customers_df, orders_df, campaigns_df = load_data()
    except RuntimeError as e:
        print(e)
        exit(1)

    print('===== Total Revenue =====')
    print(get_total_revenue(orders_df, products_df))

    print('\n===== Best Selling Products =====')
    print(get_best_products(orders_df, products_df))

    print('\n===== Cancellation Rate =====')
    print(get_cancellation_rate(orders_df), '%')

    print('\n===== Revenue by Category =====')
    print(get_revenue_by_category(orders_df, products_df))

    print('\n===== Most Profitable Products =====')
    print(get_most_profitable(products_df, orders_df))

    print('\n===== Conversion Rate =====')
    print(get_conversion_rate(orders_df), '%')

    print('\n===== Monthly Revenue =====')
    print(get_period_comparison(orders_df, products_df))

    print('\n===== Campaign Performance =====')
    print(get_campaign_performance(campaigns_df, orders_df, products_df))

    print('\n===== Repeat Purchase Rate =====')
    print(get_repeat_purchase_rate(orders_df, customers_df))