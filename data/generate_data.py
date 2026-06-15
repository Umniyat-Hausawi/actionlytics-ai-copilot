import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sqlite3
import os

random.seed(42)
np.random.seed(42)


def generate_products():
    product_list = [
        (1,  'Wireless Earbuds',            'Audio',        120, 350, random.randint(5, 100)),
        (2,  'Earbuds Carrying Case',        'Audio',         25,  75, random.randint(5, 100)),
        (3,  'Noise-Cancelling Headphones',  'Audio',        200, 580, random.randint(5, 100)),
        (4,  'Fast Charger 65W',             'Accessories',   45, 120, random.randint(5, 100)),
        (5,  'USB-C Cable 2m',               'Accessories',   15,  45, random.randint(5, 100)),
        (6,  'Smartphone Stand',             'Accessories',   20,  65, random.randint(5, 100)),
        (7,  'Smart Watch',                  'Wearables',    280, 750, random.randint(5, 100)),
        (8,  'Watch Replacement Band',       'Wearables',     30,  90, random.randint(5, 100)),
        (9,  'Wireless Charging Pad',        'Accessories',   55, 160, random.randint(5, 100)),
        (10, 'Portable Power Bank',          'Accessories',   90, 240, random.randint(5, 100)),
    ]
    products = []
    for product_id, name, category, cost, price, stock in product_list:
        products.append({
            'product_id':     product_id,
            'product_name':   name,
            'category':       category,
            'cost_price':     cost,
            'selling_price':  price,
            'stock_quantity': stock
        })
    return pd.DataFrame(products)


def generate_customers(n=400):
    cities = ['Riyadh', 'Jeddah', 'Dammam', 'Mecca', 'Medina']
    customers = []
    start_date = datetime(2023, 1, 1)
    for i in range(n):
        reg_date = start_date + timedelta(days=random.randint(0, 1185))
        customers.append({
            'customer_id':       i + 1,
            'customer_name':     f'Customer_{i + 1}',
            'city':              random.choice(cities),
            'registration_date': reg_date.strftime('%Y-%m-%d')
        })
    return pd.DataFrame(customers)


def generate_orders(customers_df, n=2000):
    TARGET_CVR = 0.025
    orders = []

    for i in range(n):
        customer = customers_df.sample(1).iloc[0]
        customer_reg_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')

        days_since_reg = (datetime(2026, 4, 30) - customer_reg_date).days
        if days_since_reg < 1:
            continue

        order_date = customer_reg_date + timedelta(days=random.randint(1, days_since_reg))
        order_time = f'{random.randint(8, 23):02d}:{random.randint(0, 59):02d}'

        orders.append({
            'order_id':    i + 1,
            'customer_id': int(customer['customer_id']),
            'product_id':  random.randint(1, 10),
            'quantity':    random.randint(1, 5),
            'order_date':  order_date.strftime('%Y-%m-%d'),
            'order_time':  order_time,
            'status':      random.choices(
                               ['completed', 'cancelled', 'refunded'],
                               weights=[85, 10, 5]
                           )[0]
        })

    orders_df = pd.DataFrame(orders)

    # Compute daily visitors from real completed orders
    orders_df['order_date_dt'] = pd.to_datetime(orders_df['order_date'])
    completed = orders_df[orders_df['status'] == 'completed']
    daily = completed.groupby('order_date_dt')['order_id'].count().reset_index()
    daily.columns = ['order_date_dt', 'daily_completed']
    daily['daily_visitors'] = (daily['daily_completed'] / TARGET_CVR).apply(
        lambda x: max(50, int(x * random.uniform(0.85, 1.15)))
    )
    orders_df = orders_df.merge(daily[['order_date_dt', 'daily_visitors']], on='order_date_dt', how='left')
    orders_df['visitors'] = orders_df['daily_visitors'].fillna(50).astype(int)
    orders_df = orders_df.drop(columns=['order_date_dt', 'daily_visitors'])

    return orders_df


def generate_campaigns():
    campaigns = [
        (1,  'Ramadan Sale 2023',    'Instagram', 5000,  12000, '2023-03-22', '2023-04-21'),
        (2,  'Summer Collection 23', 'Snapchat',  3000,   7500, '2023-06-01', '2023-06-30'),
        (3,  'Back to School 23',    'Google',    4000,   9000, '2023-08-15', '2023-09-15'),
        (4,  'White Friday 23',      'Instagram', 8000,  25000, '2023-11-24', '2023-11-30'),
        (5,  'New Year 2024',        'Snapchat',  3500,   8000, '2023-12-28', '2024-01-05'),
        (6,  'Ramadan Sale 2024',    'Instagram', 6000,  15000, '2024-03-11', '2024-04-09'),
        (7,  'Summer Sale 24',       'Google',    4500,  11000, '2024-06-01', '2024-06-30'),
        (8,  'White Friday 24',      'Instagram', 9000,  28000, '2024-11-29', '2024-12-05'),
        (9,  'New Year 2025',        'Snapchat',  4000,   9500, '2024-12-28', '2025-01-05'),
        (10, 'Ramadan Sale 2025',    'Instagram', 7000,  18000, '2025-03-01', '2025-03-30'),
        (11, 'Summer Sale 25',       'Google',    5000,  12500, '2025-06-01', '2025-06-30'),
        (12, 'White Friday 25',      'Instagram',10000,  32000, '2025-11-28', '2025-12-04'),
        (13, 'New Year 2026',        'Snapchat',  4500,  10500, '2025-12-28', '2026-01-05'),
    ]
    result = []
    for campaign_id, name, platform, budget, clicks, start, end in campaigns:
        result.append({
            'campaign_id':      campaign_id,
            'campaign_name':    name,
            'platform':         platform,
            'budget':           budget,
            'clicks':           clicks,
            'start_date':       start,
            'end_date':         end,
            'conversions':      int(clicks * random.uniform(0.02, 0.08))
        })
    return pd.DataFrame(result)


def generate_carts(customers_df, orders_df, n=400):
    completed_orders = orders_df[orders_df['status'] == 'completed']
    purchase_counts = completed_orders.groupby('customer_id')['order_id'].count().to_dict()

    carts = []
    for i in range(n):
        customer    = customers_df.sample(1).iloc[0]
        customer_id = int(customer['customer_id'])
        reg_date    = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
        days_avail  = (datetime(2026, 4, 30) - reg_date).days
        if days_avail < 1:
            continue
        cart_date = reg_date + timedelta(days=random.randint(1, days_avail))
        carts.append({
            'cart_id':      i + 1,
            'customer_id':  customer_id,
            'product_id':   random.randint(1, 10),
            'quantity':     random.randint(1, 3),
            'cart_date':    cart_date.strftime('%Y-%m-%d'),
            'status':       random.choices(['abandoned', 'completed'], weights=[40, 60])[0],
            'total_orders': purchase_counts.get(customer_id, 0)
        })
    return pd.DataFrame(carts)


def save_to_database(products_df, customers_df, orders_df, campaigns_df, carts_df):
    db_path = os.path.join(os.path.dirname(__file__), 'store.db')
    conn = sqlite3.connect(db_path)

    orders_save = orders_df.merge(
        products_df[['product_id', 'selling_price']], on='product_id', how='left'
    )
    orders_save['total_price'] = orders_save['quantity'] * orders_save['selling_price']
    orders_save = orders_save.drop(columns=['selling_price'])

    # Compute campaign_revenue from real orders
    orders_save['order_date_dt'] = pd.to_datetime(orders_save['order_date'])
    completed = orders_save[orders_save['status'] == 'completed']
    revenues = []
    for _, camp in campaigns_df.iterrows():
        start = pd.to_datetime(camp['start_date'])
        end   = pd.to_datetime(camp['end_date'])
        mask  = (completed['order_date_dt'] >= start) & (completed['order_date_dt'] <= end)
        revenues.append(round(completed[mask]['total_price'].sum(), 2))
    campaigns_df = campaigns_df.copy()
    campaigns_df['campaign_revenue'] = revenues
    orders_save = orders_save.drop(columns=['order_date_dt'])

    products_df.to_sql('products',   conn, if_exists='replace', index=False)
    customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    orders_save.to_sql('orders',     conn, if_exists='replace', index=False)
    campaigns_df.to_sql('campaigns', conn, if_exists='replace', index=False)
    carts_df.to_sql('carts',         conn, if_exists='replace', index=False)

    conn.close()
    print('Database saved successfully!')
    return campaigns_df


if __name__ == '__main__':
    print('Generating data...')
    products_df  = generate_products()
    customers_df = generate_customers()
    orders_df    = generate_orders(customers_df)
    campaigns_df = generate_campaigns()
    carts_df     = generate_carts(customers_df, orders_df)
    campaigns_df = save_to_database(products_df, customers_df, orders_df, campaigns_df, carts_df)

    print(f'Products  : {len(products_df)} rows')
    print(f'Customers : {len(customers_df)} rows')
    print(f'Orders    : {len(orders_df)} rows')
    print(f'Campaigns : {len(campaigns_df)} rows')
    print(f'Carts     : {len(carts_df)} rows')
    print(f'Date range: {orders_df["order_date"].min()} → {orders_df["order_date"].max()}')
    print(f'Statuses  : {orders_df["status"].value_counts().to_dict()}')
    print(campaigns_df[['campaign_name', 'budget', 'campaign_revenue']].to_string(index=False))