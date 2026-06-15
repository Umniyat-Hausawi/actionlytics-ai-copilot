import pandas as pd
import numpy as np
import sqlite3
import os

# ===== Load Data =====
def load_data():
    db_path = os.path.join(os.path.dirname(__file__), 'store.db')
    conn = sqlite3.connect(db_path)

    products_df = pd.read_sql('SELECT * FROM products', conn)
    customers_df = pd.read_sql('SELECT * FROM customers', conn)
    orders_df = pd.read_sql('SELECT * FROM orders', conn)
    campaigns_df = pd.read_sql('SELECT * FROM campaigns', conn)

    conn.close()
    return products_df, customers_df, orders_df, campaigns_df

# ===== Add Dirty Data =====
def add_dirty_data(customers_df, orders_df):

    # Missing values
    customers_df.loc[5:10, 'city'] = np.nan
    orders_df.loc[20:25, 'quantity'] = np.nan

    # Duplicates
    duplicate_orders = orders_df.iloc[0:5].copy()
    orders_df = pd.concat([orders_df, duplicate_orders], ignore_index=True)

    # Wrong format - city
    customers_df.loc[15:20, 'city'] = 'riyadh'
    customers_df.loc[21:25, 'city'] = 'RIYADH'

    # Wrong format - date
    orders_df.loc[40:45, 'order_date'] = '2023/06/15'

    # Negative values
    orders_df.loc[30:35, 'total_price'] = -999

    # Unrealistic values
    orders_df.loc[50:55, 'total_price'] = 1000000

    # Wrong type - letters in quantity
    orders_df['quantity'] = orders_df['quantity'].astype(str)
    orders_df.loc[60:65, 'quantity'] = 'unknown'

    # Zero quantity
    orders_df.loc[70:75, 'quantity'] = '0'

    # Extra spaces in names
    customers_df.loc[30:35, 'customer_name'] = '  Ahmed  '

    return customers_df, orders_df

# ===== Clean Data =====
def clean_data(customers_df, orders_df):

    # Fix missing values
    customers_df['city'] = customers_df['city'].fillna('Unknown')
    orders_df['quantity'] = pd.to_numeric(orders_df['quantity'], errors='coerce')
    median_qty = orders_df['quantity'].median(skipna=True)
    orders_df['quantity'] = orders_df['quantity'].fillna(median_qty)

    # Remove duplicates
    orders_df = orders_df.drop_duplicates()

    # Fix wrong format - city
    customers_df['city'] = customers_df['city'].str.strip().str.title()

    # Fix wrong format - date
    orders_df['order_date'] = orders_df['order_date'].str.replace('/', '-')

    # Remove negative values
    orders_df = orders_df[orders_df['total_price'] > 0]

    # Remove unrealistic values
    orders_df = orders_df[orders_df['total_price'] < 10000]

    # Fix wrong type - convert quantity to numeric
    orders_df['quantity'] = pd.to_numeric(orders_df['quantity'], errors='coerce')
    median_qty = orders_df['quantity'].median(skipna=True)
    orders_df['quantity'] = orders_df['quantity'].fillna(median_qty)

    # Remove zero quantity
    orders_df = orders_df[orders_df['quantity'] > 0]

    # Fix extra spaces in names
    customers_df['customer_name'] = customers_df['customer_name'].str.strip()

    return customers_df, orders_df

# ===== Save Clean Data =====
def save_clean_data(customers_df, orders_df):
    db_path = os.path.join(os.path.dirname(__file__), 'store.db')
    conn = sqlite3.connect(db_path)

    customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    orders_df.to_sql('orders', conn, if_exists='replace', index=False)

    conn.close()
    print('Clean data saved successfully!')


# ===== Main =====
if __name__ == '__main__':
    print('Loading data...')
    products_df, customers_df, orders_df, campaigns_df = load_data()

    print('Adding dirty data...')
    customers_df, orders_df = add_dirty_data(customers_df, orders_df)

    customers_before = len(customers_df)
    orders_before = len(orders_df)

    print('Cleaning data...')
    customers_df, orders_df = clean_data(customers_df, orders_df)

    print(f'Customers before: {customers_before} — after: {len(customers_df)}')
    print(f'Orders before: {orders_before} — after: {len(orders_df)}')

    save_clean_data(customers_df, orders_df)