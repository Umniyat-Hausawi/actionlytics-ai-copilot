import os
import pandas as pd
import numpy as np
import sqlite3
import anthropic
import json
from dotenv import load_dotenv

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

DB_PATH = "uploaded_store.db"

# Our expected columns per table
EXPECTED_COLUMNS = {
    "orders": ["order_id", "customer_id", "product_id", "quantity", "order_date", "order_time", "visitors", "status", "total_price"],
    "customers": ["customer_id", "customer_name", "city", "gender", "registration_date"],  # أضفنا gender
    "products": ["product_id", "product_name", "category", "cost_price", "selling_price", "stock_quantity"],
    "campaigns": ["campaign_id", "campaign_name", "platform", "budget", "clicks", "start_date", "end_date", "conversions", "campaign_revenue", "roi"],  # أضفنا campaign_revenue
    "carts": ["cart_id", "customer_id", "product_id", "quantity", "cart_date", "status", "total_orders"]
}

# ===== Read Uploaded File =====
def read_uploaded_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    return df

# ===== Map Columns with Claude =====
def map_columns_with_claude(df_columns, table_name, correction_note=None):
    expected = EXPECTED_COLUMNS.get(table_name, [])

    correction_text = ""
    if correction_note:
        correction_text = f"\nUser correction note: {correction_note}\nPlease fix the mapping based on this note."

    prompt = f"""
You are a data mapping assistant.
The user uploaded a file with these columns: {list(df_columns)}
We need to map them to our expected columns for the "{table_name}" table: {expected}

Rules:
1. Match each uploaded column to the closest expected column based on meaning and semantics
2. Arabic column names must be matched correctly:
   - اسم العميل or اسم = customer_name (NOT customer_id)
   - رقم العميل or معرف العميل = customer_id
   - المدينة or مدينة = city
   - تاريخ الطلب = order_date
   - الكمية = quantity
   - السعر or المبلغ or الإجمالي = total_price
   - الحالة = status
3. Do NOT ignore location or city columns
4. If no match found, set value to null
5. Return ONLY a JSON object, no explanation, no markdown, no backticks
{correction_text}

Example output:
{{"اسم العميل": "customer_name", "تاريخ الطلب": "order_date", "المبلغ": "total_price", "المدينة": "city"}}
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku: كافي للـ mapping وأرخص بـ 95%
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    # Clean response — remove markdown backticks if present
    raw = raw.replace('```json', '').replace('```', '').strip()
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f'Warning: Claude returned invalid JSON for column mapping: {e}. Using empty mapping.')
        return {}
    if not isinstance(mapping, dict):
        print(f'Warning: Claude mapping is not a dict, got {type(mapping)}. Using empty mapping.')
        return {}
    mapping = {k: v for k, v in mapping.items() if v is not None}
    return mapping

# ===== Detect Table Type =====
def detect_table_type(df_columns):
    all_expected = {table: cols for table, cols in EXPECTED_COLUMNS.items()}

    prompt = f"""
You are a data classification assistant.
The user uploaded a file with these columns: {list(df_columns)}

We have these table types with their expected columns:
{all_expected}

Rules:
1. Analyze the column names and decide which table type best matches
2. Choose ONE table type from: orders, customers, products, campaigns, carts
3. Return ONLY the table name as a single word, no explanation, no markdown

Example output:
orders
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku: كافي للـ mapping وأرخص بـ 95%
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )

    table_name = response.content[0].text.strip().lower()
    if table_name not in EXPECTED_COLUMNS:
        table_name = "orders"
    return table_name

# ===== Clean Uploaded Data =====
def clean_uploaded_data(df, products_df=None):
    df = df.drop_duplicates()

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.strftime('%Y-%m-%d')

    for col in df.columns:
        if col in ['quantity', 'total_price', 'cost_price', 'selling_price',
                   'stock_quantity', 'budget', 'clicks', 'conversions', 'roi', 'price']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['total_price', 'cost_price', 'selling_price', 'budget']:
        if col in df.columns:
            df = df[df[col] > 0]

    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        median_qty = df['quantity'].median(skipna=True)
        df['quantity'] = df['quantity'].fillna(median_qty)
        df = df[df['quantity'] > 0]

    if 'city' in df.columns:
        df['city'] = df['city'].fillna('Unknown')
        df['city'] = df['city'].str.title()

    if 'customer_name' in df.columns:
        df['customer_name'] = df['customer_name'].fillna('Unknown')

    # Calculate total_price if missing
    if 'total_price' not in df.columns and 'quantity' in df.columns:
        # Try price column directly in df (e.g. products table joined or price col exists)
        if 'price' in df.columns:
            df['total_price'] = df['quantity'] * df['price']
        elif 'selling_price' in df.columns:
            df['total_price'] = df['quantity'] * df['selling_price']
        elif products_df is not None and 'product_id' in df.columns and 'price' in products_df.columns:
            price_map = products_df.set_index('product_id')['price']
            df['total_price'] = df['quantity'] * df['product_id'].map(price_map).fillna(0)
        elif products_df is not None and 'product_id' in df.columns and 'selling_price' in products_df.columns:
            price_map = products_df.set_index('product_id')['selling_price']
            df['total_price'] = df['quantity'] * df['product_id'].map(price_map).fillna(0)

    return df

# ===== Data Quality Report =====
def get_data_quality_report(df_raw, df_clean, mapping, table_name):
    """Returns a dict summarizing what happened during cleaning."""
    mapped_cols     = list(mapping.values())
    all_raw_cols    = list(df_raw.columns)
    ignored_cols    = [c for c in all_raw_cols if c not in mapping]

    return {
        'table_name':        table_name,
        'rows_before':       len(df_raw),
        'rows_after':        len(df_clean),
        'rows_removed':      len(df_raw) - len(df_clean),
        'columns_mapped':    mapped_cols,
        'columns_ignored':   ignored_cols,
        'duplicates_removed': len(df_raw) - len(df_raw.drop_duplicates()),
    }


# ===== Validate Required Columns =====
REQUIRED_COLUMNS = {
    'orders':    ['customer_id', 'product_id', 'quantity', 'order_date', 'status'],
    'customers': ['customer_id', 'customer_name'],
    'products':  ['product_id', 'product_name', 'selling_price'],
    'campaigns': ['campaign_id', 'campaign_name', 'budget'],
    'carts':     ['cart_id', 'customer_id', 'product_id'],
}

def validate_before_save(df, table_name):
    """Returns (is_valid, missing_columns)."""
    required = REQUIRED_COLUMNS.get(table_name, [])
    missing  = [c for c in required if c not in df.columns]
    return (len(missing) == 0), missing


# ===== Apply Column Mapping =====
def apply_mapping(df, mapping):
    ignored = [c for c in df.columns if c not in mapping]
    if ignored:
        print(f'Warning: columns not mapped and will be ignored: {ignored}')
    df = df.rename(columns=mapping)
    valid_cols = [col for col in df.columns if col in mapping.values()]
    df = df[valid_cols]
    return df

# ===== Save to Store DB =====
def save_uploaded_data(df, table_name, db_path=None, mode='replace'):
    """
    mode: 'replace' — overwrites existing table (default for new stores)
          'append'  — adds rows to existing table
    """
    is_valid, missing = validate_before_save(df, table_name)
    if not is_valid:
        raise ValueError(f'Cannot save {table_name}: missing required columns {missing}')

    target_db = db_path if db_path else DB_PATH
    conn = sqlite3.connect(target_db)
    df.to_sql(table_name, conn, if_exists=mode, index=False)
    conn.close()

# ===== Load from Store DB =====
def load_uploaded_table(table_name, db_path=None):
    target_db = db_path if db_path else DB_PATH
    try:
        conn = sqlite3.connect(target_db)
        df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        conn.close()
        return df
    except Exception as e:
        print(f'Warning: could not load table {table_name}: {e}')
        return None

# ===== Process Single File =====
def process_uploaded_file(file_path, db_path=None, products_df=None, mode='replace', correction_note=None):
    df = read_uploaded_file(file_path)
    df_raw = df.copy()
    table_name = detect_table_type(df.columns)
    # تمرير correction_note لـ Claude عند إعادة الـ mapping
    mapping = map_columns_with_claude(df.columns, table_name, correction_note=correction_note)
    df = apply_mapping(df, mapping)
    df = clean_uploaded_data(df, products_df=products_df)
    quality_report = get_data_quality_report(df_raw, df, mapping, table_name)
    save_uploaded_data(df, table_name, db_path=db_path, mode=mode)
    return df, df_raw, mapping, table_name, quality_report

# ===== Process Multiple Files =====
def process_multiple_files(file_paths, db_path=None, mode='replace'):
    results = []
    products_df = None

    # First pass: find products file to use for price mapping
    for file_path in file_paths:
        try:
            df_tmp   = read_uploaded_file(file_path)
            detected = detect_table_type(df_tmp.columns)
            if detected == 'products':
                mapping_tmp = map_columns_with_claude(df_tmp.columns, 'products')
                df_tmp      = apply_mapping(df_tmp, mapping_tmp)
                df_tmp      = clean_uploaded_data(df_tmp)
                products_df = df_tmp
                break
        except Exception as e:
            print(f'Warning: could not pre-process products file {file_path}: {e}')

    # Second pass: process all files
    for file_path in file_paths:
        try:
            df         = read_uploaded_file(file_path)
            df_raw     = df.copy()
            table_name = detect_table_type(df.columns)
            mapping    = map_columns_with_claude(df.columns, table_name)
            df         = apply_mapping(df, mapping)
            df         = clean_uploaded_data(df, products_df=products_df if table_name == 'orders' else None)
            quality_report = get_data_quality_report(df_raw, df, mapping, table_name)
            save_uploaded_data(df, table_name, db_path=db_path, mode=mode)
            results.append({
                'file':           os.path.basename(file_path),
                'table_name':     table_name,
                'df':             df,
                'df_raw':         df_raw,
                'mapping':        mapping,
                'rows':           len(df),
                'quality_report': quality_report,
                'success':        True
            })
        except Exception as e:
            print(f'Warning: failed to process {file_path}: {e}')
            results.append({
                'file':           os.path.basename(file_path),
                'table_name':     None,
                'df':             None,
                'df_raw':         None,
                'mapping':        None,
                'rows':           0,
                'quality_report': None,
                'success':        False,
                'error':          str(e)
            })
    return results

# ===== Load All Tables from Store DB =====
def load_all_store_tables(db_path=None):
    target_db = db_path if db_path else DB_PATH
    tables = {}
    try:
        conn = sqlite3.connect(target_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]
        for table in table_names:
            try:
                tables[table] = pd.read_sql(f'SELECT * FROM {table}', conn)
            except Exception as e:
                print(f'Warning: could not load table {table}: {e}')
        conn.close()
    except Exception as e:
        print(f'Warning: could not connect to database {target_db}: {e}')
    return tables