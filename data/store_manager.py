import sqlite3
import os
import json
from datetime import datetime

# ===== Config =====
STORES_DIR      = os.path.dirname(__file__)
STORES_REGISTRY = os.path.join(STORES_DIR, "stores_registry.json")


# ===== Load Stores Registry =====
def load_registry():
    try:
        if not os.path.exists(STORES_REGISTRY):
            default = {
                "stores": [
                    {
                        "id":         "demo",
                        "name":       "المتجر التجريبي",
                        "db_path":    "store.db",
                        "created_at": "2024-01-01",
                        "tables":     ["products", "customers", "orders", "campaigns", "carts"]
                    }
                ]
            }
            save_registry(default)
            return default
        with open(STORES_REGISTRY, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'Warning: could not load stores registry: {e}')
        return {"stores": []}


# ===== Save Stores Registry =====
def save_registry(registry):
    try:
        with open(STORES_REGISTRY, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'Warning: could not save stores registry: {e}')


# ===== Get All Stores =====
def get_all_stores():
    registry = load_registry()
    return registry.get("stores", [])


# ===== Get Store by ID =====
def get_store(store_id):
    registry = load_registry()
    for store in registry.get("stores", []):
        if store["id"] == store_id:
            return store
    return None


# ===== Create New Store =====
def create_store(store_name):
    try:
        registry  = load_registry()
        store_id  = store_name.replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        store_id  = f"store_{store_id}_{timestamp}"
        db_path   = f"store_{store_id}.db"

        new_store = {
            "id":         store_id,
            "name":       store_name,
            "db_path":    db_path,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tables":     []
        }

        registry["stores"].append(new_store)
        save_registry(registry)
        return new_store
    except Exception as e:
        print(f'Warning: could not create store: {e}')
        return None


# ===== Update Store Tables =====
def update_store_tables(store_id, table_name):
    try:
        registry = load_registry()
        for store in registry.get("stores", []):
            if store["id"] == store_id:
                if table_name not in store["tables"]:
                    store["tables"].append(table_name)
        save_registry(registry)
    except Exception as e:
        print(f'Warning: could not update store tables: {e}')


# ===== Delete Store =====
def delete_store(store_id):
    if store_id == "demo":
        print('Warning: demo store cannot be deleted.')
        return False

    try:
        registry = load_registry()
        store    = get_store(store_id)

        if store:
            db_path = os.path.join(STORES_DIR, store["db_path"])
            if os.path.exists(db_path):
                os.remove(db_path)

            registry["stores"] = [s for s in registry["stores"] if s["id"] != store_id]
            save_registry(registry)
            return True

        print(f'Warning: store {store_id} not found.')
        return False
    except Exception as e:
        print(f'Warning: could not delete store {store_id}: {e}')
        return False


# ===== Load Store Data =====
def load_store_data(store_id):
    import pandas as pd

    store = get_store(store_id)
    if not store:
        print(f'Warning: store {store_id} not found — returning empty data.')
        return None, None, None, None, None

    db_path = os.path.join(STORES_DIR, store["db_path"])
    if not os.path.exists(db_path):
        print(f'Warning: database for store {store_id} not found at {db_path}.')
        return None, None, None, None, None

    try:
        conn = sqlite3.connect(db_path)

        def safe_read(table):
            try:
                return pd.read_sql(f'SELECT * FROM {table}', conn)
            except Exception as e:
                print(f'Warning: could not load table {table} from store {store_id}: {e}')
                return None

        products_df  = safe_read("products")
        customers_df = safe_read("customers")
        orders_df    = safe_read("orders")
        campaigns_df = safe_read("campaigns")
        carts_df     = safe_read("carts")

        conn.close()
        return products_df, customers_df, orders_df, campaigns_df, carts_df

    except Exception as e:
        print(f'Warning: could not connect to database for store {store_id}: {e}')
        return None, None, None, None, None


# ===== Get Store DB Path =====
def get_store_db_path(store_id):
    try:
        store = get_store(store_id)
        if store:
            return os.path.join(STORES_DIR, store["db_path"])
        print(f'Warning: store {store_id} not found.')
        return None
    except Exception as e:
        print(f'Warning: could not get db path for store {store_id}: {e}')
        return None


# ===== Check Store Has Table =====
def store_has_table(store_id, table_name):
    try:
        store = get_store(store_id)
        if store:
            return table_name in store.get("tables", [])
        return False
    except Exception as e:
        print(f'Warning: could not check table for store {store_id}: {e}')
        return False