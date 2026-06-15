import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from analytics import load_data, _ensure_total_price, COMPLETED_STATUSES

MIN_MONTHS_FOR_FORECAST = 6

# Features used for forecast — seasonal awareness
FORECAST_FEATURES = ['month_num', 'month_of_year', 'quarter', 'is_q4']

# Model registry path
MODEL_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), 'model_registry.json')


# ===== Model Registry =====
def _load_registry():
    if os.path.exists(MODEL_REGISTRY_PATH):
        try:
            with open(MODEL_REGISTRY_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"forecast": [], "demand": []}

def _save_registry(registry):
    try:
        with open(MODEL_REGISTRY_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f'Warning: could not save model registry: {e}')

def _log_model_version(model_type, model_name, metrics, features):
    """Log a new model version to the registry."""
    registry = _load_registry()
    history  = registry.get(model_type, [])
    version  = f"v{len(history) + 1}"
    entry = {
        "version":    version,
        "model":      model_name,
        "trained_on": datetime.now().strftime("%Y-%m-%d"),
        "metrics":    metrics,
        "features":   features
    }
    history.append(entry)
    registry[model_type] = history
    _save_registry(registry)
    return version


# ===== Prepare Time Series Data =====
def prepare_time_series(orders_df, products_df=None):
    df = _ensure_total_price(orders_df, products_df)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month']      = df['order_date'].dt.to_period('M')

    completed        = df[df['status'].isin(COMPLETED_STATUSES)]
    monthly          = completed.groupby('month')['total_price'].sum().reset_index()
    monthly['month'] = monthly['month'].dt.to_timestamp()
    monthly          = monthly.sort_values('month')
    monthly['month_num'] = range(len(monthly))

    # Seasonal features
    monthly['month_of_year'] = monthly['month'].dt.month
    monthly['quarter']       = monthly['month'].dt.quarter
    monthly['is_q4']         = (monthly['quarter'] == 4).astype(int)

    return monthly


# ===== Minimum Data Check =====
def _check_min_data(monthly):
    if len(monthly) < MIN_MONTHS_FOR_FORECAST:
        return False, f'Insufficient data — need at least {MIN_MONTHS_FOR_FORECAST} months, found {len(monthly)}.'
    return True, None


# ===== Time-based Backtesting =====
def _run_backtesting(monthly, model, features=None):
    """Train on 80% of data, test on last 20%. Returns MAE, RMSE, MAPE."""
    if features is None:
        features = FORECAST_FEATURES

    split   = max(1, int(len(monthly) * 0.8))
    train   = monthly.iloc[:split]
    test    = monthly.iloc[split:]

    if len(test) == 0:
        return None

    X_train = train[features]
    y_train = train['total_price']
    X_test  = test[features]
    y_test  = test['total_price']

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = round(float(mean_absolute_error(y_test, y_pred)), 2)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
    mape = round(float(np.mean(np.abs((y_test - y_pred) / y_test.replace(0, 1))) * 100), 2)

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
            'train_months': len(train), 'test_months': len(test)}


# ===== Sales Forecast =====
def forecast_sales(orders_df, products_df=None, periods=4):
    try:
        monthly = prepare_time_series(orders_df, products_df)

        ok, msg = _check_min_data(monthly)
        if not ok:
            return pd.DataFrame(), 0, {'note': msg}

        comparison_df, best_model_name = compare_forecast_models(orders_df, products_df)

        models = {
            'LinearRegression': LinearRegression(),
            'Ridge':            Ridge(alpha=1.0),
            'Lasso':            Lasso(alpha=1.0),
            'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42)
        }

        model = models[best_model_name]

        # Backtesting on held-out data
        backtest_model = models[best_model_name]
        backtesting    = _run_backtesting(monthly.copy(), backtest_model, features=FORECAST_FEATURES)

        # Train final model on all data
        X = monthly[FORECAST_FEATURES]
        y = monthly['total_price']
        model.fit(X, y)

        # Reliability based on MAPE from backtesting
        if backtesting and backtesting['MAPE'] is not None:
            reliability = round(max(0, 100 - backtesting['MAPE']), 1)
        else:
            mae         = mean_absolute_error(y, model.predict(X))
            reliability = round(max(0, 100 - (mae / y.mean() * 100)), 1)

        # Generate future dates with seasonal features
        last_month    = monthly['month_num'].max()
        last_date     = monthly['month'].max()
        future_dates  = [last_date + pd.DateOffset(months=i + 1) for i in range(periods)]

        future_df = pd.DataFrame({
            'month_num':     range(last_month + 1, last_month + periods + 1),
            'month_of_year': [d.month    for d in future_dates],
            'quarter':       [d.quarter  for d in future_dates],
            'is_q4':         [1 if d.quarter == 4 else 0 for d in future_dates]
        })

        predictions = [max(0, p) for p in model.predict(future_df[FORECAST_FEATURES])]

        forecast_df = pd.DataFrame({
            'month':             [d.strftime('%Y-%m') for d in future_dates],
            'predicted_revenue': [round(p, 2) for p in predictions],
            'reliability':       reliability,
            'note':              'Estimate based on historical patterns, not guaranteed'
        })

        # Log model version
        _log_model_version(
            model_type='forecast',
            model_name=best_model_name,
            metrics={
                'MAPE': backtesting.get('MAPE') if backtesting else None,
                'MAE':  backtesting.get('MAE')  if backtesting else None,
                'RMSE': backtesting.get('RMSE') if backtesting else None,
            },
            features=FORECAST_FEATURES
        )

        return forecast_df, reliability, backtesting or {}

    except Exception as e:
        return pd.DataFrame(), 0, {'note': f'Forecast unavailable: {e}'}


# ===== Anomaly Detection =====
def detect_anomalies(orders_df, products_df=None):
    try:
        monthly = prepare_time_series(orders_df, products_df)

        X        = monthly[['total_price']]
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_scaled)

        monthly['anomaly']       = model.predict(X_scaled)
        monthly['anomaly_score'] = model.score_samples(X_scaled)

        anomalies    = monthly[monthly['anomaly'] == -1].copy()
        normal       = monthly[monthly['anomaly'] == 1].copy()
        mean_revenue = normal['total_price'].mean()

        anomalies['type'] = anomalies['total_price'].apply(
            lambda x: 'spike' if x > mean_revenue else 'drop'
        )
        anomalies['severity'] = anomalies['total_price'].apply(
            lambda x: 'High'   if abs(x - mean_revenue) / mean_revenue > 0.5
            else      'Medium' if abs(x - mean_revenue) / mean_revenue > 0.25
            else      'Low'
        )

        return anomalies[['month', 'total_price', 'type', 'severity']], monthly

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()


# ===== Model Evaluation (standalone) =====
def evaluate_forecast(orders_df, products_df=None):
    try:
        monthly = prepare_time_series(orders_df, products_df)
        ok, msg = _check_min_data(monthly)
        if not ok:
            return {'note': msg}

        model       = LinearRegression()
        backtesting = _run_backtesting(monthly.copy(), model, features=FORECAST_FEATURES)
        return backtesting if backtesting else {'note': 'Not enough data for evaluation'}

    except Exception as e:
        return {'note': f'Evaluation unavailable: {e}'}


# ===== Product Demand Forecasting =====
def forecast_product_demand(orders_df, products_df, periods=3):
    try:
        df        = _ensure_total_price(orders_df, products_df)
        completed = df[df['status'].isin(COMPLETED_STATUSES)]
        name_col  = 'product_name' if 'product_name' in products_df.columns else 'name'

        results = []
        for product_id in products_df['product_id'].unique():
            product_orders = completed[completed['product_id'] == product_id].copy()
            name_vals      = products_df[products_df['product_id'] == product_id][name_col].values
            if len(name_vals) == 0:
                continue
            product_name = name_vals[0]

            product_orders['order_date'] = pd.to_datetime(product_orders['order_date'])
            product_orders['month']      = product_orders['order_date'].dt.to_period('M')
            monthly                      = product_orders.groupby('month')['quantity'].sum().reset_index()
            monthly['month']             = monthly['month'].dt.to_timestamp()
            monthly                      = monthly.sort_values('month')
            monthly['month_num']         = range(len(monthly))

            if len(monthly) < 3:
                continue

            # Time-based backtesting for demand — only if enough data
            X = monthly[['month_num']]
            y = monthly['quantity']

            model = LinearRegression()

            if len(monthly) >= 4:
                # Time-based split: train 80%, test 20%
                split   = max(1, int(len(monthly) * 0.8))
                train   = monthly.iloc[:split]
                test    = monthly.iloc[split:]
                X_train = train[['month_num']]
                y_train = train['quantity']
                X_test  = test[['month_num']]
                y_test  = test['quantity']

                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                mae_test    = mean_absolute_error(y_test, y_pred_test)
                reliability = round(max(0, 100 - (mae_test / max(y_test.mean(), 1) * 100)), 1)
                reliability_note = 'Estimated via time-based backtesting (80/20 split)'
            else:
                # Fallback for limited data
                model.fit(X, y)
                mae         = mean_absolute_error(y, model.predict(X))
                reliability = round(max(0, 100 - (mae / max(y.mean(), 1) * 100)), 1)
                reliability_note = 'Estimated — limited historical data per product (< 4 months)'

            # Train final model on all data for predictions
            model.fit(X, y)
            future         = pd.DataFrame({'month_num': range(monthly['month_num'].max() + 1,
                                                               monthly['month_num'].max() + periods + 1)})
            predictions    = model.predict(future)
            avg_prediction = max(0, round(predictions.mean(), 0))

            results.append({
                'product_name':       product_name,
                'avg_monthly_demand': round(y.mean(), 1),
                'predicted_demand':   avg_prediction,
                'trend':              'Growing' if predictions[-1] > predictions[0] else 'Declining',
                'reliability':        reliability,
                'reliability_note':   reliability_note
            })

        # Log demand model version
        if results:
            _log_model_version(
                model_type='demand',
                model_name='LinearRegression',
                metrics={'avg_reliability': round(sum(r['reliability'] for r in results) / len(results), 1)},
                features=['month_num']
            )

        results_df = pd.DataFrame(results)
        if results_df.empty:
            return results_df
        return results_df.sort_values('predicted_demand', ascending=False)

    except Exception as e:
        return pd.DataFrame()


# ===== Customer Segmentation =====
def segment_customers(orders_df, customers_df):
    try:
        df        = _ensure_total_price(orders_df)
        completed = df[df['status'].isin(COMPLETED_STATUSES)]

        completed2 = completed.copy()
        completed2['order_date'] = pd.to_datetime(completed2['order_date'])
        reference_date = completed2['order_date'].max()

        customer_stats = completed2.groupby('customer_id').agg(
            total_spent=('total_price', 'sum'),
            order_count=('order_id', 'count'),
            avg_order_value=('total_price', 'mean'),
            last_order_date=('order_date', 'max')
        ).reset_index()

        customer_stats['days_since_last'] = (reference_date - customer_stats['last_order_date']).dt.days
        customer_stats = customer_stats.drop(columns=['last_order_date'])

        name_col   = 'customer_name' if 'customer_name' in customers_df.columns else 'name'
        merge_cols = [c for c in ['customer_id', name_col, 'city'] if c in customers_df.columns]
        customer_stats = customer_stats.merge(customers_df[merge_cols], on='customer_id', how='left')

        if 'customer_name' not in customer_stats.columns and 'name' in customer_stats.columns:
            customer_stats = customer_stats.rename(columns={'name': 'customer_name'})

        scaler   = StandardScaler()
        features = ['total_spent', 'order_count', 'avg_order_value', 'days_since_last']
        X_scaled = scaler.fit_transform(customer_stats[features])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        customer_stats['segment'] = kmeans.fit_predict(X_scaled)

        segment_means   = customer_stats.groupby('segment')['total_spent'].mean()
        sorted_segments = segment_means.sort_values(ascending=False).index.tolist()

        segment_labels = {}
        for i, seg in enumerate(sorted_segments):
            segment_labels[seg] = ['VIP', 'Regular', 'Dormant'][i]

        customer_stats['segment_label'] = customer_stats['segment'].map(segment_labels)

        summary = customer_stats.groupby('segment_label').agg(
            count=('customer_id', 'count'),
            avg_spent=('total_spent', 'mean'),
            avg_orders=('order_count', 'mean')
        ).reset_index()

        summary['avg_spent']  = summary['avg_spent'].round(2)
        summary['avg_orders'] = summary['avg_orders'].round(1)

        return customer_stats, summary

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()


# ===== Behavior-Based Churn Detection =====
def predict_churn(orders_df, customers_df):
    try:
        df        = _ensure_total_price(orders_df)
        completed = df[df['status'].isin(COMPLETED_STATUSES)]

        customer_orders = completed.groupby('customer_id').agg(
            order_count=('order_id', 'count'),
            last_order_date=('order_date', 'max'),
            total_spent=('total_price', 'sum')
        ).reset_index()

        completed2 = completed.copy()
        completed2['order_date'] = pd.to_datetime(completed2['order_date'])
        avg_days = completed2.sort_values('order_date').groupby('customer_id')['order_date'].apply(
            lambda x: x.diff().dt.days.mean()
        ).reset_index()
        avg_days.columns = ['customer_id', 'avg_days_between']
        customer_orders = customer_orders.merge(avg_days, on='customer_id', how='left')

        repeat_customers = customer_orders[customer_orders['order_count'] > 1].copy()

        if len(repeat_customers) == 0:
            return pd.DataFrame(), {
                'total_repeat_customers': 0,
                'churn_risk_count':       0,
                'high_risk':              0,
                'medium_risk':            0,
                'churn_rate':             0
            }

        repeat_customers['last_order_date'] = pd.to_datetime(repeat_customers['last_order_date'])
        reference_date = pd.to_datetime(df['order_date']).max()
        repeat_customers['days_since_last']  = (reference_date - repeat_customers['last_order_date']).dt.days
        repeat_customers['avg_days_between'] = repeat_customers['avg_days_between'].fillna(30)
        repeat_customers['churn_threshold']  = repeat_customers['avg_days_between'] * 2
        repeat_customers['is_churn_risk']    = repeat_customers['days_since_last'] > repeat_customers['churn_threshold']

        repeat_customers['risk_level'] = repeat_customers.apply(
            lambda row: 'High'   if row['days_since_last'] > row['churn_threshold'] * 2
            else        'Medium' if row['is_churn_risk']
            else        'Low', axis=1
        )

        name_col   = 'customer_name' if 'customer_name' in customers_df.columns else 'name'
        merge_cols = [c for c in ['customer_id', name_col, 'city'] if c in customers_df.columns]
        repeat_customers = repeat_customers.merge(customers_df[merge_cols], on='customer_id', how='left')

        if 'customer_name' not in repeat_customers.columns and 'name' in repeat_customers.columns:
            repeat_customers = repeat_customers.rename(columns={'name': 'customer_name'})

        churn_risk = repeat_customers[repeat_customers['is_churn_risk']].copy()
        churn_risk = churn_risk.sort_values('days_since_last', ascending=False)

        summary = {
            'total_repeat_customers': len(repeat_customers),
            'churn_risk_count':       len(churn_risk),
            'high_risk':              len(churn_risk[churn_risk['risk_level'] == 'High']),
            'medium_risk':            len(churn_risk[churn_risk['risk_level'] == 'Medium']),
            'churn_rate':             round(len(churn_risk) / len(repeat_customers) * 100, 2)
        }

        return_cols = [c for c in ['customer_id', 'customer_name', 'city', 'days_since_last',
                                    'avg_days_between', 'risk_level', 'total_spent', 'order_count']
                       if c in churn_risk.columns]

        return churn_risk[return_cols], summary

    except Exception as e:
        return pd.DataFrame(), {
            'total_repeat_customers': 0, 'churn_risk_count': 0,
            'high_risk': 0, 'medium_risk': 0, 'churn_rate': 0,
            'note': f'Churn detection unavailable: {e}'
        }


# ===== Market Basket Analysis =====
def market_basket_analysis(orders_df, products_df, min_support=0.05, min_confidence=0.3):
    try:
        completed = orders_df[orders_df['status'].isin(COMPLETED_STATUSES)]

        orders_per_product   = completed.groupby('order_id')['product_id'].count()
        multi_product_orders = (orders_per_product > 1).sum()
        total_orders         = len(orders_per_product)

        if total_orders == 0:
            return pd.DataFrame()

        multi_ratio = multi_product_orders / total_orders
        if multi_ratio < 0.05:
            return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift',
                                         'note']).assign(
                note=[f'Insufficient multi-product orders ({multi_product_orders}/{total_orders}) — basket analysis requires more co-purchases']
            )

        basket = completed.groupby(['order_id', 'product_id'])['quantity'].sum().unstack().fillna(0)
        basket = basket.map(lambda x: True if x > 0 else False)

        name_col      = 'product_name' if 'product_name' in products_df.columns else 'name'
        product_names = products_df.set_index('product_id')[name_col].to_dict()
        basket.columns = [product_names.get(col, col) for col in basket.columns]

        frequent_items = apriori(basket, min_support=min_support, use_colnames=True)

        if len(frequent_items) == 0:
            return pd.DataFrame()

        rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
        rules = rules.sort_values('lift', ascending=False)

        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules['support']     = rules['support'].astype(float).round(3)
        rules['confidence']  = rules['confidence'].astype(float).round(3)
        rules['lift']        = rules['lift'].astype(float).round(3)

        return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    except Exception as e:
        return pd.DataFrame()


# ===== Model Comparison =====
def compare_forecast_models(orders_df, products_df=None):
    monthly = prepare_time_series(orders_df, products_df)

    X = monthly[FORECAST_FEATURES]
    y = monthly['total_price']

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge':            Ridge(alpha=1.0),
        'Lasso':            Lasso(alpha=1.0),
        'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=min(3, len(monthly)),
                                 scoring='neg_mean_absolute_error')
        mae    = abs(scores.mean())
        std    = scores.std()
        model.fit(X, y)
        y_pred = model.predict(X)
        mape   = abs((y - y_pred) / y.replace(0, 1)).mean() * 100

        results.append({
            'model':   name,
            'MAE':     round(mae, 2),
            'MAE_std': round(std, 2),
            'MAPE':    round(mape, 2)
        })

    results_df = pd.DataFrame(results).sort_values('MAE')
    best_model = results_df.iloc[0]['model']

    return results_df, best_model


# ===== Main =====
if __name__ == '__main__':
    try:
        products_df, customers_df, orders_df, campaigns_df = load_data()

        print('===== Sales Forecast =====')
        forecast_df, reliability, backtesting = forecast_sales(orders_df, products_df)
        print(f'Forecast Reliability : {reliability}%')
        if backtesting and 'MAE' in backtesting:
            print(f'Backtesting MAE      : {backtesting["MAE"]} SAR')
            print(f'Backtesting MAPE     : {backtesting["MAPE"]}%')
            print(f'Train months         : {backtesting["train_months"]} | Test months: {backtesting["test_months"]}')
        print(forecast_df)

        print('\n===== Anomaly Detection =====')
        anomalies, monthly = detect_anomalies(orders_df, products_df)
        print(f'Anomalies: {len(anomalies)}')
        print(anomalies)

        print('\n===== Product Demand Forecast =====')
        demand_df = forecast_product_demand(orders_df, products_df)
        print(demand_df[['product_name', 'predicted_demand', 'trend', 'reliability', 'reliability_note']])

        print('\n===== Customer Segmentation =====')
        customer_stats, summary = segment_customers(orders_df, customers_df)
        print(summary)

        print('\n===== Behavior-Based Churn Detection =====')
        churn_df, churn_summary = predict_churn(orders_df, customers_df)
        print(churn_summary)

        print('\n===== Market Basket Analysis =====')
        basket = market_basket_analysis(orders_df, products_df, min_support=0.01, min_confidence=0.1)
        print(basket)

        print('\n===== Model Registry =====')
        registry = _load_registry()
        for model_type, versions in registry.items():
            print(f'\n{model_type}:')
            for v in versions:
                print(f"  {v['version']} | {v['model']} | {v['trained_on']} | metrics: {v['metrics']}")

    except Exception as e:
        print(f'ML Engine failed: {e}')