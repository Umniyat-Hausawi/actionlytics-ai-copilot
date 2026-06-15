# Actionlytics — System Architecture

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │  KPI     │ │  Smart   │ │ Report   │ │  Upload  │   │
│  │Dashboard │ │ Actions  │ │ Engine   │ │ Manager  │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼───────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌──────────────────────────────────────────────────────────┐
│                    Core Layer                             │
│                                                          │
│  analytics.py    action_engine.py   report_engine.py     │
│  ml_engine.py    chatbot.py         file_analyzer.py     │
│  rag_engine.py   store_manager.py                        │
└───────────────────────────┬──────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
    ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
    │ Anthropic   │ │   SQLite    │ │    FAISS     │
    │ Claude API  │ │  Databases  │ │  Vectorstore │
    └─────────────┘ └─────────────┘ └──────────────┘
```

---

## 📦 Layer Breakdown

### 1. UI Layer — `app.py`
Streamlit-based single-page application with multiple functional modules:

```
app.py
├── Store Selector          → Switch between saved stores
├── KPI Dashboard           → Date-filtered metrics
├── Smart Actions           → 5 tabs (Churn/Restock/VIP/Cart/Campaign)
├── Report Generator        → Period-based report with charts
├── File Upload             → Multi-file upload with mapping review
└── AI Chatbot              → Natural language Q&A with auto-charts
```

### 2. Analytics Layer — `analytics.py`
Pure Python/Pandas calculations — no AI involved:

```
analytics.py
├── get_total_revenue()
├── get_cancellation_rate()
├── get_conversion_rate()
├── get_best_products()
├── get_revenue_by_category()
├── get_revenue_by_city()
├── get_period_comparison()
├── get_repeat_purchase_rate()
└── get_campaign_performance()
```

### 3. ML Layer — `ml_engine.py`
Scikit-learn based models, all trained on store data at runtime:

```
ml_engine.py
├── forecast_sales()              → Compares Linear Regression, Ridge, Lasso, and Random Forest — selects best model by evaluation metrics
├── forecast_product_demand()     → Per-product trend analysis
├── segment_customers()           → KMeans (RFM features)
├── predict_churn()               → Rule-based risk scoring
├── detect_anomalies()            → Isolation Forest
└── market_basket_analysis()      → Apriori (mlxtend)
```

### 4. AI Layer

#### Chatbot — `chatbot.py`
```
chatbot.py
├── get_analytics_summary()    → Builds full store context string
├── _needs_rag()               → Keyword-based RAG trigger
├── _needs_action_engine()     → Redirect to Smart Actions
├── _detect_language()         → Arabic/English detection
└── ask_actionlytics()         → Main Claude Sonnet API call
    ├── Analytics context
    ├── RAG context (conditional)
    ├── Conversation history (last 10 messages)
    └── System prompt with grounding, accuracy, and routing rules
```

#### Action Engine — `action_engine.py`
```
action_engine.py
├── run_churn_reminders()      → Lazy: list only, no API
├── run_vip_offers()           → Lazy: list only, no API
├── run_abandoned_cart_action()→ Lazy: list only, no API
├── run_restock_alerts()       → Eager: generates alert message
├── run_campaign_action()      → Eager: generates analysis
├── run_best_send_time()       → Eager: analyzes best hour per segment
└── generate_message_on_demand()→ On-demand Claude Haiku call
```

**Lazy Loading Pattern:**
```
Open Tab → Show customer list (no API) → Click "Generate Message" → Claude Haiku call
```

#### Report Engine — `report_engine.py`
```
report_engine.py
├── calculate_period_metrics() → Analytics for selected period
├── calculate_ml_insights()    → ML analysis for period
├── get_report_limitations()   → Auto-detected data gaps
└── generate_report()          → 2x Claude Opus 4.5 API calls
    ├── Part 1: Revenue, Products, Customers, Campaigns
    └── Part 2: Anomalies, Churn, Basket, Recommendations
```

#### File Analyzer — `file_analyzer.py`
```
file_analyzer.py
├── detect_table_type()        → Claude Haiku: orders/customers/products/campaigns/carts
├── map_columns_with_claude()  → Claude Haiku: auto column mapping
├── apply_mapping()            → Rename columns
├── clean_uploaded_data()      → Dedup, type casting, fill nulls
└── save_uploaded_data()       → SQLite persistence
```

### 5. RAG Layer — `rag_engine.py`
```
rag_engine.py
├── build_rag()                → FAISS index from knowledge_base.md
└── query_rag()                → Similarity search → top-k chunks
```

Triggered only for benchmark/comparison/performance questions — keywords include:
- "compare", "benchmark", "industry average", "معيار", "مقارنة"
- "cancellation", "conversion", "roi", "churn", "repeat", "retention", and related terms

### 6. Store Manager — `store_manager.py`
```
store_manager.py
├── stores_registry.json       → Store metadata (id, name, tables, db_path)
├── create_store()             → New store + SQLite DB
├── get_all_stores()           → List all stores
├── load_store_data()          → Load all tables from store DB
├── update_store_tables()      → Track which tables are uploaded
└── delete_store()             → Remove store + DB file
```

---

## 🔄 Data Flow

### Chat Request Flow
```
User Question
     │
     ▼
_detect_language() → ar/en
     │
     ▼
_needs_action_engine()? → YES → Redirect message (no API)
     │ NO
     ▼
get_analytics_summary() → Full store context
     │
     ├── _needs_rag()? → YES → query_rag() → append benchmarks
     │
     ▼
Claude Sonnet API call
     │
     ▼
Response + Auto-chart (keyword-based)
```

### File Upload Flow
```
User uploads CSV/Excel
     │
     ▼
detect_table_type() → Claude Haiku
     │
     ▼
map_columns_with_claude() → Claude Haiku
     │
     ▼
Show mapping table → User reviews
     │
     ├── Correction note? → Re-map with note
     │
     ▼
apply_mapping() + clean_uploaded_data()
     │
     ▼
save_uploaded_data() → SQLite
     │
     ▼
Switch active store → Invalidate cache
```

### Smart Action Flow (Lazy)
```
Click "توصيات وإجراءات"
     │
     ▼
run_churn_reminders() → NO API CALLS
List of customers with priority scores
     │
     ▼
User clicks "✉️ توليد الرسالة"
     │
     ▼
generate_message_on_demand() → Claude Haiku
     │
     ▼
Message displayed + cached in session_state
```

---

## 🗄️ Database Schema

### SQLite (per store)
```sql
orders     (order_id, customer_id, product_id, quantity, 
            order_date, order_time, visitors, status, total_price)

customers  (customer_id, customer_name, city, gender, registration_date)

products   (product_id, product_name, category, 
            cost_price, selling_price, stock_quantity)

campaigns  (campaign_id, campaign_name, platform, budget, 
            clicks, start_date, end_date, conversions, campaign_revenue, roi)

carts      (cart_id, customer_id, product_id, quantity, 
            cart_date, status, total_orders)
```

---

## 💰 API Cost Optimization

| Action | Model | Calls | Notes |
|--------|-------|-------|-------|
| Chat question | claude-sonnet-4-5 | 1 | With full context |
| Report generation | claude-opus-4-5 | 2 | Part 1 + Part 2 |
| Customer message | claude-haiku-4-5-20251001 | 1 | On-demand only |
| Best send time | claude-haiku-4-5-20251001 | 1 | Per analysis run |
| File mapping | claude-haiku-4-5-20251001 | 2 | Detect + Map |
| RAG (if triggered) | — | 0 | Local FAISS |

**Key optimization:** Smart Actions use lazy loading — no API calls until user explicitly requests a message. This reduces cost from N×API_calls to 1 per interaction.

---

## 🔁 Human-in-the-Loop Design

The Action Engine does not automatically execute business actions.

Recommendations, messages, and suggested actions are generated for review and approval by the store owner before execution.

This design prioritizes reliability, transparency, and operational safety.

---

## 🔐 Security Considerations

- API key stored in `.env` file (not committed to git)
- SQLite databases stored locally
- No authentication layer (single-user local app)
- No PII encryption (not production-ready without additional security)
