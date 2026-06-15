import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import pandas as pd
import tempfile
from dotenv import load_dotenv
from ml_engine import forecast_product_demand, segment_customers, predict_churn
from analytics import *
from analytics import _ensure_total_price
from chatbot import ask_actionlytics
from action_engine import run_churn_reminders, run_restock_alerts, run_vip_offers, run_abandoned_cart_action, run_campaign_action, run_best_send_time, generate_message_on_demand
from report_engine import run_report
from file_analyzer import process_multiple_files, load_uploaded_table, save_uploaded_data, map_columns_with_claude, apply_mapping, clean_uploaded_data, load_all_store_tables
from store_manager import get_all_stores, get_store, create_store, update_store_tables, delete_store, load_store_data, get_store_db_path

sys.path.append(os.path.dirname(__file__))

def copy_button(text, key, lang="Arabic"):
    import re, json
    plain = re.sub(r'<[^>]+>', '', text)
    plain = re.sub(r'\*\*|__|\*|_|#{1,6} ', '', plain)
    plain = plain.strip()
    safe         = json.dumps(plain)
    copy_label   = "📋 نسخ الرسالة" if lang == "Arabic" else "📋 Copy Message"
    copied_label = "✅ تم النسخ"    if lang == "Arabic" else "✅ Copied!"
    st.markdown(f"""
        <button onclick="navigator.clipboard.writeText({safe}).then(() => {{
            this.textContent = '{copied_label}';
            setTimeout(() => this.textContent = '{copy_label}', 2000);
        }})" style="background-color: #E8F4F0; color: #1A2E2A; border: 1px solid #C8D8D4; border-radius: 8px;
            padding: 4px 12px; font-size: 0.8rem; cursor: pointer; margin-top: 8px; font-weight: 500;">{copy_label}</button>
    """, unsafe_allow_html=True)


def show_smart_actions(lang, db_path=None):
    tab_labels = {
        "Arabic": ["📩 رسائل الاسترجاع", "📦 تنبيهات المخزون", "👑 عروض VIP", "🛒 السلة المتروكة", "📊 توصيات الحملات"],
        "English": ["📩 Win-back Messages", "📦 Restock Alerts", "👑 VIP Offers", "🛒 Abandoned Cart", "📊 Campaign Tips"]
    }
    action_lang = "ar" if lang == "Arabic" else "en"
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels[lang])

    # ===== Tab 1: Churn Reminders =====
    with tab1:
        results = run_churn_reminders(language=action_lang, limit=None, db_path=db_path)
        if not results:
            st.info("✅ لا يوجد عملاء معرضون للمغادرة حالياً" if lang == "Arabic" else "✅ No customers at churn risk currently")
        else:
            critical = [r for r in results if '🔴' in r.get('priority_label','')]
            high     = [r for r in results if '🟡' in r.get('priority_label','')]
            normal   = [r for r in results if '🟢' in r.get('priority_label','')]
            st.markdown(f"**{'إجمالي' if lang=='Arabic' else 'Total'}:** {len(results)} | 🔴 {len(critical)} | 🟡 {len(high)} | 🟢 {len(normal)}")
            for group_label, group_results, icon in [
                (("حرج" if lang=="Arabic" else "Critical"), critical, "🔴"),
                (("عالي" if lang=="Arabic" else "High"), high, "🟡"),
                (("عادي" if lang=="Arabic" else "Normal"), normal, "🟢"),
            ]:
                if not group_results:
                    continue
                with st.expander(f"{icon} {group_label} — {len(group_results)} {'عميل' if lang=='Arabic' else 'customers'}", expanded=(icon=="🔴")):
                    for idx, r in enumerate(group_results):
                        unique_id = r.get('customer_id', f"{r.get('customer_name','customer')}_{idx}")
                        with st.expander(f"👤 {r['customer_name']} — {r['days_inactive']} {'يوم' if lang=='Arabic' else 'days'}"):
                            st.markdown(f"**{'الخصم الموصى به' if lang=='Arabic' else 'Recommended Discount'}:** {r['recommended_discount']}%")
                            st.divider()
                            st.markdown(f"**{'رسالة العميل' if lang=='Arabic' else 'Customer Message'}:**")
                            msg_key = f"churn_msg_{unique_id}"
                            if st.session_state.get(msg_key):
                                st.markdown(st.session_state[msg_key])
                            else:
                                if st.button("✉️ " + ("توليد الرسالة" if lang=="Arabic" else "Generate Message"), key=f"gen_churn_{unique_id}"):
                                    with st.spinner("جاري التوليد..." if lang=="Arabic" else "Generating..."):
                                        res = generate_message_on_demand("churn_reminder", r.get('context', {}), r.get('language','ar'), r.get('tone','friendly'))
                                        st.session_state[msg_key] = res['message']
                                        st.rerun()

    # ===== Tab 2: Restock Alerts =====
    with tab2:
        results = run_restock_alerts(language=action_lang, db_path=db_path)
        for i, r in enumerate(results):
            # Full inventory table
            if r.get('full_inventory') is not None and len(r['full_inventory']) > 0:
                inv          = r['full_inventory'].copy()
                status_emoji = {'OK': '✅', 'LOW': '⚠️', 'CRITICAL': '🔴', 'OUT OF STOCK': '❌'}
                status_col   = 'الحالة' if lang == 'Arabic' else 'Status'
                inv[status_col] = inv['status'].map(lambda s: f"{status_emoji.get(s,'⚪')} {s}")
                inv_display = inv[['product_name', 'category', 'stock_quantity', status_col, 'days_to_stockout']].copy()
                inv_display.columns = [
                    'المنتج'          if lang=='Arabic' else 'Product',
                    'الفئة'           if lang=='Arabic' else 'Category',
                    'المخزون الحالي'  if lang=='Arabic' else 'Current Stock',
                    'الأولوية'        if lang=='Arabic' else 'Status',
                    'أيام حتى النفاد' if lang=='Arabic' else 'Days to Stockout'
                ]
                days_col = 'أيام حتى النفاد' if lang=='Arabic' else 'Days to Stockout'
                inv_display[days_col] = inv_display[days_col].apply(lambda x: '—' if x == 999 else str(x))
                st.dataframe(inv_display, use_container_width=True)

            if r.get('products_affected', 0) > 0:
                # Business Recommendation
                if r.get('problem'):
                    st.markdown(f"🔍 **{'المشكلة' if lang=='Arabic' else 'Problem'}:** {r['problem']}")
                if r.get('recommendation'):
                    st.markdown(f"💡 **{'التوصية' if lang=='Arabic' else 'Recommendation'}:** {r['recommendation']}")
                if r.get('expected_outcome'):
                    st.markdown(f"📈 **{'النتيجة المتوقعة' if lang=='Arabic' else 'Expected Outcome'}:** {r['expected_outcome']}")
                if r.get('priority_label'):
                    st.markdown(f"**{'الأولوية' if lang=='Arabic' else 'Priority'}:** {r['priority_label']}")
                st.markdown(f"**{'المنتجات المتأثرة' if lang == 'Arabic' else 'Products Affected'}:** {r['products_affected']}")
                st.divider()
                st.markdown(f"**{'رسالة المدير' if lang=='Arabic' else 'Manager Alert'}:**")
                st.markdown(r['message'])
                if r.get('message'): copy_button(r['message'], key=f"restock_{i}", lang=lang)
            else:
                st.success(r.get('message', '✅ All products are well-stocked.'))

    # ===== Tab 3: VIP Offers =====
    with tab3:
        results = run_vip_offers(language=action_lang, limit=None, db_path=db_path)
        if not results:
            st.info("لا يوجد عملاء VIP حالياً" if lang == "Arabic" else "No VIP customers found")
        else:
            results = sorted(results, key=lambda x: x['total_spent'], reverse=True)
            st.markdown(f"**{'إجمالي عملاء VIP' if lang=='Arabic' else 'Total VIP'}:** {len(results)}")
            premium = [r for r in results if r['total_spent'] >= 5000]
            regular = [r for r in results if 2000 <= r['total_spent'] < 5000]
            basic   = [r for r in results if r['total_spent'] < 2000]
            for group_label, group_results, icon in [
                ("VIP Premium +5,000 SAR", premium, "💎"),
                ("VIP Regular 2,000-5,000 SAR", regular, "👑"),
                (("VIP Basic أقل من 2,000 SAR" if lang=="Arabic" else "VIP Basic under 2,000 SAR"), basic, "⭐"),
            ]:
                if not group_results:
                    continue
                with st.expander(f"{icon} {group_label} — {len(group_results)}", expanded=(icon=="💎")):
                    for idx, r in enumerate(group_results):
                        unique_id = r.get('customer_id', f"{r.get('customer_name','customer')}_{idx}")
                        with st.expander(f"👑 {r['customer_name']} — {r['total_spent']:,.0f} SAR"):
                            st.markdown(f"**{'الخصم الموصى به' if lang=='Arabic' else 'Recommended Discount'}:** {r['recommended_discount']}%")
                            st.divider()
                            st.markdown(f"**{'رسالة العميل' if lang=='Arabic' else 'Customer Message'}:**")
                            msg_key = f"vip_msg_{unique_id}"
                            if st.session_state.get(msg_key):
                                st.markdown(st.session_state[msg_key])
                            else:
                                if st.button("✉️ " + ("توليد الرسالة" if lang=="Arabic" else "Generate Message"), key=f"gen_vip_{unique_id}"):
                                    with st.spinner("جاري التوليد..." if lang=="Arabic" else "Generating..."):
                                        res = generate_message_on_demand("vip_offer", r.get('context', {}), r.get('language','ar'), r.get('tone','formal'))
                                        st.session_state[msg_key] = res['message']
                                        st.rerun()

    # ===== Tab 4: Abandoned Cart =====
    with tab4:
        results = run_abandoned_cart_action(language=action_lang, limit=None, db_path=db_path)
        if not results:
            st.info("لا توجد سلات متروكة" if lang == "Arabic" else "No abandoned carts found")
        else:
            results = sorted(results, key=lambda x: x['cart_value'], reverse=True)
            st.markdown(f"**{'إجمالي السلات المتروكة' if lang=='Arabic' else 'Total Abandoned Carts'}:** {len(results)}")
            high_val = [r for r in results if r['cart_value'] >= 200]
            mid_val  = [r for r in results if 50 <= r['cart_value'] < 200]
            low_val  = [r for r in results if r['cart_value'] < 50]
            for group_label, group_results, icon in [
                (("قيمة عالية +200 SAR" if lang=="Arabic" else "High Value +200 SAR"), high_val, "🔴"),
                (("قيمة متوسطة" if lang=="Arabic" else "Mid Value"), mid_val, "🟡"),
                (("قيمة منخفضة" if lang=="Arabic" else "Low Value"), low_val, "🟢"),
            ]:
                if not group_results:
                    continue
                with st.expander(f"{icon} {group_label} — {len(group_results)}", expanded=(icon=="🔴")):
                    for idx, r in enumerate(group_results):
                        unique_id = r.get('cart_id') or f"{r.get('customer_name','customer')}_{r.get('product_name','product')}_{idx}"
                        with st.expander(f"🛒 {r['customer_name']} — {r['product_name']} — {r['cart_value']} SAR"):
                            st.markdown(f"**{'الخصم الموصى به' if lang=='Arabic' else 'Recommended Discount'}:** {r['recommended_discount']}%")
                            st.divider()
                            st.markdown(f"**{'رسالة العميل' if lang=='Arabic' else 'Customer Message'}:**")
                            msg_key = f"cart_msg_{unique_id}"
                            if st.session_state.get(msg_key):
                                st.markdown(st.session_state[msg_key])
                            else:
                                if st.button("✉️ " + ("توليد الرسالة" if lang=="Arabic" else "Generate Message"), key=f"gen_cart_{unique_id}"):
                                    with st.spinner("جاري التوليد..." if lang=="Arabic" else "Generating..."):
                                        res = generate_message_on_demand("abandoned_cart", r.get('context', {}), r.get('language','ar'), r.get('tone','friendly'))
                                        st.session_state[msg_key] = res['message']
                                        st.rerun()

    # ===== Tab 5: Campaign Tips =====
    with tab5:
        results = run_campaign_action(language=action_lang, db_path=db_path)
        if not results:
            st.info("لا توجد بيانات حملات" if lang == "Arabic" else "No campaign data found")
        for i, r in enumerate(results):
            # Business Recommendation
            if r.get('problem'):
                st.markdown(f"🔍 **{'المشكلة' if lang=='Arabic' else 'Problem'}:** {r['problem']}")
            if r.get('recommendation'):
                st.markdown(f"💡 **{'التوصية' if lang=='Arabic' else 'Recommendation'}:** {r['recommendation']}")
            if r.get('expected_outcome'):
                st.markdown(f"📈 **{'النتيجة المتوقعة' if lang=='Arabic' else 'Expected Outcome'}:** {r['expected_outcome']}")
            if r.get('priority_label'):
                st.markdown(f"**{'الأولوية' if lang=='Arabic' else 'Priority'}:** {r['priority_label']}")
            st.markdown(f"**{'أفضل حملة' if lang == 'Arabic' else 'Best Campaign'}:** {r['best_campaign']} — {r['best_platform']}")
            st.divider()
            st.markdown(r['message'])
            if r.get('message'): copy_button(r['message'], key=f"campaign_{i}", lang=lang)


st.set_page_config(page_title="Actionlytics", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

if "language" not in st.session_state:
    st.session_state.language = "Arabic"
if "active_store_id" not in st.session_state:
    st.session_state.active_store_id = "demo"

def get_active_store_data():
    store_id = st.session_state.active_store_id
    if store_id == "demo":
        return load_data()
    else:
        products_df, customers_df, orders_df, campaigns_df, carts_df = load_store_data(store_id)
        if orders_df is None:
            return load_data()
        orders_df = _ensure_total_price(orders_df, products_df)
        return products_df, customers_df, orders_df, campaigns_df

def get_active_db_path():
    store_id = st.session_state.active_store_id
    if store_id == "demo":
        return None
    return get_store_db_path(store_id)

products_df, customers_df, orders_df, campaigns_df = get_active_store_data()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', 'IBM Plex Sans Arabic', sans-serif; }
.stApp { background-color: #F0F4F3; color: #1A2E2A; }
div.stButton > button { background-color: #FFFFFF; color: #1A2E2A; border: 1px solid #C8D8D4; border-radius: 10px; padding: 6px 18px; font-size: 0.85rem; transition: all 0.2s; font-weight: 500; }
div.stButton > button:hover { background-color: #1D9E75; color: #FFFFFF; border-color: #1D9E75; }
.logo-text { font-size: 1.4rem; font-weight: 600; color: #1A2E2A; letter-spacing: -0.3px; }
.logo-dot { color: #1D9E75; }
.hero-title { font-size: 1.5rem; font-weight: 600; color: #1A2E2A; margin-bottom: 4px; }
.hero-sub { font-size: 0.95rem; color: #5A7A72; margin-bottom: 20px; }
.metric-card { background: #FFFFFF; border: 1px solid #E0EEEA; border-radius: 12px; padding: 16px 18px; }
.metric-value { font-size: 1.6rem; font-weight: 600; color: #1D9E75; margin: 4px 0 2px; }
.metric-label { font-size: 0.75rem; color: #7A9A92; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }
.metric-sub { font-size: 0.78rem; color: #A0B8B2; }
.section-title { font-size: 1rem; font-weight: 600; color: #1A2E2A; border-left: 3px solid #1D9E75; padding-left: 10px; margin: 20px 0 12px 0; }
.chat-message-user { background: #FFFFFF; border: 1px solid #E0EEEA; border-radius: 14px 14px 4px 14px; padding: 10px 14px; margin: 6px 0; color: #1A2E2A; text-align: right; font-size: 0.9rem; }
.chat-message-bot { background: #F0FAF6; border: 1px solid #C8E8DA; border-radius: 14px 14px 14px 4px; padding: 14px 18px; margin: 6px 0; color: #1A2E2A; font-size: 0.9rem; line-height: 1.8; }
.chat-message-bot h1, .chat-message-bot h2, .chat-message-bot h3 { font-size: 1rem; font-weight: 600; color: #1A2E2A; margin: 12px 0 5px; }
.chat-message-bot p, .chat-message-bot li { font-size: 0.88rem; color: #2A4A42; font-weight: 400; line-height: 1.8; }
.report-text-area { background: #FFFFFF; border: 1px solid #E0EEEA; border-radius: 12px; padding: 24px 28px; margin-top: 12px; line-height: 1.9; }
.report-text-area h1, .report-text-area h2, .report-text-area h3 { font-size: 1.05rem; font-weight: 600; color: #1A2E2A; margin: 16px 0 6px; }
.report-text-area p, .report-text-area li { font-size: 0.88rem; color: #2A4A42; font-weight: 400; line-height: 1.8; }
div[data-testid="stChatInput"] { background: #FFFFFF; border: 1px solid #D8E8E4; border-radius: 12px; }
.stSelectbox > div > div { background: #FFFFFF; border: 1px solid #D8E8E4; border-radius: 10px; color: #1A2E2A; font-size: 0.88rem; }
.stMetric label { font-size: 0.78rem; color: #7A9A92; font-weight: 500; }
.stMetric [data-testid="metric-container"] div { color: #1D9E75; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

texts = {
    "English": {
        "hero_title": "How can I help you today?", "hero_sub": "Choose an option or ask your question directly",
        "btn_kpi": "Store Performance", "btn_action": "Smart Actions",
        "btn_report": "Generate Report", "btn_upload": "Upload My Data",
        "performance": "Store Performance", "total_revenue": "TOTAL REVENUE",
        "cancellation": "CANCELLATION RATE", "conversion": "CONVERSION RATE",
        "avg_order": "AVG ORDER VALUE", "top_product": "TOP PRODUCT",
        "total_orders": "TOTAL ORDERS", "total_customers": "TOTAL CUSTOMERS", "of_orders": "of orders",
        "visitors": "visitors to buyers", "units": "units", "sar": "SAR",
        "assistant": "Actionlytics Assistant", "chat_placeholder": "Ask about your store...",
        "upload_title": "Upload Store Data", "report_title": "Dynamic Report",
        "report_lang_label": "Report language", "generate_btn": "Generate Report",
        "period_label": "Select period", "smart_actions_title": "Smart Actions",
        "select_store": "Select Store", "upload_new_store": "New Store",
        "upload_existing_store": "Add to Existing Store",
        "store_name_placeholder": "Store name (e.g. Fashion Store)",
        "select_existing_store": "Select store to add data to",
    },
    "Arabic": {
        "hero_title": "كيف أساعدك اليوم؟", "hero_sub": "اختر ما تريد أو اكتب سؤالك مباشرة",
        "btn_kpi": "أداء المتجر", "btn_action": "توصيات وإجراءات",
        "btn_report": "توليد تقرير", "btn_upload": "رفع بياناتي",
        "performance": "أداء المتجر", "total_revenue": "إجمالي الإيرادات",
        "cancellation": "معدل الإلغاء", "conversion": "معدل التحويل",
        "avg_order": "متوسط الفاتورة", "top_product": "أفضل منتج",
        "total_orders": "عدد الطلبات", "total_customers": "عدد العملاء", "of_orders": "من الطلبات",
        "visitors": "من الزوار اشتروا", "units": "وحدة", "sar": "ريال سعودي",
        "assistant": "مساعد Actionlytics", "chat_placeholder": "اسأل عن متجرك...",
        "upload_title": "رفع بيانات المتجر", "report_title": "التقرير الديناميكي",
        "report_lang_label": "لغة التقرير", "generate_btn": "توليد التقرير",
        "period_label": "اختر الفترة", "smart_actions_title": "توصيات وإجراءات",
        "select_store": "اختر المتجر", "upload_new_store": "متجر جديد",
        "upload_existing_store": "إضافة لمتجر محفوظ",
        "store_name_placeholder": "اسم المتجر (مثال: متجر الملابس)",
        "select_existing_store": "اختر المتجر لإضافة البيانات إليه",
    }
}

lang = st.session_state.language
t = texts[lang]

# ===== Topbar =====
col_logo, col_spacer, col_lang = st.columns([4, 4, 2])
with col_logo:
    st.markdown('<div class="logo-text">Action<span class="logo-dot">lytics</span></div>', unsafe_allow_html=True)
with col_lang:
    selected_lang = st.selectbox("", options=["Arabic", "English"], index=0 if lang == "Arabic" else 1, label_visibility="collapsed")
    if selected_lang != lang:
        st.session_state.language = selected_lang
        st.rerun()

st.markdown("---")

# ===== Store Selector =====
st.markdown(f'<div class="section-title">🏪 {t["select_store"]}</div>', unsafe_allow_html=True)
all_stores = get_all_stores()
store_options = {s["id"]: f"{'🏪' if s['id'] != 'demo' else '📊'} {s['name']}" for s in all_stores}
selected_store_id = st.selectbox("", options=list(store_options.keys()),
    format_func=lambda x: store_options[x],
    index=list(store_options.keys()).index(st.session_state.active_store_id) if st.session_state.active_store_id in store_options else 0,
    label_visibility="collapsed")
if selected_store_id != st.session_state.active_store_id:
    st.session_state.active_store_id = selected_store_id
    st.session_state.messages = []
    st.session_state.conversation_history = []
    from chatbot import invalidate_analytics_cache
    invalidate_analytics_cache(get_store_db_path(selected_store_id))
    st.rerun()

active_store = get_store(st.session_state.active_store_id)
if active_store:
    tables_info = f"جداول: {', '.join(active_store['tables'])}" if active_store['tables'] else "لا توجد بيانات مرفوعة بعد"
    st.caption(f"📁 {active_store['name']} — {tables_info}")

products_df, customers_df, orders_df, campaigns_df = get_active_store_data()
active_db_path = get_active_db_path()

st.markdown("---")

# ===== Hero + Quick Actions =====
st.markdown(f'<div class="hero-title">{t["hero_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">{t["hero_sub"]}</div>', unsafe_allow_html=True)

col_q1, col_q2, col_q3, col_q4 = st.columns(4)
with col_q1:
    if st.button(f"📊 {t['btn_kpi']}", use_container_width=True):
        st.session_state["show_kpi"] = not st.session_state.get("show_kpi", True)
with col_q2:
    if st.button(f"⚡ {t['btn_action']}", use_container_width=True):
        st.session_state["show_smart_actions"] = not st.session_state.get("show_smart_actions", False)
with col_q3:
    if st.button(f"📋 {t['btn_report']}", use_container_width=True):
        st.session_state["show_report"] = not st.session_state.get("show_report", False)
        if not st.session_state["show_report"]:
            if "last_report" in st.session_state:
                del st.session_state["last_report"]
with col_q4:
    if st.button(f"📂 {t['btn_upload']}", use_container_width=True):
        st.session_state["show_upload"] = not st.session_state.get("show_upload", False)
        if not st.session_state["show_upload"]:
            for k in ["upload_mode", "upload_results", "upload_confirmed", "pending_store_save"]:
                st.session_state.pop(k, None)

# ===== Quick Questions =====
QUICK_QUESTIONS = {
    "Arabic": {
        "📊 المبيعات": [
            "كم إجمالي إيراداتي؟",
            "ما أفضل منتج مبيعاً؟",
            "ما الفئة الأكثر إيراداً؟",
            "أظهر لي المبيعات الشهرية",
            "ما الإيرادات حسب كل مدينة؟",
        ],
        "👥 العملاء": [
            "قسّم لي العملاء حسب قيمتهم",
            "من العملاء المعرضون للرحيل؟",
            "كم معدل الشراء المتكرر؟",
            "اعرض أسماء عملاء VIP",
            "ما توزيع العملاء حسب المدن؟",
        ],
        "📣 الحملات": [
            "ما أفضل حملة إعلانية عندي؟",
            "ما عائد الاستثمار للحملات؟",
            "أي منصة إعلانية أداؤها أفضل؟",
            "هل ميزانية الحملات فعالة؟",
            "كيف أحسن أداء الحملات؟",
        ],
        "🤖 التنبؤ": [
            "توقع لي المبيعات الأشهر القادمة",
            "هل في أشهر غير طبيعية في مبيعاتي؟",
            "ما المنتجات التي يشتريها العملاء مع بعض؟",
            "ما المنتج المتوقع عليه أعلى طلب؟",
            "ما المنتجات التي ينخفض الطلب عليها؟",
        ],
    },

    "English": {
        "📊 Sales": [
            "What is my total revenue?",
            "What is my best selling product?",
            "Which category generates most revenue?",
            "Show me monthly sales trend",
            "Show revenue by city",
        ],

        "👥 Customers": [
            "Segment my customers by value",
            "Which customers are at risk of leaving?",
            "What is my repeat purchase rate?",
            "Show VIP customer names",
            "How are customers distributed by city?",
        ],

        "📣 Campaigns": [
            "What is my best performing campaign?",
            "What is my campaign ROI?",
            "Which marketing platform performs best?",
            "Is my campaign budget effective?",
            "How can I improve campaign performance?",
        ],

        "🤖 Forecast": [
            "Forecast my sales for next months",
            "Are there unusual months in my sales?",
            "Which products are purchased together?",
            "Which product has the highest predicted demand?",
            "Which products show declining demand?",
        ],
    },
}

with st.expander("💡 " + ("أسئلة جاهزة" if lang == "Arabic" else "Quick Questions"), expanded=False):
    qq = QUICK_QUESTIONS[lang]
    for category, questions in qq.items():
        st.markdown(f"**{category}**")
        cols = st.columns(len(questions))
        for i, q in enumerate(questions):
            with cols[i]:
                if st.button(q, key=f"qq_{category}_{i}", use_container_width=True):
                    st.session_state["quick_question"] = q
                    st.rerun()

# ===== KPIs =====
if st.session_state.get("show_kpi", True):
    st.markdown(f'<div class="section-title">📊 {t["performance"]}</div>', unsafe_allow_html=True)
    if orders_df is not None and len(orders_df) > 0:
        orders_df_dates = pd.to_datetime(orders_df['order_date'])
        min_date = orders_df_dates.min().date()
        max_date = orders_df_dates.max().date()
        col_df1, col_df2, col_df3 = st.columns([2, 2, 1])
        with col_df1:
            kpi_start = st.date_input("من" if lang=="Arabic" else "From", value=min_date, min_value=min_date, max_value=max_date, key="kpi_start")
        with col_df2:
            kpi_end   = st.date_input("إلى" if lang=="Arabic" else "To",   value=max_date, min_value=min_date, max_value=max_date, key="kpi_end")
        with col_df3:
            st.markdown("<br>", unsafe_allow_html=True)
            reset_dates = st.button("↺" + (" تحديث" if lang=="Arabic" else " Reset"), key="kpi_reset")
        if reset_dates:
            kpi_start = min_date
            kpi_end   = max_date

        filtered_orders_kpi = orders_df[
            (orders_df_dates.dt.date >= kpi_start) &
            (orders_df_dates.dt.date <= kpi_end)
        ]

        total_revenue     = get_total_revenue(filtered_orders_kpi, products_df)
        cancellation_rate = get_cancellation_rate(filtered_orders_kpi)
        conversion_rate   = get_conversion_rate(filtered_orders_kpi)
        completed_orders  = _ensure_total_price(filtered_orders_kpi[filtered_orders_kpi['status'].isin(['completed', 'delivered'])], products_df)
        avg_order         = completed_orders['total_price'].mean() if len(completed_orders) > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["total_revenue"]}</div><div class="metric-value">{total_revenue:,.0f}</div><div class="metric-sub">{t["sar"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["cancellation"]}</div><div class="metric-value">{cancellation_rate}%</div><div class="metric-sub">{t["of_orders"]}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["conversion"]}</div><div class="metric-value">{conversion_rate}%</div><div class="metric-sub">{t["visitors"]}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["avg_order"]}</div><div class="metric-value">{avg_order:,.0f}</div><div class="metric-sub">{t["sar"]}</div></div>', unsafe_allow_html=True)

        col5, col6, col7 = st.columns(3)
        if products_df is not None and len(products_df) > 0:
            best_products = get_best_products(filtered_orders_kpi, products_df)
            with col5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{t["top_product"]}</div><div class="metric-value" style="font-size:1rem">{best_products.iloc[0]["product_name"]}</div><div class="metric-sub">{best_products.iloc[0]["quantity"]} {t["units"]}</div></div>', unsafe_allow_html=True)
        # كرت عدد الطلبات — يتغير مع Date Filter
        total_orders_count = filtered_orders_kpi["order_id"].nunique() if "order_id" in filtered_orders_kpi.columns else len(filtered_orders_kpi)
        with col6:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["total_orders"]}</div><div class="metric-value">{total_orders_count:,}</div><div class="metric-sub">{t["of_orders"]}</div></div>', unsafe_allow_html=True)
        # كرت عدد العملاء — يتغير مع Date Filter
        total_cust_count = filtered_orders_kpi["customer_id"].nunique() if "customer_id" in filtered_orders_kpi.columns else 0
        with col7:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["total_customers"]}</div><div class="metric-value">{total_cust_count:,}</div><div class="metric-sub">{"عميل نشط" if lang=="Arabic" else "active customers"}</div></div>', unsafe_allow_html=True)
    else:
        st.info("📂 ارفع بيانات المتجر لعرض مؤشرات الأداء" if lang == "Arabic" else "📂 Upload store data to view performance metrics")
# ===== Smart Actions =====
if st.session_state.get("show_smart_actions", False):
    st.markdown(f'<div class="section-title">⚡ {t["smart_actions_title"]}</div>', unsafe_allow_html=True)
    show_smart_actions(lang, db_path=active_db_path)
# ===== File Upload =====
if st.session_state.get("show_upload", False):
    st.markdown(f'<div class="section-title">📂 {t["upload_title"]}</div>', unsafe_allow_html=True)
    if "upload_mode" not in st.session_state:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button(f"🆕 {t['upload_new_store']}", use_container_width=True):
                st.session_state["upload_mode"] = "new"
                st.session_state["upload_confirmed"] = False
                st.session_state["pending_store_save"] = False
                st.rerun()
        with col_m2:
            saved_stores = [s for s in all_stores if s["id"] != "demo"]
            if saved_stores:
                if st.button(f"➕ {t['upload_existing_store']}", use_container_width=True):
                    st.session_state["upload_mode"] = "existing"
                    st.session_state["upload_confirmed"] = False
                    st.session_state["pending_store_save"] = False
                    st.rerun()
            else:
                st.button(f"➕ {t['upload_existing_store']}", use_container_width=True, disabled=True)
                st.caption("لا توجد متاجر محفوظة بعد" if lang == "Arabic" else "No saved stores yet")
    if st.session_state.get("upload_mode") == "existing":
        saved_stores  = [s for s in all_stores if s["id"] != "demo"]
        saved_options = {s["id"]: f"🏪 {s['name']}" for s in saved_stores}
        st.markdown(f"**{t['select_existing_store']}**")
        target_store_id = st.selectbox("", options=list(saved_options.keys()),
            format_func=lambda x: saved_options[x], label_visibility="collapsed", key="upload_target_store")
        st.session_state["upload_target_store_id"] = target_store_id
    if st.session_state.get("upload_mode") in ("new", "existing"):
        mode_label = f"🆕 {t['upload_new_store']}" if st.session_state["upload_mode"] == "new" else f"➕ {t['upload_existing_store']}"
        st.caption(f"{'الوضع' if lang == 'Arabic' else 'Mode'}: {mode_label}")
        uploaded_files = st.file_uploader("ارفع ملف أو أكثر" if lang == "Arabic" else "Upload one or more files",
            type=["csv", "xlsx", "xls"], accept_multiple_files=True, label_visibility="visible")
        if uploaded_files and st.button("🔍 تحليل الملفات", use_container_width=True):
            with st.spinner("Actionlytics يحلل بياناتك..."):
                tmp_paths = []
                for uf in uploaded_files:
                    suffix = ".csv" if uf.name.endswith(".csv") else ".xlsx"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_paths.append(tmp.name)
                results = process_multiple_files(tmp_paths)
                for path in tmp_paths:
                    os.unlink(path)
                st.session_state["upload_results"]     = results
                st.session_state["upload_confirmed"]   = False
                st.session_state["pending_store_save"] = False
if "upload_results" in st.session_state and st.session_state["upload_results"]:
    results = st.session_state["upload_results"]
    for i, r in enumerate(results):
        if r["success"]:
            st.markdown(f"#### ✅ {r['file']} — نوع البيانات: `{r['table_name']}`")
            st.table([{"عمودك": k, "عمودنا": v} for k, v in r["mapping"].items()])
            st.dataframe(r["df"].head(3))
            # ===== Correction Note UI — واضح ومبرز #26 =====
            st.markdown("""<div style="background:#FFF8E1;border:1px solid #FFD54F;border-radius:8px;padding:10px 14px;margin-top:8px;"><b>🔧 هل في خطأ بالتطابق؟</b> اكتب ملاحظتك وسيعيد النظام التحليل مرة أخرى</div>""", unsafe_allow_html=True)
            col_fix1, col_fix2 = st.columns([3, 1])
            with col_fix1:
                correction = st.text_input("مثال: عمود campaign_revenue هو الإيراد الفعلي للحملة مو roi", key=f"fix_{i}")
            with col_fix2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 إعادة التحليل", key=f"reanalyze_{i}", use_container_width=True) and correction:
                    with st.spinner("Actionlytics يعيد التحليل بناءً على ملاحظتك..."):
                        new_mapping = map_columns_with_claude(r["df_raw"].columns, r["table_name"], correction_note=correction)
                        new_df = apply_mapping(r["df_raw"], new_mapping)
                        new_df = clean_uploaded_data(new_df)
                        results[i]["df"] = new_df
                        results[i]["mapping"] = new_mapping
                        st.session_state["upload_results"] = results
                        st.rerun()
        else:
            st.error(f"❌ {r['file']} — فشل التحليل: {r.get('error', '')}")
    if not st.session_state.get("upload_confirmed", False):
        st.warning("⚠️ تأكد من صحة التطابق قبل الحفظ" if lang == "Arabic" else "⚠️ Verify mappings before saving")
        if st.button("✅ تأكيد", use_container_width=True):
            st.session_state["pending_store_save"] = True
            st.rerun()
    if st.session_state.get("pending_store_save", False):
        st.markdown("---")
        upload_mode = st.session_state.get("upload_mode", "new")
        if upload_mode == "existing":
            target_id    = st.session_state.get("upload_target_store_id")
            target_store = get_store(target_id)
            st.markdown(f"### ➕ {'إضافة البيانات إلى' if lang == 'Arabic' else 'Adding data to'}: **{target_store['name']}**")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                if st.button("💾 حفظ وتحليل" if lang == "Arabic" else "💾 Save & Analyze", use_container_width=True):
                    with st.spinner("Actionlytics يحفظ بياناتك..."):
                        db_path = get_store_db_path(target_id)
                        for r in results:
                            if r["success"]:
                                save_uploaded_data(r["df"], r["table_name"], db_path=db_path)
                                update_store_tables(target_id, r["table_name"])
                        st.session_state.active_store_id = target_id
                        st.session_state["upload_confirmed"] = True
                        st.session_state["pending_store_save"] = False
                        st.session_state["messages"] = []
                        st.session_state["conversation_history"] = []
                        from chatbot import invalidate_analytics_cache
                        invalidate_analytics_cache(db_path)
                        st.success(f"✅ تمت الإضافة إلى '{target_store['name']}'" if lang == "Arabic" else f"✅ Data added to '{target_store['name']}'")
                        st.rerun()
            with col_a2:
                if st.button("↩️ إلغاء" if lang == "Arabic" else "↩️ Cancel", use_container_width=True):
                    st.session_state["pending_store_save"] = False
                    st.rerun()
        else:
            st.markdown("### 💾 هل تريد حفظ بيانات هذا المتجر؟" if lang == "Arabic" else "### 💾 Save store data?")
            col_yn1, col_yn2 = st.columns(2)
            with col_yn1:
                save_choice = st.radio("",
                    options=["نعم، احفظ للمرة القادمة", "لا، استخدم فقط الآن"] if lang == "Arabic" else ["Yes, save for later", "No, use this session only"],
                    label_visibility="collapsed")
            with col_yn2:
                if "نعم" in save_choice or "Yes" in save_choice:
                    store_name = st.text_input("اسم المتجر" if lang == "Arabic" else "Store Name", placeholder=t["store_name_placeholder"])
                    if st.button("💾 حفظ وتحليل", use_container_width=True) and store_name:
                        with st.spinner("Actionlytics يحفظ بياناتك..."):
                            new_store = create_store(store_name)
                            db_path   = get_store_db_path(new_store["id"])
                            for r in results:
                                if r["success"]:
                                    save_uploaded_data(r["df"], r["table_name"], db_path=db_path)
                                    update_store_tables(new_store["id"], r["table_name"])
                            st.session_state.active_store_id = new_store["id"]
                            st.session_state["upload_confirmed"] = True
                            st.session_state["pending_store_save"] = False
                            st.session_state["messages"] = []
                            st.session_state["conversation_history"] = []
                            from chatbot import invalidate_analytics_cache
                            invalidate_analytics_cache(db_path)
                            st.success(f"✅ تم حفظ متجر '{store_name}' بنجاح!" if lang == "Arabic" else f"✅ Store '{store_name}' saved!")
                            st.rerun()
                else:
                    if st.button("▶️ تحليل بدون حفظ", use_container_width=True):
                        with st.spinner("Actionlytics يحلل بياناتك..."):
                            tmp_db = "temp_session.db"
                            for r in results:
                                if r["success"]:
                                    save_uploaded_data(r["df"], r["table_name"], db_path=tmp_db)
                            st.session_state["temp_db_path"]       = tmp_db
                            st.session_state["upload_confirmed"]   = True
                            st.session_state["pending_store_save"] = False
                            st.session_state["quick_question"]     = "حلل لي بياناتي المرفوعة" if lang == "Arabic" else "Analyze my uploaded data"
                            st.rerun()
    if st.session_state.get("upload_confirmed", False):
        successful   = [r for r in results if r["success"]]
        tables_saved = [r["table_name"] for r in successful]
        st.success(f"✅ تم تحليل {len(successful)} ملف — الجداول: {', '.join(tables_saved)}" if lang == "Arabic" else f"✅ {len(successful)} files analyzed — Tables: {', '.join(tables_saved)}")
# ===== Chatbot =====
st.markdown("---")
if "show_chat" not in st.session_state:
    st.session_state.show_chat = True

chat_col1, chat_col2 = st.columns([6, 1])
with chat_col1:
    st.markdown(f'<div class="section-title">🤖 {t["assistant"]}</div>', unsafe_allow_html=True)
with chat_col2:
    toggle_label = "🙈 إخفاء المحادثة" if (lang == "Arabic" and st.session_state.show_chat) else (
        "💬 إظهار المحادثة" if lang == "Arabic" else ("🙈 Hide Chat" if st.session_state.show_chat else "💬 Show Chat")
    )
    if st.button(toggle_label, use_container_width=True, key="toggle_chat_visibility"):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if st.session_state.show_chat:
    if "quick_question" in st.session_state:
        quick_q = st.session_state.pop("quick_question")
        st.session_state.messages.append({"role": "user", "content": quick_q})
        with st.spinner("Actionlytics يفكر..."):
            uploaded_df    = None
            uploaded_table = None
            if st.session_state.get("upload_confirmed") and "upload_results" in st.session_state:
                orders_result = next((r for r in st.session_state["upload_results"] if r["success"] and r["table_name"] == "orders"), None)
                if orders_result:
                    uploaded_df    = orders_result["df"]
                    uploaded_table = "orders"
            answer, st.session_state.conversation_history = ask_actionlytics(
                quick_q, st.session_state.conversation_history,
                uploaded_df=uploaded_df, uploaded_table=uploaded_table,
                db_path=active_db_path, products_df=products_df)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message-user">🧑 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-bot">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    if prompt := st.chat_input(t["chat_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-message-user">🧑 {prompt}</div>', unsafe_allow_html=True)
        with st.spinner("Actionlytics يفكر..."):
            uploaded_df    = None
            uploaded_table = None
            if st.session_state.get("upload_confirmed") and "upload_results" in st.session_state:
                orders_result = next((r for r in st.session_state["upload_results"] if r["success"] and r["table_name"] == "orders"), None)
                if orders_result:
                    uploaded_df    = orders_result["df"]
                    uploaded_table = "orders"
            answer, st.session_state.conversation_history = ask_actionlytics(
                prompt, st.session_state.conversation_history,
                uploaded_df=uploaded_df, uploaded_table=uploaded_table,
                db_path=active_db_path, products_df=products_df)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f'<div class="chat-message-bot">🤖 {answer}</div>', unsafe_allow_html=True)
        prompt_lower = prompt.lower()
        if orders_df is not None and any(w in prompt_lower for w in ['monthly','revenue','مبيعات','إيرادات','شهري','شهرية']):
            monthly = get_period_comparison(orders_df, products_df)
            fig = px.line(monthly, x='month', y='total_price', markers=True,
                labels={'month': 'الشهر' if lang=='Arabic' else 'Month',
                        'total_price': 'الإيرادات (ريال)' if lang=='Arabic' else 'Revenue (SAR)'})
            fig.update_traces(line_color='#1D9E75', marker=dict(color='#1D9E75', size=8), fill='tozeroy', fillcolor='rgba(29,158,117,0.08)')
            fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                xaxis=dict(showgrid=False, tickangle=45),
                yaxis=dict(showgrid=True, gridcolor='#E8F4F0'),
                margin=dict(l=20, r=20, t=20, b=20), height=320)
            st.plotly_chart(fig, use_container_width=True)
        elif orders_df is not None and products_df is not None and any(w in prompt_lower for w in ['category','categories','فئة','فئات','الفئة','الفئات','حسب الفئ','إيرادات الفئ']):
            rev_cat = get_revenue_by_category(orders_df, products_df)
            fig = px.pie(rev_cat, values='total_price', names='category', color_discrete_sequence=['#1D9E75','#378ADD','#D85A30','#BA7517'])
            fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                margin=dict(l=20, r=20, t=20, b=20), height=320, legend=dict(bgcolor='#FFFFFF'))
            st.plotly_chart(fig, use_container_width=True)
        elif orders_df is not None and products_df is not None and any(w in prompt_lower for w in ['product','منتج','منتجات','best','أفضل','مبيعا']):
            best = get_best_products(orders_df, products_df)
            fig = px.bar(best, x='quantity', y='product_name', orientation='h', color='quantity',
                color_continuous_scale=['#9FE1CB','#1D9E75'],
                labels={'quantity': 'الكمية' if lang=='Arabic' else 'Quantity',
                        'product_name': 'المنتج' if lang=='Arabic' else 'Product'})
            fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                xaxis=dict(showgrid=True, gridcolor='#E8F4F0'),
                yaxis=dict(showgrid=False),
                margin=dict(l=20, r=20, t=20, b=20), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        elif campaigns_df is not None and any(w in prompt_lower for w in ['campaign','حملة','حملات','roi','تسويق']):
            camp = get_campaign_performance(campaigns_df)
            fig = px.bar(camp, x='campaign_name', y='roi', color='roi', color_continuous_scale=['#9FE1CB','#1D9E75'])
            fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                xaxis=dict(showgrid=False, tickangle=45), yaxis=dict(showgrid=True, gridcolor='#E8F4F0'),
                margin=dict(l=20, r=20, t=20, b=20), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        elif orders_df is not None and customers_df is not None and any(w in prompt_lower for w in ['segment','عملاء','تقسيم','vip','فئات']):
            customer_stats, seg_summary = segment_customers(orders_df, customers_df)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig = px.bar(seg_summary, x='segment_label', y='count', color='avg_spent',
                    color_continuous_scale=['#9FE1CB','#1D9E75'], text='count')
                fig.update_traces(textposition='outside')
                fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E8F4F0'),
                    margin=dict(l=20, r=20, t=20, b=20), height=320, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_s2:
                fig_pie = px.pie(seg_summary, values='count', names='segment_label',
                    color_discrete_map={'VIP':'#1D9E75','Regular':'#378ADD','Dormant':'#D85A30'})
                fig_pie.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                    margin=dict(l=20, r=20, t=20, b=20), height=320)
                st.plotly_chart(fig_pie, use_container_width=True)
        elif orders_df is not None and customers_df is not None and any(w in prompt_lower for w in ['churn','رحيل','خطر','مغادرة','معرضين']):
            churn_df, churn_summary = predict_churn(orders_df, customers_df)
            if len(churn_df) > 0:
                fig = px.bar(churn_df['risk_level'].value_counts().reset_index(), x='risk_level', y='count',
                    color='risk_level', color_discrete_map={'High':'#D85A30','Medium':'#BA7517','Low':'#1D9E75'})
                fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E8F4F0'),
                    margin=dict(l=20, r=20, t=20, b=20), height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        elif any(w in prompt_lower for w in ['action','توصيات','إجراءات','رسالة','استرجاع','خامل','مخزون','سلة','حملة توصية']):
            st.markdown(f'<div class="section-title">⚡ {t["smart_actions_title"]}</div>', unsafe_allow_html=True)
            show_smart_actions(lang, db_path=active_db_path)
# ===== Dynamic Report =====
if st.session_state.get("show_report", False):
    st.markdown("---")
    st.markdown(f'<div class="section-title">📋 {t["report_title"]}</div>', unsafe_allow_html=True)
    col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
    with col_r1:
        period_option = st.selectbox(t["period_label"],
            options=["custom","today","week","month","quarter","half_year","year","two_years","all"],
            format_func=lambda x: {
                "custom":    "📅 Custom Period"  if lang == "English" else "📅 فترة مخصصة",
                "today":     "Today"             if lang == "English" else "اليوم",
                "week":      "Last Week"         if lang == "English" else "آخر أسبوع",
                "month":     "Last Month"        if lang == "English" else "آخر شهر",
                "quarter":   "Last 3 Months"     if lang == "English" else "آخر 3 أشهر",
                "half_year": "Last 6 Months"     if lang == "English" else "آخر 6 أشهر",
                "year":      "Last Year"         if lang == "English" else "آخر سنة",
                "two_years": "Last 2 Years"      if lang == "English" else "آخر سنتين",
                "all":       "All Time"          if lang == "English" else "منذ البداية",
            }[x])
    with col_r2:
        report_lang = st.selectbox(t["report_lang_label"], options=["ar","en"], format_func=lambda x: "عربي" if x=="ar" else "English")
    with col_r3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button(f"🚀 {t['generate_btn']}", use_container_width=True)
    if period_option == "custom":
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("من تاريخ", value=None)
        with col_d2:
            end_date = st.date_input("إلى تاريخ", value=None)
    else:
        start_date = end_date = None
    if generate_clicked:
        with st.spinner("Actionlytics يولّد التقرير..."):
            active_store_obj   = get_store(st.session_state.active_store_id)
            report_data_source = active_store_obj['name'] if active_store_obj else 'Demo Store'
            if period_option == "custom" and start_date and end_date:
                result = run_report(start_date=str(start_date), end_date=str(end_date), language=report_lang,
                    products_df=products_df, customers_df=customers_df, orders_df=orders_df, campaigns_df=campaigns_df,
                    data_source=report_data_source)
            elif period_option != "custom":
                result = run_report(period=period_option, language=report_lang,
                    products_df=products_df, customers_df=customers_df, orders_df=orders_df, campaigns_df=campaigns_df,
                    data_source=report_data_source)
            else:
                st.warning("اختاري تاريخ البداية والنهاية")
                result = None
        if result:
            if "error" in result:
                st.warning(f"⚠️ {result['error']}")
            else:
                st.session_state["last_report"] = result
    if "last_report" in st.session_state:
        result = st.session_state["last_report"]
        is_ar  = report_lang == "ar"
        # Generated on + Data header
        if result.get('data_header'):
            st.caption(result['data_header'])
        st.markdown(f"### 📊 {'تقرير الفترة' if is_ar else 'Report Period'}: {result['period_label']}")
        m           = result['metrics']
        rev_display = f"{float(m['total_revenue']):,.0f} SAR" if isinstance(m['total_revenue'], (int, float)) else str(m['total_revenue'])
        aov_display = f"{float(m['avg_order_value']):,.0f} SAR" if isinstance(m['avg_order_value'], (int, float)) else str(m['avg_order_value'])
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("إجمالي الإيرادات" if is_ar else "Total Revenue",    rev_display)
        with c2: st.metric("عدد الطلبات"       if is_ar else "Total Orders",     m['total_orders'])
        with c3: st.metric("متوسط قيمة الطلب"  if is_ar else "Avg Order Value",  aov_display)
        with c4: st.metric("العملاء الفريدون"   if is_ar else "Unique Customers", m['unique_customers'])
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            filtered = result['filtered_orders'].copy()
            filtered['order_date'] = pd.to_datetime(filtered['order_date'])
            filtered['month'] = filtered['order_date'].dt.to_period('M').astype(str)
            filtered = _ensure_total_price(filtered, products_df)
            monthly  = filtered[filtered['status'].isin(['completed','delivered'])].groupby('month')['total_price'].sum().reset_index()
            fig = px.line(monthly, x='month', y='total_price', markers=True,
                title="📈 الإيرادات الشهرية" if is_ar else "📈 Monthly Revenue")
            fig.update_traces(line_color='#1D9E75', marker=dict(color='#1D9E75', size=8))
            fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72', height=280)
            st.plotly_chart(fig, use_container_width=True)
        with col_c2:
            rev_cat = pd.DataFrame(m['revenue_by_category'])
            if not rev_cat.empty:
                fig = px.pie(rev_cat, values='total_price', names='category',
                    title="🍩 الإيرادات حسب الفئة" if is_ar else "🍩 Revenue by Category",
                    color_discrete_sequence=['#1D9E75','#378ADD','#D85A30','#BA7517','#9B59B6','#E67E22'])
                fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72', height=280)
                st.plotly_chart(fig, use_container_width=True)
        if 'top_products' in m:
            top_p = pd.DataFrame(m.get('top_products', []))
        elif products_df is not None:
            try:
                top_p = get_best_products(result['filtered_orders'], products_df)
            except Exception:
                top_p = pd.DataFrame()
        else:
            top_p = pd.DataFrame()
        if not top_p.empty and products_df is not None:
            price_col = 'price' if 'price' in products_df.columns else ('selling_price' if 'selling_price' in products_df.columns else None)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if price_col:
                    name_col  = 'product_name' if 'product_name' in products_df.columns else 'name'
                    price_map = products_df.set_index(name_col)[price_col]
                    top_p['revenue'] = top_p.apply(lambda r: r['quantity'] * price_map.get(r['product_name'], 0), axis=1)
                    fig_rev = px.bar(top_p.sort_values('revenue', ascending=True), x='revenue', y='product_name',
                        orientation='h',
                        title="💰 إيرادات أفضل المنتجات" if is_ar else "💰 Top Products by Revenue",
                        color='revenue', color_continuous_scale=['#9FE1CB','#1D9E75'])
                    fig_rev.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                        font_color='#5A7A72', height=280, coloraxis_showscale=False)
                    st.plotly_chart(fig_rev, use_container_width=True)
            with col_p2:
                fig_qty = px.bar(top_p.sort_values('quantity', ascending=True), x='quantity', y='product_name',
                    orientation='h',
                    title="📦 أفضل المنتجات حسب الكمية" if is_ar else "📦 Top Products by Quantity",
                    color='quantity', color_continuous_scale=['#9FE1CB','#1D9E75'])
                fig_qty.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                    font_color='#5A7A72', height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_qty, use_container_width=True)
        try:
            _, seg_summary = segment_customers(result['filtered_orders'], customers_df)
            col_seg1, col_seg2 = st.columns(2)
            with col_seg1:
                fig_seg = px.pie(seg_summary, values='count', names='segment_label',
                    title="👥 شرائح العملاء" if is_ar else "👥 Customer Segments",
                    color_discrete_map={'VIP':'#1D9E75','Regular':'#378ADD','Dormant':'#D85A30'})
                fig_seg.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72', height=280)
                st.plotly_chart(fig_seg, use_container_width=True)
            with col_seg2:
                if customers_df is not None and 'city' in customers_df.columns:
                    city_counts = customers_df['city'].value_counts().head(8).reset_index()
                    city_counts.columns = ['city', 'count']
                    fig_city = px.bar(city_counts.sort_values('count'), x='count', y='city',
                        orientation='h',
                        title="🏙️ توزيع العملاء حسب المدينة" if is_ar else "🏙️ Customers by City",
                        color='count', color_continuous_scale=['#9FE1CB','#1D9E75'])
                    fig_city.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', font_color='#5A7A72',
                        xaxis=dict(showgrid=True, gridcolor='#E8F4F0'), yaxis=dict(showgrid=False),
                        margin=dict(l=20, r=20, t=40, b=20), height=280, coloraxis_showscale=False)
                    st.plotly_chart(fig_city, use_container_width=True)
        except Exception:
            pass
        # Limitations
        if result.get('limitations'):
            with st.expander("⚠️ " + ("ملاحظات على البيانات" if is_ar else "Report Limitations")):
                for lim in result['limitations']:
                    st.caption(f"• {lim}")
        st.markdown('<div class="report-text-area">', unsafe_allow_html=True)
        if 'report_intro' in result:
            st.info(result['report_intro'])
        if 'report_part1' in result:
            st.markdown(f"### {'الجزء الأول' if is_ar else 'Part 1'}")
            st.markdown(result['report_part1'])
        if 'report_part2' in result:
            st.markdown(f"### {'الجزء الثاني' if is_ar else 'Part 2'}")
            st.markdown(result['report_part2'])
        if 'report_text' in result and 'report_part1' not in result:
            st.markdown(result['report_text'])
        st.markdown('</div>', unsafe_allow_html=True)
