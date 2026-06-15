# Known Limitations — Actionlytics

This document explains the current limitations of Actionlytics honestly and professionally.

Actionlytics is a strong Applied AI portfolio project, but it is not presented as a finished production SaaS platform. The limitations below show what works today, what is intentionally simplified, and what should be improved before production.

---

## 1. Portfolio / Prototype Scope

Actionlytics is currently designed as a portfolio and prototype system.

It demonstrates analytics, ML, RAG, LLM integration, smart recommendations, report generation, and data upload workflows.

However, it does not yet include production SaaS features such as authentication, billing, user roles, enterprise security, or multi-tenant isolation.

**Future improvement:** Add authentication, secure user-level data isolation, access control, and production deployment infrastructure.

---

## 2. SQLite Limitation

The project uses SQLite for simplicity and portability.

SQLite is suitable for local development, demos, and small datasets, but it is not ideal for many concurrent users, large-scale multi-tenant workloads, or heavy analytics traffic.

**Future improvement:** Migrate to PostgreSQL with indexing, migrations, and tenant-aware data design.

---

## 3. Forecasting Limitations

The forecasting layer uses classical regression models with time and seasonal features.

It includes time-based backtesting, MAE, RMSE, MAPE, and a reliability estimate. However, forecasts are still limited by available historical data and do not yet include external signals such as holidays, stockouts, campaign spend, promotions, or competitor activity.

The reliability score is an estimate derived from model error, not a statistical confidence interval.

**Future improvement:** Add richer external features, holiday calendars, promotion signals, longer historical data, and confidence intervals.

---

## 4. Product Demand Forecasting Limitation

Product-level demand forecasting uses time-based backtesting when enough monthly history exists. If a product has limited history, the system falls back to a simpler reliability estimate and clearly labels it as limited.

**Future improvement:** Add stronger minimum-data rules, category-level fallback models, stockout awareness, and product-level seasonal features.

---

## 5. Customer Segmentation Limitation

Customer segmentation uses KMeans with behavioral features and maps clusters into VIP, Regular, and Dormant.

This is useful and interpretable, but it assumes three customer groups and may not capture more complex customer behavior.

**Future improvement:** Use silhouette score or elbow method, add category preferences, discount sensitivity, and dynamic segment counts.

---

## 6. Repeat Purchase Rate Limitation

Repeat purchase rates above 80% are flagged as statistically unusual and may indicate synthetic or demo data patterns rather than real customer behavior.

Real-world e-commerce repeat purchase rates typically range between 20–40% depending on the product category.

**Future improvement:** Add benchmark comparison directly in the analytics view to contextualize repeat purchase rates.

---

## 7. Churn Detection Limitation

Churn detection is behavior-based, not supervised classification.

It uses average purchase gap and days since last order. This is appropriate because the project does not assume labeled churn data, but it is not a trained churn classifier.

**Future improvement:** If labeled churn data becomes available, train and evaluate a supervised churn classification model.

---

## 8. Market Basket Limitation

Market Basket Analysis depends on enough multi-product orders. If most orders contain only one product, association rules may be weak or unavailable.

**Future improvement:** Use larger transaction datasets and category-level association analysis when product-level patterns are sparse.

---

## 9. Anomaly Detection Limitation

Anomaly detection uses Isolation Forest on monthly revenue.

It can detect unusual spikes and drops, but it does not fully understand campaign effects, holidays, stockouts, seasonality, or external business events.

**Future improvement:** Add multivariate anomaly detection using revenue, orders, conversion rate, campaign spend, and seasonal baselines.

---

## 10. RAG Knowledge Base Limitation

The RAG layer uses a static Markdown knowledge base.

It works for benchmarks and best practices, but knowledge is not automatically updated. Retrieval evaluation reached strong accuracy on the current test set, but there is a known edge case where Arabic questions mixing multiple category meanings may retrieve a related but not exact category.

**Future improvement:** Add metadata-aware retrieval, a larger evaluation set, source freshness tracking, and dynamic knowledge updates.

---

## 11. Similarity Threshold Limitation

FAISS returns a distance-like score, which is converted into an approximate similarity value. The threshold reduces irrelevant retrieval, but it is heuristic and not a calibrated probability.

**Future improvement:** Tune thresholds using a larger evaluation dataset and compare multiple embedding models.

---

## 12. LLM Reliability Limitation

Claude is used for natural language generation, reports, and messages. The system reduces hallucination risk by calculating metrics in Python, using structured context, applying RAG conditionally, and adding fallback behavior.

However, LLM outputs may still vary in wording and may sometimes fail to perfectly follow a requested format.

**Future improvement:** Add stronger schema validation, retry logic, output parsers, and automated response tests.

---

## 13. API Cost Transparency

Approximate API usage per operation:

- Report generation: 2 Claude Opus 4.5 calls (~6,000 tokens total)
- Chatbot: 1 Claude Sonnet 4.5 call per question
- Smart Action messages: 1 Claude Haiku call per customer (on-demand only)

Messages are generated lazily — only when explicitly requested — to minimize unnecessary API usage.

---

## 14. Action Engine Limitation

The Action Engine produces structured recommendations with impact, urgency, priority, evidence, and expected outcome.

However, it does not execute actions automatically. It does not send emails, WhatsApp messages, purchase orders, or campaign changes.

This is intentional because the current design follows a human-in-the-loop approach.

**Future improvement:** Add an approval workflow: Generated → Reviewed → Approved → Executed → Tracked.

---

## 15. VIP Grouping Thresholds Limitation

VIP grouping thresholds are currently fixed values:

- Premium: above 5,000 SAR
- Regular: 2,000–5,000 SAR
- Basic: under 2,000 SAR

These thresholds are not auto-calibrated to store size or average order value, which may make them less meaningful for stores with very different revenue scales.

**Future improvement:** Dynamic threshold calibration based on store percentiles.

---

## 16. Discount Logic Limitation

Discount recommendations are currently rule-based and based on recency, frequency, and monetary value.

They do not yet account for product margins, customer lifetime value, discount sensitivity, or inventory constraints.

**Future improvement:** Add margin-aware and CLV-aware discounting before enabling automated promotional workflows.

---

## 17. Campaign ROI Limitation

Campaign ROI is calculated only when campaign revenue and budget data are available. The system avoids inventing ROI when revenue tracking is missing.

However, attribution is simplified and does not include multi-touch attribution, attribution windows, or assisted conversions.

**Future improvement:** Add stronger attribution modeling and integrations with marketing platforms.

---

## 18. Report Engine Limitation

Reports are historical only. Forecasting is intentionally excluded to avoid mixing historical analysis with future predictions.

Some sections may be limited or skipped when required data is missing.

**Future improvement:** Add multiple report templates such as Executive, Marketing, Inventory, Customer, and Forecast reports.

---

## 19. Upload Mapping Limitation

Column mapping uses Claude to map uploaded CSV/Excel columns to the expected schema. This supports Arabic and English columns, but automated mapping can still be imperfect.

Manual correction notes are supported, but confidence scores are not yet shown.

Large files above 50,000 rows may cause slow processing. Recommended maximum is 10,000 rows per file for optimal performance.

**Future improvement:** Add mapping confidence scores, stronger validation UI, file size warnings, and required-field checks before saving.

---

## 20. Data Quality Limitation

The system includes cleaning and validation, but real business data can still be messy.

Examples include missing IDs, inconsistent product names, invalid dates, incorrect prices, and incomplete campaign data.

**Future improvement:** Add a full data quality dashboard showing missing values, invalid rows, duplicate rates, and schema coverage.

---

## 21. Model Registry Limitation

The model registry is currently stored in a JSON file. This is useful for portfolio-level model versioning, but it is not production-grade and can grow over time. During local testing, registry entries accumulate with each forecast run.

**Future improvement:** Use MLflow or a database-backed registry for tracking experiments, parameters, metrics, and model artifacts.

---

## 22. Monitoring Limitation

The project does not yet include production model monitoring because it is currently a portfolio/demo application and does not continuously receive real-world prediction feedback.

**Future improvement:** Track prediction vs actual outcomes, monitor MAE/MAPE drift, detect data drift, and trigger retraining alerts.

---

## 23. Deployment Limitation

The project can run locally or on a simple Streamlit deployment, but it does not yet include Docker, CI/CD, cloud database, production logging, or monitoring dashboards.

**Future improvement:** Deploy first using Streamlit Cloud or a cloud VM, then later add Docker, CI/CD, PostgreSQL, and monitoring.

---

## 24. Demo Environment Limitation

The public demo environment is intended for portfolio review, technical evaluation, and interview discussions.

To control infrastructure and API costs, the hosted demo may apply usage limits and is shared selectively upon request.

**Future improvement:** Move to a production-grade deployment with scalable infrastructure, monitoring, and controlled access management.

---

## Summary

Actionlytics demonstrates strong end-to-end Applied AI thinking:

- data processing
- analytics
- machine learning
- RAG
- LLM grounding
- context engineering
- smart recommendations
- report generation
- honest limitation handling

The next production step would be improving reliability, data governance, deployment, monitoring, and real user feedback loops.
