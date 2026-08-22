# Project Visuals & Dashboards Directory

This directory centralizes all analytical visual assets, machine learning model evaluation charts, Power BI report screenshots, and Streamlit interactive web application screenshots for the **E-Commerce Intelligence Platform**.

---

## Directory Structure

```
visuals/
├── notebook_charts/           # Analytical charts & ML evaluation plots from Jupyter Notebook (44+ charts)
│   ├── delay_roc_curve.png
│   ├── forecast_actual_vs_predicted.png
│   ├── optimal_k_chart.png
│   ├── pareto_champions.png
│   ├── review_feature_importance.png
│   ├── segment_scatter.png
│   └── (38 extracted cell EDA plots)
│
├── power_bi_screenshots/     # High-resolution screenshots of Power BI dashboards
│   └── README.md
│
└── streamlit_screenshots/    # Screenshots of the interactive Streamlit Web App
    └── README.md
```

---

## Folders Guide

### 1. `notebook_charts/`
Contains all charts exported directly from `notebooks/ecommerce_intelligence_analysis.ipynb`, including:
- Customer RFM Segmentation & K-Means Clusters
- 80/20 Pareto Analysis (Champions vs. other segments)
- Delivery Delay Model ROC Curves & AUC Analysis
- Sales Time-Series Forecasting (Actual vs. Predicted)
- Review Rating Driver Feature Importances
- Full Exploratory Data Analysis (EDA) distributions, heatmaps, and boxplots

### 2. `power_bi_screenshots/`
Drop high-resolution screenshots of your Power BI report (`Supply Chain Analysis.pbix`):
- Executive Summary Dashboard
- Logistics & Shipping Performance
- Customer Behavior & Monetary Profiles
- Product & Category Financials

### 3. `streamlit_screenshots/`
Drop screenshots of the Streamlit interactive dashboard (`app.py`):
- KPI Overview & Dynamic Scorecards
- Customer Segmentation 3D/2D Visualizations & Strategy Playbook
- What-If Scenario Simulators for Delivery Delay & Customer Reviews
- Forecast Horizon Projections & Trends
