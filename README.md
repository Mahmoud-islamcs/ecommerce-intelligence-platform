# E-Commerce Supply Chain & Customer Intelligence Platform

An enterprise-grade, end-to-end data analytics and machine learning solution designed to optimize supply chain operations, enhance logistics performance, segment customer behavior, and forecast revenue. Built on the Brazilian E-Commerce public dataset (Olist), this platform covers the complete data analytics lifecycle: raw data preparation, relational exploratory analysis, RFM segmentation, predictive modeling, interactive Streamlit deployment, and executive Power BI dashboards.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Key Business Objectives](#key-business-objectives)
- [System Architecture](#system-architecture)
- [Dataset Architecture](#dataset-architecture)
- [Core Analytical Modules](#core-analytical-modules)
  - [1. Supply Chain & Logistics Analytics](#1-supply-chain--logistics-analytics)
  - [2. RFM Customer Segmentation & K-Means Clustering](#2-rfm-customer-segmentation--k-means-clustering)
  - [3. Predictive Machine Learning Models](#3-predictive-machine-learning-models)
- [Business Intelligence & Visualizations](#business-intelligence--visualizations)
  - [Power BI Executive Dashboard](#power-bi-executive-dashboard)
  - [Interactive Streamlit Web Application](#interactive-streamlit-web-application)
- [Project Directory Structure](#project-directory-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
- [Business Impact & Strategic Recommendations](#business-impact--strategic-recommendations)
- [Authors & Contact](#authors--contact)

---

## Executive Summary

Modern e-commerce platforms generate massive volumes of transactional, logistical, and customer satisfaction data. Without centralized intelligence, businesses suffer from delivery delays, inefficient customer retention, and inaccurate demand planning.

This project delivers an integrated business intelligence system that:
- Identifies critical bottlenecks across freight logistics and delivery timelines.
- Uncovers the 80/20 Pareto revenue concentration across customer segments.
- Deploys machine learning classifiers to predict delivery delays before fulfillment.
- Forecasts future sales revenue with confidence intervals to support inventory planning.
- Delivers an interactive decision-support interface via Streamlit and Power BI.

---

## Key Business Objectives

- **Logistics Optimization**: Quantify delivery duration, carrier lead times, and estimate accuracy across Brazilian states and metropolitan areas.
- **Customer Lifetime Value & Retention**: Segment customers via Recency, Frequency, and Monetary (RFM) modeling paired with unsupervised K-Means clustering.
- **Predictive Risk Mitigation**: Train supervised machine learning models to detect at-risk delayed shipments and estimate customer satisfaction scores.
- **Demand & Revenue Forecasting**: Formulate time-series regression models to project daily and monthly order volume and GMV.
- **Prescriptive Strategy Formulation**: Translate quantitative findings into actionable retention campaigns and logistics SLA improvements.

---

## System Architecture

```
[Raw Data: 9 Relational CSVs]
         │
         ▼
[Data Cleaning & Feature Engineering Pipeline]
         │
         ├──► [Cleaned Datasets (Data/cleaned/)]
         │           │
         │           ├──► [Power BI Data Model (Supply Chain Analysis.pbix)]
         │           │
         │           └──► [Streamlit Web Application (app.py)]
         │
         ▼
[Notebook Analysis & Model Training (notebooks/)]
         │
         ├──► RFM Customer Segmentation (K-Means, PCA)
         ├──► Delivery Delay Classifier (Random Forest / Logistic Regression)
         ├──► Review Rating Estimator (Gradient Boosting)
         └──► Sales Time-Series Forecaster (Regressors)
                     │
                     ▼
         [Serialized Models & Metadata (models/)]
                     │
                     ▼
         [Interactive Inference Engine (app.py)]
```

---

## Dataset Architecture

The project utilizes the Brazilian E-Commerce Public Dataset by Olist, comprising over 100,000 anonymized orders from 2016 to 2018 across Brazil:

| Table Name | Description | Key Attributes |
|:---|:---|:---|
| `customers_dataset` | Customer demographic records | `customer_id`, `customer_unique_id`, `customer_state`, `customer_city` |
| `orders_dataset` | Order lifecycle and status timestamps | `order_id`, `order_status`, `order_purchase_timestamp`, `order_delivered_customer_date` |
| `order_items_dataset` | Line-item product and freight data | `order_id`, `product_id`, `seller_id`, `price`, `freight_value` |
| `order_payments_dataset` | Payment method and installment records | `order_id`, `payment_type`, `payment_installments`, `payment_value` |
| `products_dataset` | Product physical dimensions and categories | `product_id`, `product_category_name`, `product_weight_g`, `product_photos_qty` |
| `sellers_dataset` | Seller geographic and fulfillment data | `seller_id`, `seller_city`, `seller_state`, `seller_zip_code_prefix` |
| `reviews_dataset` | Customer feedback and ratings | `review_id`, `order_id`, `review_score`, `review_creation_date` |
| `geolocation_dataset` | Latitude/longitude spatial data | `geolocation_zip_code_prefix`, `geolocation_lat`, `geolocation_lng` |

---

## Core Analytical Modules

### 1. Supply Chain & Logistics Analytics
- **Delivery Duration**: Computed exact fulfillment duration (`delivered_customer_date - purchase_timestamp`) and shipping duration.
- **Estimated vs. Actual SLAs**: Benchmarked delivery accuracy against promised carrier dates to calculate on-time delivery rates.
- **Geographic Disparities**: Visualized freight costs and transit durations across major economic regions (e.g., Southeast vs. North/Northeast states).

### 2. RFM Customer Segmentation & K-Means Clustering
- **Recency**: Days elapsed since customer's last completed purchase.
- **Frequency**: Total count of unique completed orders per unique customer.
- **Monetary Value**: Lifetime gross merchandise value (GMV) per customer.
- **Unsupervised Clustering**: Standardized log-transformed RFM vectors and applied K-Means with Elbow Method and Silhouette Score optimization ($K=4$).
- **Identified Segments**:
  1. **Champions / High Spenders**: 29.8% of customers accounting for 57.5% of total revenue.
  2. **Recent / Promising**: Highly active recent buyers with moderate spend.
  3. **Loyal / Repeat Buyers**: High frequency buyers with consistent engagement.
  4. **At Risk / Churned**: Low recency, dormant accounts requiring re-engagement.

### 3. Predictive Machine Learning Models

#### A. Delivery Delay Prediction (Classification)
- **Problem**: Predict whether an incoming order will arrive past the estimated delivery date.
- **Algorithms Evaluated**: Random Forest Classifier, Logistic Regression.
- **Key Features**: Distance between seller and customer, product weight, freight value, item count, freight-to-price ratio.
- **Performance**: ROC-AUC = 0.774, Precision-Recall balanced thresholds for operational risk alerts.

#### B. Customer Review Score Estimation (Regression)
- **Problem**: Estimate customer rating (1 to 5 stars) based on operational and product signals.
- **Algorithms Evaluated**: Gradient Boosting Regressor, Random Forest Regressor.
- **Key Drivers**: Delivery delay duration (largest negative impact), freight ratio, product description length.

#### C. Revenue & Demand Forecasting (Time-Series Regression)
- **Problem**: Forecast future daily and monthly gross sales revenue.
- **Methodology**: Lag features, rolling averages, seasonality indicators, and trend variables.
- **Outputs**: Baseline sales projections with upper and lower prediction intervals.

---

## Business Intelligence & Visualizations

### Power BI Executive Dashboard
The project includes a comprehensive Power BI analytical report (`Supply Chain Analysis.pbix`) structured into four core analytical layers:
- **Executive Summary**: High-level KPIs, total revenue, average order value (AOV), and gross order trends.
- **Supply Chain & Logistics**: Freight-to-order ratios, state-by-state transit timelines, and carrier SLA compliance.
- **Customer Intelligence**: RFM value matrices, customer distribution by state, and repeat purchase rates.
- **Product & Category Performance**: Top-performing categories by revenue and margin contribution.

### Interactive Streamlit Web Application
A multi-page production-grade web dashboard (`app.py`) featuring:
- **Executive KPIs**: Real-time business metrics, revenue split charts, and Pareto analysis.
- **RFM Analysis & Customer Explorer**: Interactive metric distribution filters, customer cohort tables, and download capabilities.
- **Cluster Profiling**: 3D and 2D PCA cluster visualizations, segment radar profiles, and tailored strategic playbooks.
- **Predictive Models & Simulators**: What-If interactive simulators allowing managers to input order variables and instantly evaluate delay risk probabilities and estimated review scores.

---

## Project Directory Structure

```
ecommerce-intelligence-platform/
│
├── Data/
│   ├── raw/                           # Original Olist raw CSV datasets
│   └── cleaned/                       # Preprocessed and engineered CSV datasets
│
├── models/                            # Trained machine learning model artifacts
│   ├── delivery_delay_model.joblib    # Random Forest delivery delay classifier
│   ├── delivery_metadata.joblib       # Feature names, scalers, and metadata
│   ├── forecast_model.joblib          # Time-series sales forecasting regressor
│   ├── forecast_metadata.joblib       # Forecasting lag and calendar parameters
│   └── review_score_model.joblib      # Customer satisfaction rating regressor
│
├── notebooks/
│   ├── ecommerce_intelligence_analysis.ipynb  # End-to-end data pipeline and research
│   ├── delay_roc_curve.png            # Model evaluation visual
│   ├── forecast_actual_vs_predicted.png
│   ├── optimal_k_chart.png
│   ├── pareto_champions.png
│   ├── review_feature_importance.png
│   └── segment_scatter.png
│
├── visuals/                           # Organized visual repository
│   ├── notebook_charts/               # 40+ exported analytical & EDA charts
│   ├── power_bi_screenshots/          # High-resolution Power BI dashboard views
│   ├── streamlit_screenshots/         # Streamlit web application interface views
│   └── README.md                      # Detailed catalog of all project visuals
│
├── .streamlit/
│   └── config.toml                    # Streamlit server and theme configuration
│
├── app.py                             # Main Streamlit web application
├── export_visuals.py                  # Automated visual extractor script
├── Presentation - E-Commerce Supply Chain Intelligence.pdf  # Executive slide deck
├── Supply Chain Analysis.pbix         # Interactive Power BI analytical report
├── requirements.txt                   # Python environment dependencies
└── README.md                          # Project documentation
```

---

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mahmoud-islamcs/ecommerce-intelligence-platform.git
   cd ecommerce-intelligence-platform
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

### Running the Streamlit Web Application
Launch the local web dashboard:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

### Exporting and Refreshing Visual Assets
To re-extract all embedded notebook plots and organize charts into the `visuals/` directory:
```bash
python export_visuals.py
```

### Viewing the Power BI Report
Open `Supply Chain Analysis.pbix` using **Power BI Desktop** to explore the data models, DAX measures, and interactive report pages.

---

## Business Impact & Strategic Recommendations

1. **Logistics SLA Re-alignment**: Re-calibrate estimated delivery algorithms for Northern and Northeastern regions to reduce artificial delay penalties.
2. **High-Value Retention**: Deploy dedicated VIP retention workflows and loyalty incentives for the Champions segment (responsible for over 57% of platform revenue).
3. **Proactive Delay Alerting**: Integrate the Delivery Delay Classifier into the order dispatch pipeline to notify customer support teams when delay probability exceeds 65%.
4. **Freight Subsidy Targeting**: Re-evaluate freight costs on low-margin, high-weight items to boost conversion and average order ratings.

---

## Authors & Contact

Developed as an advanced end-to-end data analytics and intelligence initiative:

- **Mahmoud Islam** — [LinkedIn](https://www.linkedin.com/in/mahmoud-islam-analytics/) | [GitHub](https://github.com/Mahmoud-islamcs)
- **Mina Gabra**

For feedback, questions, or collaboration inquiries, please open an issue or reach out via LinkedIn.
