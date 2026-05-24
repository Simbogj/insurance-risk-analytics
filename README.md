# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This project focuses on end-to-end insurance risk analytics for AlphaCare Insurance Solutions (ACIS), using historical auto insurance data from South Africa.

The goal is to analyze historical insurance claims, identify low-risk customer segments, and build reproducible analytics pipelines that support risk-based pricing and strategic decision-making.

The project includes:

- Exploratory Data Analysis (EDA)
- Insurance risk diagnostics
- Loss ratio analysis
- Geographic and demographic risk segmentation
- Data Version Control (DVC)
- Reproducible analytics workflows
- Predictive modeling (upcoming tasks)

---

# Business Objective

AlphaCare Insurance Solutions (ACIS) aims to improve pricing accuracy and optimize marketing strategy using data-driven insurance analytics.

The analysis focuses on:

- Understanding historical claim behavior
- Identifying high-risk and low-risk segments
- Measuring portfolio profitability using Loss Ratio
- Supporting future predictive pricing models

---

# Dataset

Dataset used:

```text
data/MashineLearningRating_v3.txt
```

The dataset contains insurance policy, customer, vehicle, premium, and claim information.

Key variables include:

- TotalPremium
- TotalClaims
- Province
- VehicleType
- Gender
- PostalCode
- TransactionMonth
- CustomValueEstimate

---

# Project Structure

```text
insurance-risk-analytics/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── MashineLearningRating_v3.txt.dvc
│   └── cleaned_insurance_data.csv.dvc
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_data_cleaning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── eda_utils.py
│
├── .dvc/
├── .dvcignore
├── dvc.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Task 1 — Exploratory Data Analysis (EDA)

## Objectives

- Understand insurance portfolio performance
- Assess data quality
- Analyze claim distributions
- Identify geographic and demographic risk patterns
- Compute Loss Ratio and Claim Frequency

## Key Analyses Performed

### Data Summarization
- Descriptive statistics
- Data type inspection
- Numerical and categorical analysis

### Data Quality Assessment
- Missing value analysis
- Invalid and zero-premium policy handling

### Univariate Analysis
- Premium distributions
- Claim distributions
- Vehicle and province distributions

### Bivariate / Multivariate Analysis
- Loss Ratio by:
  - Province
  - VehicleType
  - Gender
- Correlation analysis
- Scatter plots and trend analysis

### Geographic Risk Analysis
- Province-level profitability comparison
- Vehicle risk segmentation

### Temporal Analysis
- Monthly claim trends
- Claim severity evolution over time

---

# Key Findings

- The insurance portfolio recorded a Loss Ratio above 100%, indicating underwriting losses.
- Certain provinces and vehicle categories exhibit significantly higher risk profiles.
- Claim frequency is relatively low, but claim severity is high.
- Financial variables are highly skewed with notable outliers.

---

# Task 2 — Data Version Control (DVC)

## Objective

Implement reproducible and auditable data pipelines using DVC.

## DVC Workflow

### Initialize DVC

```bash
dvc init
```

### Track Raw Dataset

```bash
dvc add data/MashineLearningRating_v3.txt
```

### Track Cleaned Dataset

```bash
dvc add data/cleaned_insurance_data.csv
```

### Push Data to Local Remote Storage

```bash
dvc push
```

---

# Reproducing the Project

## 1. Clone Repository

```bash
git clone https://github.com/Simbogj/insurance-risk-analytics.git
cd insurance-risk-analytics
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Pull Dataset Using DVC

```bash
dvc pull
```

This restores all tracked datasets locally.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Git & GitHub
- GitHub Actions
- DVC (Data Version Control)

---

# Future Work

Upcoming tasks include:

- Statistical hypothesis testing
- A/B testing
- Predictive modeling
- Risk-based pricing
- SHAP/LIME model interpretability

---

# Author

Simbo Getachew