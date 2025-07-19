# Diamond Price Monitoring & Forecasting Dashboard

![Logo](diamond_app/static/logo.jpg)

## Overview
This project is an **AI-powered dashboard for monitoring, forecasting, and analyzing diamond prices in Africa**. Built with a hybrid machine learning model (XGBoost + Gradient Boosting), it provides real-time insights, price predictions, volatility analysis, and authenticity tracking for stakeholders in the diamond industry.

## Features
- **Historical & Predicted Prices:** Visualize trends and forecast future prices.
- **Volatility Metrics:** Analyze causes and trends in price volatility.
- **Major Producers & Consumers:** Track top African producers and global consumers.
- **Industry Insights:** Explore uses, processing industries, and challenges.
- **Authenticity Tracking:** Learn about verification methods and emerging technologies.
- **Interactive Dashboard:** User-friendly, visually appealing, and responsive.

## Screenshots
> _Add screenshots of your dashboard here_

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/diamond-price-dashboard.git
   cd diamond-price-dashboard
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app locally:**
   ```bash
   python -m diamond_app.app
   ```
   Or:
   ```bash
   export FLASK_APP=diamond_app/app.py
   flask run
   ```

### Deployment
- **Render/Heroku:**
  - Add a `Procfile` with:
    ```
    web: gunicorn diamond_app.app:app
    ```
  - Deploy via Render, Heroku, or similar platforms.

### Project Structure
```
diamond_app/
  app.py
  hybrid_model.py
  Hybrid_model.pkl
  label_encoders.pkl
  top_10_features.pkl
  static/
    logo.jpg
    diamond_bg.jpg
  templates/
    ...
README.md
requirements.txt
```

## About the Model
- **Hybrid Model:** Combines XGBoost and Gradient Boosting for highly accurate price prediction (R² Score: 0.9937, RMSE: 0.0807).
- **Focus:** Tailored for African diamond markets, addressing unique volatility and authenticity challenges.

## Value to Stakeholders
- **Consumers:** Make informed, fair, and authentic purchases.
- **Producers:** Optimize production and manage risk.
- **Policymakers/Investors:** Support evidence-based policy and sustainable growth.

## Contact
For questions or collaboration, contact [Your Name] at [your.email@example.com]. 
