from flask import Flask, render_template, request
from hybrid_model import HybridRegressor  # correct import
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# === Load model and artifacts ===
with open('Hybrid_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('top_10_features.pkl', 'rb') as f:
    top_10_features = pickle.load(f)

# === Identify categorical features (present in encoders) ===
categorical_features = [col for col in top_10_features if col in encoders]

dropdown_options = {
    feature: list(encoders[feature].inverse_transform(range(len(encoders[feature].classes_))))
    for feature in categorical_features
}

# === Mock/sample data for dashboard sections ===
historical_prices = [
    {"year": 2018, "price": 1000},
    {"year": 2019, "price": 1100},
    {"year": 2020, "price": 950},
    {"year": 2021, "price": 1200},
    {"year": 2022, "price": 1300},
    {"year": 2023, "price": 1250},
]

# Top African diamond producers (mock data)
producers = [
    {"country": "Botswana", "production": 23000, "value": 4000},
    {"country": "DR Congo", "production": 18000, "value": 3200},
    {"country": "Angola", "production": 12000, "value": 2500},
    {"country": "South Africa", "production": 9000, "value": 2000},
    {"country": "Zimbabwe", "production": 4000, "value": 900},
    {"country": "Namibia", "production": 2000, "value": 800},
    {"country": "Lesotho", "production": 1200, "value": 500},
    {"country": "Sierra Leone", "production": 700, "value": 300},
    {"country": "Tanzania", "production": 500, "value": 200},
    {"country": "Central African Republic", "production": 400, "value": 150},
]

consumers = [
    {"country": "USA", "consumption": 5000},
    {"country": "China", "consumption": 3000},
    {"country": "India", "consumption": 2500},
    {"country": "Europe", "consumption": 2000},
]

industry_insights = [
    {"industry": "Jewelry", "use": "Luxury goods, engagement rings"},
    {"industry": "Industrial", "use": "Cutting, grinding, drilling"},
    {"industry": "Technology", "use": "Semiconductors, optics"},
]

volatility_causes = [
    "Political instability in producing countries",
    "Global economic downturns",
    "Synthetic diamond competition",
    "Supply chain disruptions",
    "Changing consumer preferences"
]

challenges = [
    "High cost of extraction",
    "Illicit trade and smuggling",
    "Environmental impact",
    "Price manipulation",
    "Authenticity verification"
]

authenticity_methods = [
    "Laser inscription",
    "Certification (GIA, IGI, etc.)",
    "Blockchain tracking",
    "Spectroscopic analysis"
]

# === Routes for dashboard pages ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prices', methods=['GET', 'POST'])
def prices():
    prediction = None
    if request.method == 'POST':
        input_data = {}
        for feature in top_10_features:
            value = request.form.get(feature)
            if feature in categorical_features:
                value = encoders[feature].transform([value])[0]
            else:
                value = float(value)
            input_data[feature] = value
        input_df = pd.DataFrame([input_data])
        log_price = model.predict(input_df)[0]
        price = np.expm1(log_price)
        prediction = round(price, 2)
    return render_template('prices.html', historical_prices=historical_prices, dropdown_options=dropdown_options, top_10_features=top_10_features, prediction=prediction)

@app.route('/producers')
def producers_page():
    return render_template('producers.html', producers=producers, consumers=consumers)

@app.route('/industry')
def industry():
    return render_template('industry.html', industry_insights=industry_insights, challenges=challenges)

@app.route('/volatility')
def volatility():
    return render_template('volatility.html', volatility_causes=volatility_causes, historical_prices=historical_prices)

@app.route('/authenticity')
def authenticity():
    return render_template('authenticity.html', authenticity_methods=authenticity_methods)

@app.route('/features')
def features():
    # OLS summary and feature economic importance (to be shown in template)
    ols_summary = '''
    Final Linear Regression Model Summary (Using Stepwise Features):
                               OLS Regression Results                            
==============================================================================
Dep. Variable:              log_price   R-squared:                       0.869
Model:                            OLS   Adj. R-squared:                  0.869
Method:                 Least Squares   F-statistic:                 4.467e+04
Date:                Sat, 19 Jul 2025   Prob (F-statistic):               0.00
Time:                        09:56:37   Log-Likelihood:                -22474.
No. Observations:               53919   AIC:                         4.497e+04
Df Residuals:                   53910   BIC:                         4.505e+04
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
const              6.2616      0.078     80.759      0.000       6.110       6.414
carat              1.9931      0.018    113.129      0.000       1.959       2.028
color             -0.0758      0.001    -77.931      0.000      -0.078      -0.074
clarity            0.0499      0.001     52.942      0.000       0.048       0.052
table             -0.0057      0.001     -7.553      0.000      -0.007      -0.004
depth_pct_calc    -0.3294      0.057     -5.792      0.000      -0.441      -0.218
carat_per_mm3     62.4631     10.040      6.221      0.000      42.784      82.142
volume             0.0006      0.000      5.716      0.000       0.000       0.001
cut                0.0047      0.002      3.011      0.003       0.002       0.008
==============================================================================
Omnibus:                    13528.444   Durbin-Watson:                   0.903
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            89395.207
Skew:                          -1.044   Prob(JB):                         0.00
Kurtosis:                       8.952   Cond. No.                     1.01e+06
==============================================================================
    '''
    features_list = [
        {"name": "carat", "desc": "Carat weight is the most direct measure of a diamond's size and a primary driver of price. Larger diamonds are rarer and command higher prices."},
        {"name": "color", "desc": "Diamond color (graded D-Z) affects value, with colorless stones being more valuable. Color is a key quality attribute in pricing."},
        {"name": "clarity", "desc": "Clarity measures internal and external flaws. Higher clarity increases value, as flawless diamonds are rare."},
        {"name": "table", "desc": "The table is the largest facet. Its size impacts brilliance and market value, with optimal proportions preferred by buyers."},
        {"name": "depth_pct_calc", "desc": "Depth percentage affects how light travels through the diamond, influencing sparkle and thus price."},
        {"name": "carat_per_mm3", "desc": "Carat per cubic millimeter reflects density and cut quality, impacting perceived value and price."},
        {"name": "volume", "desc": "Volume relates to the diamond's physical size, which, along with carat, influences price."},
        {"name": "cut", "desc": "Cut quality determines brilliance and fire, making it a major economic factor in diamond pricing."},
    ]
    return render_template('features.html', ols_summary=ols_summary, features_list=features_list)

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
