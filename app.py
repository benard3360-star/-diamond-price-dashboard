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

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
