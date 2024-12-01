from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models and encoders
with open('robust_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('target_encoder.pkl', 'rb') as file:
    target_encoder = pickle.load(file)

with open('rfe_model.pkl', 'rb') as file:
    rfe_model = pickle.load(file)

with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Feature list for proper input mapping
selected_features = ['Age', 'Items Purchased', 'Total Spent', 'Discount (%)',
       'Satisfaction Score', 'Revenue']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON data
    data = request.json
    try:
        # Convert to DataFrame for processing
        input_data = pd.DataFrame([data])

        # Scale numerical features
        input_data[['Total Spent', 'Discount (%)', 'Revenue']] = scaler.transform(
            input_data[['Total Spent', 'Discount (%)', 'Revenue']]
        )

        # One-hot encode categorical features
        categorical_data = input_data[['Gender', 'Region', 'Product Category', 'Preferred Visit Time', 'Payment Method']]
        encoded_categorical = encoder.transform(categorical_data)
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical, 
            columns=encoder.get_feature_names_out(categorical_data.columns)
        )

        # Combine scaled and encoded features
        input_data = pd.concat([input_data.drop(columns=categorical_data.columns), encoded_categorical_df], axis=1)

        # Select RFE-selected features
        input_data = input_data[selected_features]

        # Make prediction
        prediction = best_model.predict(input_data)
        category = target_encoder.inverse_transform(prediction)[0]

        return jsonify({'Loyalty Category': category})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
