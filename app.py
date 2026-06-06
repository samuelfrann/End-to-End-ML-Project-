import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
CORS(app)

# ✅ Serve landing page
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Serve the prediction form (GET) + run prediction (POST)
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        data = request.get_json()
        custom_data = CustomData(
            policy_number=data.get('policy_number'),
            age=data.get('age'),
            umbrella_limit=data.get('umbrella_limit'),
            claim_amount=data.get('claim_amount'),
            policy_annual_premium=data.get('policy_annual_premium'),
            number_of_vehicles_involved=data.get('number_of_vehicles_involved'),
            incident_hour_of_the_day=data.get('incident_hour_of_the_day'),
            bodily_injuries=data.get('bodily_injuries'),
            witnesses=data.get('witnesses'),
            auto_year=data.get('auto_year'),
            policy_deductable=data.get('policy_deductable'),
            insured_sex=data.get('insured_sex'),
            insured_education_level=data.get('insured_education_level'),
            collision_type=data.get('collision_type'),
            police_report_available=data.get('police_report_available'),
            policy_state=data.get('policy_state'),
            policy_csl=data.get('policy_csl'),
            insured_occupation=data.get('insured_occupation'),
            incident_type=data.get('incident_type'),
            incident_severity=data.get('incident_severity'),
            authorities_contacted=data.get('authorities_contacted'),
            property_damage=data.get('property_damage')
        )
        pred_df = custom_data.get_data_as_data_frame()

        if 'policy_number' in pred_df.columns:
            pred_df = pred_df.drop(columns=['policy_number'])

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        prediction_label = "Fraud Detected" if results[0] == 1 else "No Fraud Detected"
        return jsonify({"result": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)