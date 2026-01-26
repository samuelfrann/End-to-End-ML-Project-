from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            policy_number = request.form.get('policy_number'),
            age = request.form.get('age'),
            umbrella_limit = request.form.get('umbrella_limit'),
            claim_amount = request.form.get('claim_amount'),
            policy_annual_premium = request.form.get('policy_annual_premium'),
            number_of_vehicles_involved = request.form.get('number_of_vehicles_involved'),
            incident_hour_of_the_day = request.form.get('incident_hour_of_the_day'),
            bodily_injuries = request.form.get('bodily_injuries'),
            witnesses = request.form.get('witnesses'),
            auto_year = request.form.get('auto_year'),
            policy_deductable = request.form.get('policy_deductable'),
            insured_sex = request.form.get('insured_sex'),
            insured_education_level = request.form.get('insured_education_level'),
            collision_type = request.form.get('collision_type'),
            police_report_available = request.form.get('police_report_available'),
            policy_state = request.form.get('policy_state'),
            policy_csl = request.form.get('policy_csl'),
            insured_occupation = request.form.get('insured_occupation'),
            incident_type = request.form.get('incident_type'),
            incident_severity = request.form.get('incident_severity'),
            authorities_contacted = request.form.get('authorities_contacted'),
            property_damage = request.form.get('property_damage')
        )

        pred_df = data.get_data_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
