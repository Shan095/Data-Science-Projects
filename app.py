# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)
model = load('catboost_model_30f.joblib')
features=model.feature_names_
print(features)

@app.route('/', methods=['GET', 'POST'])  # une méthode de recevoir les données, à travers le serveur
def pred_model():
    return render_template("test.html")

@app.route('/model', methods=['GET', 'POST'])
def predict():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('model.html')
    else:
        amount_credit = request.form['amt_credit']
        amt_annuity=request.form['amt_annuity']
        credit_length=request.form['credit_length']
        ext_source_2=request.form['ext_source_2']
        ext_source_3=request.form['ext_source_3']
        ext_source_1=request.form['ext_source_1']
        age=request.form['age']
        months_employed=request.form['months_employed']
        name_education_type=request.form['name_education_type']
        month_employed=request.form['months_employed']
        amt_credit=request.form['amt_credit']
        amt_goods_price=request.form['amt_goods_price']
        amt_annuity=request.form['amt_annuity']
        bureau_days_credit_max=request.form['bureau_days_credit_max']
        days_id_publish=request.form['days_id_publish']
        bureau_days_credit_enddate_max=request.form['bureau_days_credit_enddate_max']
        days_last_phone_change=request.form['days_last_phone_change']
        bureau_days_credit_mean=request.form['bureau_days_credit_mean']
        occupation_type=request.form['occupation_type']
        ratio_annuite_income=request.form['ratio_annuite_income']
        days_registration=request.form['days_registration']
        occupation_type=request.form['occupation_type']
        code_gender=request.form['code_gender']
        region_population_relative=request.form['region_population_relative']
        bureau_amt_credit_sum_debt_mean=request.form['bureau_amt_credit_sum_debt_mean']
        income_per_person=request.form['income_per_person']
        own_car_age=request.form['own_car_age']
        bureau_amt_credit_sum_mean=request.form['bureau_amt_credit_sum_mean']
        bureau_days_credit_sum=request.form['bureau_days_credit_sum']
        bureau_amt_credit_sum_debt_max=request.form['bureau_amt_credit_sum_debt_max']
        bureau_amt_credit_sum_max=request.form['bureau_amt_credit_sum_max']
        bureau_amt_credit_sum_min=request.form['bureau_amt_credit_sum_min']
        name_income_type=request.form['name_income_type']
        age=-float(age)*365
        months_employed=-float(month_employed)*30
        days_id_publish=-float(days_id_publish)
        features_input=[ext_source_2, ext_source_3, credit_length, ext_source_1, age, months_employed, name_education_type, amt_credit, amt_goods_price, amt_annuity, bureau_days_credit_max, days_id_publish, bureau_days_credit_enddate_max, days_last_phone_change, bureau_days_credit_mean, occupation_type, ratio_annuite_income, days_registration, code_gender, region_population_relative, bureau_amt_credit_sum_debt_mean, income_per_person, own_car_age, bureau_amt_credit_sum_mean, bureau_days_credit_sum, bureau_amt_credit_sum_debt_max, bureau_amt_credit_sum_max, bureau_amt_credit_sum_min, name_income_type]
        test_np_input = np.array([0.5718071677924758, 0.5262949398096192, 39.55696202531646,
                                       0.33796326232743, -12932, -2569.0, 'Higher education', 3150000.0,
                                      3150000.0, 79632.0, -259.0, -3828, 1204.0, -863.0, -1381.25,
                                      'Sales staff', 0.18243298969072164, -6555.0, 1, 0.025164, 0.0,
                                     436500.0, 3.0, 564068.25, -11050.0, 0.0, 931500.0, 0.0,
                                     'Commercial associate'])
        preds = model.predict_proba(features_input)
        preds_as_str = str(preds[1])
        return preds_as_str

@app.route('/birth', methods=['GET', 'POST'])
def show_age():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('model.html')
    else:
        age=request.form['age']
        make_picture('X_train_catboost.csv', age)
        return render_template('model.html')




y_train = pd.read_csv('y_train_catboost.csv', sep=',')


def make_picture(training_data_filename, np_arr):
    # Plot training data with model
    data = pd.read_csv(training_data_filename, sep=',')
    ages = -(data['days_birth'] / 365)
    ages = round(ages, 0)
    print(ages)
    income_per_person = data['income_per_person']
    income_per_person = round(income_per_person, 0)
    train_targets = y_train['target']
    fig = px.box(x=train_targets, y=ages, title="distribution of age by target", labels={'x': 'Target', 'y': 'Age'})
    fig.add_scatter(x=[0, 1], y=[np_arr,np_arr],mode="markers", marker=dict(size=20, color="LightSeaGreen"), showlegend=True)
    fig.show()

def make_picture_ratio(training_data_filename, np_arr):
    # Plot training data with model
    data = pd.read_csv(training_data_filename, sep=',')
    ages = -(data['days_birth'] / 365)
    ages = round(ages, 0)
    print(ages)
    income_per_person = data['income_per_person']
    income_per_person = round(income_per_person, 0)
    train_targets = y_train['target']
    fig = px.box(x=train_targets, y=ages, title="distribution of age by target", labels={'x': 'Target', 'y': 'Age'})
    fig.add_scatter(x=[0, 1], y=[np_arr,np_arr],mode="markers", marker=dict(size=20, color="LightSeaGreen"), showlegend=True)
    fig.show()

def floats_string_to_np_arr(s: str) -> float:
    try:
        floating = float(s)
    except ValueError:
        floating = [float(x) for x in s.split(',')]
    arr = np.array(floating)
    # arr = floating.reshape(-1, 1)
    print("my arr")
    print(arr)
    print(type(arr[0]))
    return arr


if __name__ == "__main__":
    app.run(debug=True)
