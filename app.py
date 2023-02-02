from flask import Flask, render_template , request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("Car_price.pkl",'rb'))

car=pd.read_csv('Cleaned data.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique())
    fuel_type = sorted(car['fuel_type'].unique())

    return render_template('index.html',companies = companies,car_models = car_models, year = year,fuel_type=fuel_type)
    

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_models = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    km = int(request.form.get('km'))

    final_value = model.predict(pd.DataFrame([[car_models,company,year,km,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    
    #print(final_value)


    return str(np.round(final_value[0],2))

if __name__ == "__main__":
    app.run(debug = True)