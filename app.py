import pickle as pkl
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)

ridge=pkl.load(open('models/ridge_regressor.pkl','rb'))
scaler=pkl.load(open('models/scaler2.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temp=float(request.form.get('Temperature'))
        rh=float(request.form.get('RH'))
        ws=float(request.form.get('Ws'))
        rain=float(request.form.get('Rain'))
        ffmc=float(request.form.get('FFMC'))
        dmc=float(request.form.get('DMC'))
        isi=float(request.form.get('ISI'))
        classes=float(request.form.get('Classes'))
        reg=float(request.form.get('Region'))

        new_scaled_data=scaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,classes,reg]])
        result=ridge.predict(new_scaled_data)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='localhost',port=5000)