# input libraries
import flask
from flask import Flask,request,jsonify
import numpy as np
import pickle
import pandas as pd


# Load the trained model which saved

clf = pickle.load(open('crop-model.model.pkl', 'rb'))

#model_col = pickle.load(open('crop-model.model_col.pkl', 'rb'))

# Make an instance of flask api from flask-restful

app = Flask(__name__)

@app.route('/')

def index():
    return "Hello World"

@app.route('/predict',methods=['POST'])

# fetch attributes from the user inputs

def predict():

    Humidity = request.form.get('Humidity')
    Temperature = request.form.get('Temperature')
    SoilMoisture = request.form.get('SoilMoisture')
    PH = request.form.get('PH')
    Waterlevel = request.form.get('Waterlevel')
    Space = request.form.get('Space')

    data = []
    data.append([Humidity,Temperature,SoilMoisture,PH,Waterlevel,Space])

    prediction = clf.predict(data)[0]


    #print the result
    return jsonify({

            "suitable crop is": str(prediction)

        })



    if __name__ == '__main__':
        app.run(host= '0.0.0.0')

