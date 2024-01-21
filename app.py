from flask import Flask, jsonify, request
from flask_pymongo import PyMongo 
from flask_cors import CORS
from flask_caching import Cache
from bson import json_util, ObjectId

import os
from dotenv import load_dotenv

import joblib
import numpy as np

load_dotenv()

app = Flask(__name__)
app.config['MONGO_URI'] = os.getenv('MONGO_URL')

mongo = PyMongo(app)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})
CORS(app)


@app.route('/api/offline', methods=['GET'])
def offline_data():
    return cache.get('/api/all')

@app.route('/api/all', methods=['GET'])
@cache.cached(timeout=60)
def dataall():
    dataall = []
    doc = mongo.db.data.find_one(sort=[('_id', -1)])
    if doc:
        doc_dict = json_util.loads(json_util.dumps(doc))
        dataall.append({
            '_id': str(doc_dict['_id']),
            'Rain': doc_dict['Rain '],
            'Temperature': doc_dict['Temperature '],
            'RH': doc_dict['RH '],
            'Dew_Point': doc_dict['Dew Point'],
            'Wind_Speed': doc_dict['Wind Speed '],
            'Gust_Speed': doc_dict['Gust Speed '],
            'Wind_Direction': doc_dict['Wind Direction '],
            'Date': doc_dict['Date']
        })
    return jsonify(dataall)

@app.route('/api/predict', methods=['POST'])
def dataarray():
    data = request.get_json()
    planta = float(data['PLANTA'])
    fruto = float(data['FRUTO'])
    Dew_Point = float(data['Dew_Point'])
    Gust_Speed = float(data['Gust_Speed'])
    RH = float(data['RH'])
    Rain = float(data['Rain'])
    Temperature = float(data['Temperature'])
    Wind_Direction = float(data['Wind_Direction'])
    Wind_Speed = float(data['Wind_Speed'])
    severidad = float(data['SEVERIDAD'])

    dataarray = []   

    dataarray.append(
        [
            Rain,
            Dew_Point,
            Temperature,
            RH,
            Wind_Speed,
            Gust_Speed,
            Wind_Direction,
            planta,
            fruto,
            severidad,
        ]
    )
    model = joblib.load("model/Abeldb.pkl")
    X_test = np.array(dataarray)
    prediction = model.predict(X_test).tolist()
    datainsert = dataarray[0] + [prediction[0]]
    mongo.db.Predictions.insert_one({
        'Rain': datainsert[0],
        'Temperature': datainsert[1],
        'RH': datainsert[2],
        'Dew_Point': datainsert[3],
        'Wind_Speed': datainsert[4],
        'Gust_Speed': datainsert[5],
        'Wind_Direction':datainsert[6],
        'planta': datainsert[7],
        'fruto': datainsert[8],
        'severidad': datainsert[9],
        'incidencia': datainsert[10],       
    })
    if (prediction[0] == 1):
        return jsonify('Cultivo infectado')
    else:
        return jsonify('Cultivo sano')

@app.route('/api/allp', methods=['GET'])
def dataallp():
    dataall = []
    doc = mongo.db.data.find_one(sort=[('_id', -1)])
    if doc:
        doc_dict = json_util.loads(json_util.dumps(doc))
        dataall.append({
            '_id': str(doc_dict['_id']),
            'Rain': doc_dict['Rain '],
            'Temperature': doc_dict['Temperature '],
            'RH': doc_dict['RH '],
            'Dew_Point': doc_dict['Dew Point'],
            'Wind_Speed': doc_dict['Wind Speed '],
            'Gust_Speed': doc_dict['Gust Speed '],
            'Wind_Direction': doc_dict['Wind Direction '],
            'planta': doc_dict['PLANTA'],
            'fruto': doc_dict['FRUTO'],
            'incidencia': doc_dict['INCIDENCIA'],
            'severidad': doc_dict['SEVERIDAD'],
            'Date': doc_dict['Date']
        })
    return jsonify(dataall)

@app.route('/api/allpredict', methods=['GET'])
def allpredict():
    predictions = mongo.db.Predictions.find()
    prediction_list = []
    
    for prediction in predictions:
        prediction_data = {
            'Rain': prediction['Rain'],
            'Temperature': prediction['Temperature'],
            'RH': prediction['RH'],
            'Dew_Point': prediction['Dew_Point'],
            'Wind_Speed': prediction['Wind_Speed'],
            'Gust_Speed': prediction['Gust_Speed'],
            'Wind_Direction': prediction['Wind_Direction'],
            'planta': prediction['planta'],
            'fruto': prediction['fruto'],
            'severidad': prediction['severidad'],
            'incidencia': prediction['incidencia']
        }
        prediction_list.append(prediction_data)
    
    return jsonify(prediction_list)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)