#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:11:05 2018

@author: Steven
"""

from flask import Flask, jsonify, request
import pickle
import numpy as np
import json

filename = '../Model/Iris_model_v1.pkl'
iris_model = pickle.load(open(filename, 'rb'))

# create instance of flask app
app = Flask(__name__)
app.debug = True


@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json(force=True)
    
    predict_request = [data['sl'], data['sw'], data['pl'], data['pw']]
    predict_request = np.array(predict_request).reshape(1, -1)
    
    y_pred = iris_model.predict(predict_request)
    
    output = str([y_pred[0]])
    
    return jsonify(results=output)


# if this condition met this script will be executed
if __name__ == '__main__':

	app.secret_key = 'secret123'
	app.run()

    
import requests
    
    
    
    
    
    
    
    
    
    